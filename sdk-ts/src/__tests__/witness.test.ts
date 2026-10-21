import { GolemSDK, PreExecutionBOM } from '../sdk';
import * as http from 'http';
import * as crypto from 'crypto';
import * as cbor from 'cbor';

const sampleBOM: PreExecutionBOM = {
  agent_id: 'test-ts-agent',
  mcp_server_id: 'test-server',
  tool_name: 'read_file',
  tool_arguments: { path: '/etc/hosts' },
};

// --- Minimal protobuf wire-format decoder ---------------------------------
//
// Enough of the wire format to parse a LedgerEntry: varint (0),
// length-delimited (2) and fixed32/64 fields. Mirrors the decoder in the
// Python SDK's tests. Returns { fieldNumber: value[] }; length-delimited
// values are Buffers for the caller to interpret.

function decodeProtobuf(buf: Buffer): Record<number, any[]> {
  const fields: Record<number, any[]> = {};
  let pos = 0;
  const readVarint = (): number => {
    let result = 0;
    let shift = 0;
    for (;;) {
      const b = buf[pos++];
      result += (b & 0x7f) * Math.pow(2, shift);
      if (!(b & 0x80)) break;
      shift += 7;
    }
    return result;
  };
  while (pos < buf.length) {
    const tag = readVarint();
    const fieldNum = tag >> 3;
    const wireType = tag & 0x07;
    let val: any;
    if (wireType === 0) {
      val = readVarint();
    } else if (wireType === 2) {
      const len = readVarint();
      val = buf.subarray(pos, pos + len);
      pos += len;
    } else if (wireType === 5) {
      val = buf.subarray(pos, pos + 4);
      pos += 4;
    } else if (wireType === 1) {
      val = buf.subarray(pos, pos + 8);
      pos += 8;
    } else {
      throw new Error(`unsupported wire type ${wireType}`);
    }
    (fields[fieldNum] ||= []).push(val);
  }
  return fields;
}

interface ParsedEntry {
  cose_sign1: Buffer;
  bom_hash: string;
  raw_payload: Buffer;
  session_id: string;
  host_id: string;
  agent_id: string;
  mcp_server_id: string;
  tool_name: string;
  sdk_public_key: Buffer;
  metadata: Record<string, string>;
}

/** The policy the mock agent serves at /api/v1/redaction-policy; null = 404. */
let mockPolicy: unknown = { enabled: false, entity_types: [], operator: { type: 'replace' }, custom: [] };

function parseLedgerEntry(body: Buffer): ParsedEntry {
  const f = decodeProtobuf(body);
  const s = (n: number): string => (f[n] ? (f[n][0] as Buffer).toString('utf-8') : '');
  const b = (n: number): Buffer => (f[n] ? (f[n][0] as Buffer) : Buffer.alloc(0));

  const env = f[1] ? decodeProtobuf(f[1][0] as Buffer) : {};
  return {
    cose_sign1: env[1] ? (env[1][0] as Buffer) : Buffer.alloc(0),
    bom_hash: env[2] ? (env[2][0] as Buffer).toString('utf-8') : '',
    raw_payload: b(4),
    session_id: s(7),
    host_id: s(8),
    agent_id: s(9),
    mcp_server_id: s(10),
    tool_name: s(11),
    sdk_public_key: b(17),
    metadata: Object.fromEntries(
      (f[16] ?? []).map((e: Buffer) => {
        const kv = decodeProtobuf(e);
        return [(kv[1]?.[0] as Buffer).toString('utf-8'), kv[2] ? (kv[2][0] as Buffer).toString('utf-8') : ''];
      }),
    ),
  };
}

/** Spin up a mock witness server that records requests (protobuf transport). */
function createMockWitness(): Promise<{
  server: http.Server;
  port: number;
  requests: Array<{ path: string; contentType: string; entry: ParsedEntry }>;
}> {
  return new Promise((resolve) => {
    const requests: Array<{ path: string; contentType: string; entry: ParsedEntry }> = [];

    const server = http.createServer((req, res) => {
      if (req.method === 'GET' && req.url === '/api/v1/redaction-policy') {
        if (mockPolicy === null) {
          res.writeHead(404);
          res.end();
        } else {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(mockPolicy));
        }
        return;
      }
      const chunks: Buffer[] = [];
      req.on('data', (chunk) => chunks.push(chunk));
      req.on('end', () => {
        const body = Buffer.concat(chunks);
        requests.push({
          path: req.url || '',
          contentType: req.headers['content-type'] || '',
          entry: parseLedgerEntry(body),
        });
        res.writeHead(201, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok' }));
      });
    });

    server.listen(0, '127.0.0.1', () => {
      const addr = server.address() as { port: number };
      resolve({ server, port: addr.port, requests });
    });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('GolemSDK Witness Mode', () => {
  let mockServer: http.Server;
  let mockPort: number;
  let requests: Array<{ path: string; contentType: string; entry: ParsedEntry }>;

  beforeEach(async () => {
    const mock = await createMockWitness();
    mockServer = mock.server;
    mockPort = mock.port;
    requests = mock.requests;
  });

  afterEach((done) => {
    mockPolicy = { enabled: false, entity_types: [], operator: { type: 'replace' }, custom: [] };
    mockServer.close(done);
  });

  it('redacts what the machine\'s policy names before it signs', async () => {
    mockPolicy = {
      enabled: true,
      entity_types: ['EMAIL_ADDRESS', 'GITHUB_TOKEN'],
      operator: { type: 'replace' },
      custom: [],
    };
    const token = 'ghp_' + 'a'.repeat(36);
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    const args = { to: 'jane.roe@example.org', auth: { token }, list: ['ok', 'x@y.io'] };
    const result = await sdk.interceptExecution({ ...sampleBOM, tool_arguments: args, system_prompt: 'mail a@b.co' });
    await sleep(300);
    const signed = result.cborPayload.toString('latin1') + result.coseSign1.toString('latin1');
    for (const planted of ['jane.roe@example.org', token, 'x@y.io', 'a@b.co']) {
      expect(signed).not.toContain(planted);
    }
    const decoded = await cbor.decodeFirst(result.cborPayload);
    expect(decoded.tool_arguments.to).toBe('<EMAIL_ADDRESS>');
    expect(decoded.tool_arguments.auth.token).toBe('<GITHUB_TOKEN>');
    expect(args.to).toBe('jane.roe@example.org');
    expect(crypto.createHash('sha256').update(result.cborPayload).digest('hex')).toBe(result.bomHash);
    expect(requests[0].entry.metadata.redaction).toBe('{"EMAIL_ADDRESS":3,"GITHUB_TOKEN":1}');
  });

  it('says so when it could not get a policy, and still sends', async () => {
    mockPolicy = null;
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sleep(300);
    expect(requests).toHaveLength(1);
    expect(requests[0].entry.metadata.redaction).toBe('policy_unavailable');
  });

  it('should POST to /api/v1/sdk/tool-use', async () => {
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    expect(requests.length).toBe(1);
    expect(requests[0].path).toBe('/api/v1/sdk/tool-use');
  });

  it('should send protobuf (application/x-protobuf)', async () => {
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    expect(requests[0].contentType).toBe('application/x-protobuf');
  });

  it('should send all required fields including cose_sign1 and sdk_public_key', async () => {
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    const entry = requests[0].entry;
    expect(entry.cose_sign1.length).toBeGreaterThan(0);
    expect(entry.bom_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(entry.sdk_public_key.length).toBeGreaterThan(0);
    expect(entry.agent_id).toBe('test-ts-agent');
    expect(entry.mcp_server_id).toBe('test-server');
    expect(entry.tool_name).toBe('read_file');
  });

  it('should send a valid P-256 SPKI DER public key', async () => {
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    const pubKeyDer = requests[0].entry.sdk_public_key;
    // SPKI DER for P-256 is 91 bytes
    expect(pubKeyDer.length).toBe(91);

    // Verify it's parseable as a public key (this is what the witness does)
    const pubKey = crypto.createPublicKey({
      key: pubKeyDer,
      format: 'der',
      type: 'spki',
    });
    expect(pubKey.asymmetricKeyType).toBe('ec');
  });

  it('should produce a valid COSE_Sign1 envelope with correct bom_hash', async () => {
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    const entry = requests[0].entry;
    const decoded = cbor.decodeFirstSync(entry.cose_sign1);
    expect(decoded).toBeInstanceOf(cbor.Tagged);
    expect(decoded.tag).toBe(18);

    const coseArray = decoded.value;
    expect(coseArray.length).toBe(4);

    // Extract embedded CBOR payload and verify bom_hash
    const innerPayload = coseArray[2];
    const computedHash = crypto.createHash('sha256').update(innerPayload).digest('hex');
    expect(computedHash).toBe(entry.bom_hash);
    // raw_payload field carries the same CBOR bytes
    expect(Buffer.compare(entry.raw_payload, innerPayload)).toBe(0);

    // Verify protected headers contain ES256 algorithm
    const protectedHeaders = cbor.decodeFirstSync(coseArray[0]);
    expect(protectedHeaders.get(1)).toBe(-7); // ES256

    // Verify signature is 64 bytes (raw r||s for P-256)
    expect(coseArray[3].length).toBe(64);
  });

  it('should NOT return storageUri in witness mode', async () => {
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    const result = await sdk.interceptExecution(sampleBOM);

    expect(result.storageUri).toBeUndefined();
  });

  it('should forward session_id and host_id', async () => {
    const sdk = new GolemSDK({
      witnessUrl: `http://127.0.0.1:${mockPort}`,
      sessionId: 'sess-456',
      hostId: 'my-laptop',
    });
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    const entry = requests[0].entry;
    expect(entry.session_id).toBe('sess-456');
    expect(entry.host_id).toBe('my-laptop');
  });

  it('records every call from one SDK under one session by default', async () => {
    // Without an id the Golem agent made a new session for every call, so a
    // run's calls never chained or read as one conversation.
    const sdk = new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` });
    await sdk.interceptExecution(sampleBOM);
    await sdk.interceptExecution(sampleBOM);
    await sleep(500);

    expect(requests.length).toBe(2);
    expect(new Set(requests.map((r) => r.entry.session_id))).toEqual(new Set([sdk.session]));
    expect(sdk.session).toMatch(/^sdk-[0-9a-f]{20}$/);
    expect(new GolemSDK({ witnessUrl: `http://127.0.0.1:${mockPort}` }).session).not.toBe(sdk.session);
  });

  it('should not throw when witness is unreachable', async () => {
    const sdk = new GolemSDK({ witnessUrl: 'http://127.0.0.1:1' });
    // Fire-and-forget — should not throw
    const result = await sdk.interceptExecution(sampleBOM);
    expect(result.bomHash).toMatch(/^[0-9a-f]{64}$/);
  });

  it('should still work in legacy mode with storage', async () => {
    const { LocalDiskStorage } = await import('../storage');
    const fs = await import('fs');
    const os = await import('os');
    const path = await import('path');

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'golem-witness-test-'));
    try {
      // Suppress deprecation warning in test
      const origWarn = console.warn;
      console.warn = () => {};
      const sdk = new GolemSDK({ storage: new LocalDiskStorage(tmpDir) });
      console.warn = origWarn;

      const result = await sdk.interceptExecution(sampleBOM);
      expect(result.storageUri).toBeDefined();
      expect(result.storageUri).toMatch(/^file:\/\//);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
