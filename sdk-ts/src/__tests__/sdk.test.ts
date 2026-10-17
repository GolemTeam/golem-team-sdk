import { GolemSDK, PreExecutionBOM } from '../sdk';
import { LocalDiskStorage } from '../storage';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as crypto from 'crypto';
import * as cbor from 'cbor';

describe('GolemSDK', () => {
  let sdk: GolemSDK;
  let tmpDir: string;

  const sampleBOM: PreExecutionBOM = {
    agent_id: 'agent-test-001',
    mcp_server_id: 'mcp-server-001',
    tool_name: 'read_file',
    tool_arguments: { path: '/etc/hosts' },
  };

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'golem-test-'));
    sdk = new GolemSDK({ storage: new LocalDiskStorage(tmpDir) });
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should instantiate with LocalDiskStorage', () => {
    expect(sdk).toBeInstanceOf(GolemSDK);
  });

  it('should produce a valid BOM with all required fields', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    expect(result).toHaveProperty('bomHash');
    expect(result).toHaveProperty('storageUri');
    expect(result).toHaveProperty('coseSign1');
    expect(result).toHaveProperty('cborPayload');
  });

  it('should produce a valid hex SHA-256 hash (64 chars)', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    expect(result.bomHash).toMatch(/^[0-9a-f]{64}$/);
  });

  it('should produce a non-empty COSE_Sign1 envelope', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    expect(result.coseSign1).toBeInstanceOf(Buffer);
    expect(result.coseSign1.length).toBeGreaterThan(0);
  });

  it('should produce a valid CBOR payload buffer', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    expect(result.cborPayload).toBeInstanceOf(Buffer);
    expect(result.cborPayload.length).toBeGreaterThan(0);
  });

  it('should produce different hashes for different inputs', async () => {
    const result1 = await sdk.interceptExecution(sampleBOM);
    const result2 = await sdk.interceptExecution({
      ...sampleBOM,
      tool_name: 'write_file',
      tool_arguments: { path: '/tmp/out', content: 'hello' },
    });

    expect(result1.bomHash).not.toEqual(result2.bomHash);
  });

  it('should produce consistent hashes for identical inputs', async () => {
    const result1 = await sdk.interceptExecution(sampleBOM);
    const result2 = await sdk.interceptExecution(sampleBOM);

    expect(result1.bomHash).toEqual(result2.bomHash);
  });

  it('should store the raw payload to disk and return a file URI', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    expect(result.storageUri).toBeDefined();
    expect(result.storageUri).toMatch(/^file:\/\//);

    const filePath = result.storageUri!.replace('file://', '');
    expect(fs.existsSync(filePath)).toBe(true);

    const stored = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    expect(stored.agent_id).toEqual(sampleBOM.agent_id);
    expect(stored.tool_name).toEqual(sampleBOM.tool_name);
  });

  it('should handle optional fields (system_prompt, retrieved_chunks)', async () => {
    const bomWithOptionals: PreExecutionBOM = {
      ...sampleBOM,
      system_prompt: 'You are a helpful assistant.',
      retrieved_chunks: [
        { id: 'chunk-1', content: 'Relevant context here', source_uri: 'rag://doc1' },
      ],
    };

    const result = await sdk.interceptExecution(bomWithOptionals);
    expect(result.bomHash).toMatch(/^[0-9a-f]{64}$/);

    expect(result.storageUri).toBeDefined();
    const filePath = result.storageUri!.replace('file://', '');
    const stored = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    expect(stored.system_prompt).toEqual('You are a helpful assistant.');
    expect(stored.retrieved_chunks).toHaveLength(1);
  });

  // COSE_Sign1 specific tests

  it('should produce a valid COSE_Sign1 envelope (CBOR Tag 18)', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    // Decode the COSE_Sign1 bytes as CBOR
    const decoded = cbor.decodeFirstSync(result.coseSign1);

    // Should be a Tagged value with tag 18 (COSE_Sign1)
    expect(decoded).toBeInstanceOf(cbor.Tagged);
    expect(decoded.tag).toBe(18);

    // The value should be an array of 4 elements:
    // [protected, unprotected, payload, signature]
    const coseArray = decoded.value;
    expect(Array.isArray(coseArray)).toBe(true);
    expect(coseArray.length).toBe(4);
  });

  it('should embed the CBOR payload inside the COSE_Sign1 envelope', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    const decoded = cbor.decodeFirstSync(result.coseSign1);
    const coseArray = decoded.value;

    // coseArray[2] is the payload — should match the cborPayload
    const embeddedPayload = coseArray[2];
    expect(Buffer.isBuffer(embeddedPayload)).toBe(true);
    expect(Buffer.from(embeddedPayload).equals(result.cborPayload)).toBe(true);
  });

  it('should have bomHash matching SHA-256 of the inner CBOR payload', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    const decoded = cbor.decodeFirstSync(result.coseSign1);
    const innerPayload = decoded.value[2];

    const computedHash = crypto.createHash('sha256').update(innerPayload).digest('hex');
    expect(computedHash).toEqual(result.bomHash);
  });

  it('should include ES256 algorithm in COSE protected headers', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    const decoded = cbor.decodeFirstSync(result.coseSign1);
    const protectedBytes = decoded.value[0];

    // Decode the protected headers CBOR
    const headers = cbor.decodeFirstSync(protectedBytes);

    // alg (key=1) should be -7 (ES256)
    expect(headers.get(1)).toBe(-7);

    // kid (key=4) should be a Buffer (the key ID bytes)
    const kid = headers.get(4);
    expect(Buffer.isBuffer(kid)).toBe(true);
    expect(kid.length).toBeGreaterThan(0);
  });

  it('should produce a 64-byte raw signature (r||s for P-256)', async () => {
    const result = await sdk.interceptExecution(sampleBOM);

    const decoded = cbor.decodeFirstSync(result.coseSign1);
    const signature = decoded.value[3];

    expect(Buffer.isBuffer(signature)).toBe(true);
    expect(signature.length).toBe(64);
  });
});

describe('GolemSDK labels parameter', () => {
  it('stores valid labels for forwarding to the witness', () => {
    const sdk = new GolemSDK({
      witnessUrl: 'http://localhost:39393',
      labels: { branch: 'feat/foo', env: 'dev', workspace: 'alice' },
    });
    // Internal field intentionally — verifying the filter applied.
    expect((sdk as any).labels).toEqual({
      branch: 'feat/foo',
      env: 'dev',
      workspace: 'alice',
    });
  });

  it('drops unknown keys silently', () => {
    const sdk = new GolemSDK({
      witnessUrl: 'http://localhost:39393',
      labels: { branch: 'main', cwd: '/should/be/dropped' } as any,
    });
    expect((sdk as any).labels).toEqual({ branch: 'main' });
  });

  it('drops malformed values', () => {
    const sdk = new GolemSDK({
      witnessUrl: 'http://localhost:39393',
      labels: { branch: 'main', purpose: 'rm -rf /; echo pwned' },
    });
    expect((sdk as any).labels).toEqual({ branch: 'main' });
  });

  it('drops oversized values (>255 chars)', () => {
    const sdk = new GolemSDK({
      witnessUrl: 'http://localhost:39393',
      labels: { branch: 'main', purpose: 'a'.repeat(300) },
    });
    expect((sdk as any).labels).toEqual({ branch: 'main' });
  });

  it('omitted labels yields empty object', () => {
    const sdk = new GolemSDK({ witnessUrl: 'http://localhost:39393' });
    expect((sdk as any).labels).toEqual({});
  });
});
