import { SdkBuffer, DeliverOutcome } from '../wal';
import { WitnessClient } from '../witness-client';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as http from 'http';

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'golem-sdk-wal-'));
}

describe('SdkBuffer', () => {
  it('appends and drains oldest-first on success', async () => {
    const dir = tmpDir();
    const buf = new SdkBuffer(dir);
    buf.append(Buffer.from('body-0'));
    buf.append(Buffer.from('body-1'));
    buf.append(Buffer.from('body-2'));
    expect(buf.pending()).toBe(3);

    const got: string[] = [];
    await buf.drain(async (b) => {
      got.push(b.toString());
      return 'ok';
    });
    expect(buf.pending()).toBe(0);
    expect(got).toEqual(['body-0', 'body-1', 'body-2']);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('stops draining on a retryable failure and keeps records', async () => {
    const dir = tmpDir();
    const buf = new SdkBuffer(dir);
    buf.append(Buffer.from('a'));
    buf.append(Buffer.from('b'));

    let attempts = 0;
    await buf.drain(async (): Promise<DeliverOutcome> => {
      attempts++;
      return 'retry';
    });
    expect(attempts).toBe(1);
    expect(buf.pending()).toBe(2);

    await buf.drain(async () => 'ok');
    expect(buf.pending()).toBe(0);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('evicts the oldest record over capacity', async () => {
    const dir = tmpDir();
    const buf = new SdkBuffer(dir, 3);
    for (let i = 0; i < 6; i++) buf.append(Buffer.from(`e-${i}`));
    expect(buf.pending()).toBe(3);
    const survivors: string[] = [];
    await buf.drain(async (b) => {
      survivors.push(b.toString());
      return 'ok';
    });
    expect(survivors).toEqual(['e-3', 'e-4', 'e-5']);
    fs.rmSync(dir, { recursive: true, force: true });
  });
});

describe('WitnessClient buffering', () => {
  const sampleArgs = {
    agent_id: 'a',
    mcp_server_id: 's',
    tool_name: 'read_file',
    bom_hash: 'deadbeef',
    cbor_payload: Buffer.from('cbor'),
    cose_sign1: Buffer.from('cose'),
    public_key: Buffer.from('pub'),
  };

  it('buffers a failed post, then replays it on the next client', async () => {
    const dir = tmpDir();
    process.env.GOLEM_SDK_WAL_PATH = dir;
    try {
      // 1. Witness unreachable → record buffers after retries.
      const down = new WitnessClient('http://127.0.0.1:1');
      down.record(sampleArgs);
      // deliver() runs async with up to 3 attempts against a dead port.
      await new Promise((r) => setTimeout(r, 400));
      const buf = new SdkBuffer(dir);
      expect(buf.pending()).toBe(1);

      // 2. A live witness drains the backlog when the next client starts.
      let received = 0;
      const srv = http.createServer((req, res) => {
        req.resume();
        received++;
        res.writeHead(201);
        res.end();
      });
      await new Promise<void>((r) => srv.listen(0, '127.0.0.1', r));
      const port = (srv.address() as { port: number }).port;
      try {
        // Constructing the client kicks off a background drain.
        new WitnessClient(`http://127.0.0.1:${port}`);
        await new Promise((r) => setTimeout(r, 400));
        expect(received).toBe(1);
        expect(new SdkBuffer(dir).pending()).toBe(0);
      } finally {
        srv.close();
      }
    } finally {
      process.env.GOLEM_SDK_WAL_PATH = 'off';
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
