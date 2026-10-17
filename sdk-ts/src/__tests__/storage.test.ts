import { LocalDiskStorage } from '../storage';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

// Valid SHA-256 hex strings for testing
const HASH_1 = 'a'.repeat(64);  // aaaa...a (64 chars)
const HASH_2 = 'b'.repeat(64);  // bbbb...b (64 chars)
const HASH_3 = '0123456789abcdef'.repeat(4);  // 64 chars

describe('LocalDiskStorage', () => {
  let storage: LocalDiskStorage;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'golem-storage-test-'));
    storage = new LocalDiskStorage(tmpDir);
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should create the base directory', () => {
    expect(fs.existsSync(tmpDir)).toBe(true);
  });

  it('should store a payload and return a file URI', async () => {
    const payload = { agent_id: 'test', tool_name: 'foo' };

    const uri = await storage.storeRawPayload(HASH_1, payload);

    expect(uri).toContain(`file://`);
    expect(uri).toContain(`${HASH_1}.json`);
  });

  it('should write valid JSON to disk', async () => {
    const payload = {
      agent_id: 'agent-1',
      mcp_server_id: 'server-1',
      tool_name: 'read_file',
      tool_arguments: { path: '/tmp/data' },
    };

    await storage.storeRawPayload(HASH_3, payload);

    const filePath = path.join(tmpDir, `${HASH_3}.json`);
    expect(fs.existsSync(filePath)).toBe(true);

    const contents = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    expect(contents.agent_id).toEqual('agent-1');
    expect(contents.tool_name).toEqual('read_file');
    expect(contents.tool_arguments.path).toEqual('/tmp/data');
  });

  it('should handle multiple payloads without collision', async () => {
    const payload1 = { id: 1, data: 'first' };
    const payload2 = { id: 2, data: 'second' };

    await storage.storeRawPayload(HASH_1, payload1);
    await storage.storeRawPayload(HASH_2, payload2);

    const file1 = JSON.parse(fs.readFileSync(path.join(tmpDir, `${HASH_1}.json`), 'utf-8'));
    const file2 = JSON.parse(fs.readFileSync(path.join(tmpDir, `${HASH_2}.json`), 'utf-8'));

    expect(file1.id).toEqual(1);
    expect(file2.id).toEqual(2);
  });
});
