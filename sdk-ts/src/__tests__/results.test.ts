import * as http from 'http';
import { GolemSDK } from '../sdk';

interface Agent {
  url: string;
  close: () => Promise<void>;
  seen: Array<{ path: string; body: Buffer }>;
  wait: (path: string) => Promise<Buffer>;
}

function agent(answers: Record<string, unknown>): Promise<Agent> {
  const seen: Array<{ path: string; body: Buffer }> = [];
  const srv = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => {
      seen.push({ path: req.url ?? '', body: Buffer.concat(chunks) });
      res.writeHead(201, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(answers[req.url ?? ''] ?? { status: 'ok' }));
    });
  });
  const wait = async (path: string): Promise<Buffer> => {
    for (let i = 0; i < 150; i++) {
      const hit = seen.find((s) => s.path === path);
      if (hit) return hit.body;
      await new Promise((r) => setTimeout(r, 20));
    }
    throw new Error(`nothing posted to ${path}`);
  };
  return new Promise((resolve) => {
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as { port: number }).port;
      resolve({ url: `http://127.0.0.1:${port}`, seen, wait, close: () => new Promise((r) => srv.close(() => r())) });
    });
  });
}

describe('tool results and model usage', () => {
  beforeAll(() => {
    process.env.GOLEM_SDK_WAL_PATH = 'off';
  });

  it('pairs a result with its call, and says when to withhold it', async () => {
    const a = await agent({
      '/api/v1/sdk/tool-result': { status: 'ok', action: 'withhold', notice: '[Golem withheld this result]' },
    });
    try {
      const sdk = new GolemSDK({ witnessUrl: a.url, sessionId: 'run-1' });
      const call = await sdk.interceptExecution({
        agent_id: 'bot', mcp_server_id: 'langchain', tool_name: 'fetch_page', tool_arguments: { url: 'https://blog.example' },
      });
      const d = await sdk.recordResult({
        toolName: 'fetch_page', result: { text: 'page' }, toolUseId: call.toolUseId, mcpServerId: 'langchain',
      });
      const callBody = await a.wait('/api/v1/sdk/tool-use');
      const sent = JSON.parse((await a.wait('/api/v1/sdk/tool-result')).toString('utf8'));

      expect(call.toolUseId).toMatch(/^call-[0-9a-f]{16}$/);
      expect(callBody.includes(Buffer.from(call.toolUseId))).toBe(true);
      expect(d).toEqual({ action: 'withhold', advice: '', notice: '[Golem withheld this result]', reachable: true });
      expect(sent).toMatchObject({ tool_use_id: call.toolUseId, session_id: 'run-1', result: { text: 'page' } });
    } finally {
      await a.close();
    }
  });

  it('fails open when the Golem agent cannot answer', async () => {
    const d = await new GolemSDK({ witnessUrl: 'http://127.0.0.1:9' }).recordResult({ toolName: 'search', result: 'x' });
    expect(d).toEqual({ action: 'allow', advice: '', notice: '', reachable: false });
  });

  it('sends model usage as tokens, and nothing without a model', async () => {
    const a = await agent({});
    try {
      const sdk = new GolemSDK({ witnessUrl: a.url, sessionId: 'run-1' });
      sdk.recordUsage({ model: '', inputTokens: 1 });
      sdk.recordUsage({ model: 'claude-sonnet-5-5', inputTokens: 900, outputTokens: 40, provider: 'anthropic' });
      const sent = JSON.parse((await a.wait('/api/v1/sdk/usage')).toString('utf8'));
      expect(sent).toMatchObject({ model: 'claude-sonnet-5-5', input_tokens: 900, output_tokens: 40, session_id: 'run-1' });
      expect(a.seen.filter((s) => s.path === '/api/v1/sdk/usage')).toHaveLength(1);
    } finally {
      await a.close();
    }
  });
});
