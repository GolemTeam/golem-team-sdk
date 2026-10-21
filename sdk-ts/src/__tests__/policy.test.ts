import * as http from 'http';
import { GolemSDK } from '../sdk';

interface Served {
  url: string;
  close: () => Promise<void>;
  seen: Array<Record<string, any>>;
}

function serve(status: number, reply: string): Promise<Served> {
  const seen: Array<Record<string, any>> = [];
  const srv = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => {
      if (req.url === '/api/v1/policy-check') {
        seen.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      }
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(reply);
    });
  });
  return new Promise((resolve) => {
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as { port: number }).port;
      resolve({ url: `http://127.0.0.1:${port}`, seen, close: () => new Promise((r) => srv.close(() => r())) });
    });
  });
}

describe('checkExecution', () => {
  beforeAll(() => {
    process.env.GOLEM_SDK_WAL_PATH = 'off';
  });

  it('reports a block, with the rule that made it', async () => {
    const s = await serve(200, JSON.stringify({ action: 'block', rule_name: 'no-deletes', reason: 'no' }));
    try {
      const d = await new GolemSDK({ witnessUrl: s.url }).checkExecution({ toolName: 'drop_table' });
      expect(d).toEqual({ allowed: false, action: 'block', reason: 'no', reachable: true, ruleName: 'no-deletes' });
    } finally {
      await s.close();
    }
  });

  it('sends the arguments, server and this SDK session that rules match on', async () => {
    const s = await serve(200, JSON.stringify({ action: 'none' }));
    try {
      const sdk = new GolemSDK({ witnessUrl: s.url, hostId: 'ci-runner' });
      const d = await sdk.checkExecution({
        toolName: 'drop_table',
        agentId: 'bot',
        toolArguments: { table: 'users' },
        mcpServerId: 'langchain',
      });
      expect(d.allowed).toBe(true);
      expect(s.seen[0]).toMatchObject({
        tool_name: 'drop_table',
        agent_id: 'bot',
        tool_arguments: { table: 'users' },
        mcp_server_id: 'langchain',
        event_type: 'sdk_tool_call',
        session_id: sdk.session,
        host_id: 'ci-runner',
      });
    } finally {
      await s.close();
    }
  });

  it('fails open when the Golem agent cannot be reached', async () => {
    const d = await new GolemSDK({ witnessUrl: 'http://127.0.0.1:1' }).checkExecution({ toolName: 't' });
    expect(d).toMatchObject({ allowed: true, reachable: false });
  });

  it('fails open on a 5xx or an answer it cannot read', async () => {
    for (const [status, body] of [[500, '{}'], [200, 'not json']] as const) {
      const s = await serve(status, body);
      try {
        const d = await new GolemSDK({ witnessUrl: s.url }).checkExecution({ toolName: 't' });
        expect(d).toMatchObject({ allowed: true, reachable: false });
      } finally {
        await s.close();
      }
    }
  });
});
