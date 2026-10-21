/**
 * WitnessClient: Fire-and-forget HTTP client for the Golem Witness.
 *
 * The Witness verifies the SDK's COSE_Sign1 signature, generates an
 * embedding, stores the raw artifact to BYOB, and forwards to Ingest.
 *
 * Transport is protobuf: the SDK POSTs a `LedgerEntry` message
 * (Content-Type: application/x-protobuf) to `/api/v1/sdk/tool-use`.
 *
 * Durable delivery: a POST that fails after a few live retries is written to an
 * on-disk buffer (~/.golem/sdk-wal) and replayed on the next SDK run, so a
 * witness that is momentarily down/restarting doesn't cost the ledger an entry.
 * Disable with GOLEM_SDK_WAL_PATH=off. Replay is idempotent (409 -> delivered).
 */

import * as http from 'http';
import { encodeLedgerEntry } from './transport';
import { SdkBuffer, DeliverOutcome, resolveWalDir } from './wal';

export interface WitnessRecordArgs {
  agent_id: string;
  mcp_server_id: string;
  tool_name: string;
  bom_hash: string;
  /** CBOR-canonical BOM bytes (the COSE payload / signing input). */
  cbor_payload: Buffer;
  /** Complete COSE_Sign1 envelope bytes. */
  cose_sign1: Buffer;
  /** DER (SPKI) public key — the witness needs this to verify the signature. */
  public_key: Buffer;
  session_id?: string;
  host_id?: string;
  // SDK-supplied labels (branch, workspace, env, etc.). Map onto the
  // LedgerEntry.metadata field; the witness re-validates against its
  // allow-list and drops anything outside it.
  labels?: Record<string, string>;
  /** What the SDK says about the redaction it did (`redaction` key). */
  metadata?: Record<string, string>;
}

const POST_MAX_TRIES = 3;

/**
 * One synchronous policy answer. `reachable` is separate from `allowed`:
 * allowed with reachable false is "nobody said no", not "the policy said yes".
 */
export interface PolicyDecision {
  allowed: boolean;
  action: string;
  reason: string;
  reachable: boolean;
  ruleName: string;
}

export interface CheckExecutionArgs {
  toolName: string;
  /** The shell line, for a shell tool. */
  command?: string;
  agentId?: string;
  sessionId?: string;
  hostId?: string;
  /** The call's arguments: rules match on them. */
  toolArguments?: Record<string, unknown>;
  /** The server or framework the tool belongs to: rules and inventory decisions match on it. */
  mcpServerId?: string;
}

/**
 * What to do with a tool's result before the model reads it. `withhold`: hand
 * the model `notice` instead. `advise`: hand it the result with `advice`
 * beside it. Unreachable reads as allow.
 */
export interface ResultDecision {
  action: string;
  advice: string;
  notice: string;
  reachable: boolean;
}

const RESULT_ALLOW: ResultDecision = { action: 'allow', advice: '', notice: '', reachable: false };
const MAX_RESULT_BYTES = 4 * 1024 * 1024;

function jsonSafe(_key: string, value: unknown): unknown {
  return typeof value === 'bigint' ? value.toString() : value;
}

function jsonLength(body: unknown): number {
  try {
    return Buffer.byteLength(JSON.stringify(body, jsonSafe));
  } catch {
    return Infinity;
  }
}

export class WitnessClient {
  private readonly endpoint: string;
  private readonly checkEndpoint: string;
  private readonly resultEndpoint: string;
  private readonly usageEndpoint: string;
  private readonly policyEndpoint: string;
  private readonly timeout: number;
  private readonly buffer: SdkBuffer | null;

  constructor(witnessUrl: string = 'http://localhost:39393') {
    this.endpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/sdk/tool-use';
    this.checkEndpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/policy-check';
    this.resultEndpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/sdk/tool-result';
    this.usageEndpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/sdk/usage';
    this.policyEndpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/redaction-policy';
    this.timeout = 5000; // ms

    const dir = resolveWalDir();
    this.buffer = null;
    if (dir) {
      try {
        this.buffer = new SdkBuffer(dir);
      } catch (e) {
        console.warn(`[golem-sdk] buffer disabled: cannot use ${dir} (${(e as NodeJS.ErrnoException).code ?? 'error'})`);
      }
    }
    if (this.buffer) {
      // Replay any backlog from a previous run, off the caller's path.
      void this.buffer.drain((b) => this.postOnce(b)).catch(() => {});
    }
  }

  /**
   * Post a signed BOM entry to the Witness. Fire-and-forget — never blocks; on
   * persistent failure the entry is buffered to disk and replayed later.
   */
  record(args: WitnessRecordArgs): void {
    const body = encodeLedgerEntry({
      coseSign1: args.cose_sign1,
      bomHash: args.bom_hash,
      rawPayload: args.cbor_payload,
      payloadHash: args.bom_hash,
      sessionId: args.session_id,
      hostId: args.host_id,
      agentId: args.agent_id,
      mcpServerId: args.mcp_server_id,
      toolName: args.tool_name,
      eventType: 'sdk_tool_call',
      metadata: { ...(args.labels ?? {}), ...(args.metadata ?? {}) },
      sdkPublicKey: args.public_key,
    });
    // Fire-and-forget: do not await.
    void this.deliver(body, args.tool_name);
  }

  /** GET this machine's redaction policy. Rejects on any failure. */
  fetchRedactionPolicy(): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.policyEndpoint);
      const req = http.get(
        { hostname: url.hostname, port: url.port, path: url.pathname, timeout: 1000 },
        (res) => {
          if (res.statusCode !== 200) {
            res.resume();
            reject(new Error(`redaction policy: HTTP ${res.statusCode}`));
            return;
          }
          const chunks: Buffer[] = [];
          let size = 0;
          res.on('data', (c: Buffer) => {
            size += c.length;
            if (size > 1 << 20) {
              req.destroy(new Error('redaction policy too large'));
              return;
            }
            chunks.push(c);
          });
          res.on('end', () => {
            try {
              resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            } catch (e) {
              reject(e);
            }
          });
          res.on('error', reject);
        },
      );
      req.on('timeout', () => req.destroy(new Error('redaction policy timed out')));
      req.on('error', reject);
    });
  }

  /** Deliver with a few live retries; buffer to disk on persistent failure. */
  private async deliver(body: Buffer, toolName: string): Promise<void> {
    for (let i = 0; i < POST_MAX_TRIES; i++) {
      const outcome = await this.postOnce(body);
      if (outcome === 'ok') {
        return;
      }
      if (outcome === 'permanent') {
        console.warn(`Witness permanently rejected entry for tool=${toolName}`);
        return;
      }
    }
    if (this.buffer) {
      this.buffer.append(body);
    } else {
      console.warn(`Failed to POST to witness (buffering disabled) for tool=${toolName}`);
    }
  }

  /**
   * Ask the Golem agent whether this call may run, before it runs.
   *
   * FAILS OPEN, like the Python SDK: an agent that is down, slow or unreachable
   * answers allowed with reachable false, because a collector outage must not
   * stop a customer's agent. A caller that wants to fail closed can check
   * `reachable`.
   */
  checkExecution(args: CheckExecutionArgs): Promise<PolicyDecision> {
    const open: PolicyDecision = { allowed: true, action: 'allow', reason: '', reachable: false, ruleName: '' };
    const body: Record<string, unknown> = {
      tool_name: args.toolName,
      command: args.command ?? '',
      agent_id: args.agentId ?? '',
      session_id: args.sessionId ?? '',
      host_id: args.hostId ?? '',
    };
    if (args.toolArguments && Object.keys(args.toolArguments).length > 0) {
      body.tool_arguments = args.toolArguments;
    }
    if (args.mcpServerId) {
      body.mcp_server_id = args.mcpServerId;
      body.event_type = 'sdk_tool_call';
    }
    return this.postJson(this.checkEndpoint, body, 2000).then((r) => {
      if (!r) return open;
      const action = String(r.action || 'allow').toLowerCase();
      return {
        allowed: action !== 'block' && action !== 'deny',
        action,
        reason: String(r.reason || ''),
        reachable: true,
        ruleName: String(r.rule_name || ''),
      };
    });
  }

  /**
   * Record a tool's result and resolve to what the injection check said. The
   * caller awaits it, since a verdict after the model read the result is no
   * use. Fails open.
   */
  recordResult(body: Record<string, unknown>): Promise<ResultDecision> {
    let sized = body;
    if (jsonLength(body) > MAX_RESULT_BYTES) {
      sized = { ...body, result: String(body.result).slice(0, MAX_RESULT_BYTES / 2) + '...[truncated]' };
    }
    return this.postJson(this.resultEndpoint, sized, this.timeout).then((r) => {
      if (!r) return { ...RESULT_ALLOW };
      return {
        action: String(r.action || 'allow').toLowerCase(),
        advice: String(r.advice || ''),
        notice: String(r.notice || ''),
        reachable: true,
      };
    });
  }

  /** Post one model call's token counts. Fire-and-forget. */
  recordUsage(body: Record<string, unknown>): void {
    void this.postJson(this.usageEndpoint, body, this.timeout);
  }

  /** POST JSON; resolves to the parsed 2xx answer, or null on any failure. Never rejects. */
  private postJson(endpoint: string, body: Record<string, unknown>, timeoutMs: number): Promise<Record<string, any> | null> {
    let payload: Buffer;
    try {
      payload = Buffer.from(JSON.stringify(body, jsonSafe));
    } catch {
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      const url = new URL(endpoint);
      const req = http.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Content-Length': payload.length },
          timeout: timeoutMs,
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on('data', (c: Buffer) => chunks.push(c));
          res.on('end', () => {
            const status = res.statusCode ?? 0;
            if (status < 200 || status >= 300) {
              resolve(null);
              return;
            }
            try {
              resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}'));
            } catch {
              resolve(null);
            }
          });
        },
      );
      req.on('error', () => resolve(null));
      req.on('timeout', () => {
        req.destroy();
        resolve(null);
      });
      req.write(payload);
      req.end();
    });
  }

  /** One POST attempt, classified: 'ok' | 'permanent' | 'retry'. */
  private postOnce(body: Buffer): Promise<DeliverOutcome> {
    return new Promise<DeliverOutcome>((resolve) => {
      const url = new URL(this.endpoint);
      const req = http.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname,
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-protobuf',
            'Content-Length': body.length,
          },
          timeout: this.timeout,
        },
        (res) => {
          res.resume();
          const s = res.statusCode ?? 0;
          if (s === 200 || s === 201 || s === 202 || s === 409) {
            resolve('ok'); // 409 = witness already has it (idempotent)
          } else if (s >= 400 && s < 500) {
            resolve('permanent');
          } else {
            resolve('retry');
          }
        },
      );
      req.on('error', () => resolve('retry')); // network / witness down
      req.on('timeout', () => {
        req.destroy();
        resolve('retry');
      });
      req.write(body);
      req.end();
    });
  }
}
