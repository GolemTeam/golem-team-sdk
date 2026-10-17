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

export class WitnessClient {
  private readonly endpoint: string;
  private readonly policyEndpoint: string;
  private readonly timeout: number;
  private readonly buffer: SdkBuffer | null;

  constructor(witnessUrl: string = 'http://localhost:39393') {
    this.endpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/sdk/tool-use';
    this.policyEndpoint = witnessUrl.replace(/\/$/, '') + '/api/v1/redaction-policy';
    this.timeout = 5000; // ms

    const dir = resolveWalDir();
    this.buffer = dir ? new SdkBuffer(dir) : null;
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
