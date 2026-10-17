/**
 * On-disk durable-delivery buffer for the TypeScript SDK.
 *
 * The witness runs a WAL for its witness->cloud hop; this is the SDK-side
 * equivalent for the SDK->witness hop. SDK processes are often short-lived, so
 * the buffer is on disk — entries survive process exit and are replayed on a
 * later run. Each record is the raw protobuf LedgerEntry body.
 *
 * Replay is idempotent: the witness / Ingest dedup by entry_hash, so re-POSTing
 * a record whose original delivery landed is a no-op (409 -> treated as ok).
 *
 * Disable with GOLEM_SDK_WAL_PATH=off.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const WAL_EXT = '.wal';

/** Outcome of a single delivery attempt used to drive the buffer. */
export type DeliverOutcome = 'ok' | 'permanent' | 'retry';

/** Resolve the buffer directory, or null when disabled. */
export function resolveWalDir(): string | null {
  const v = (process.env.GOLEM_SDK_WAL_PATH || '').trim();
  if (['off', 'disabled', 'false', '0'].includes(v.toLowerCase())) {
    return null;
  }
  if (v) {
    return v;
  }
  const home = os.homedir();
  if (!home) {
    return null;
  }
  return path.join(home, '.golem', 'sdk-wal');
}

export class SdkBuffer {
  private readonly dir: string;
  private readonly max: number;
  private seq = 0;

  constructor(dir: string, maxRecords = 10000) {
    this.dir = dir;
    this.max = maxRecords > 0 ? maxRecords : 10000;
    fs.mkdirSync(this.dir, { recursive: true, mode: 0o700 });
  }

  /** Record files, oldest-first (names sort chronologically). */
  private files(): string[] {
    let names: string[];
    try {
      names = fs.readdirSync(this.dir).filter((n) => n.endsWith(WAL_EXT));
    } catch {
      return [];
    }
    names.sort();
    return names;
  }

  pending(): number {
    return this.files().length;
  }

  /** Durably buffer one LedgerEntry body, dropping the oldest if over capacity. */
  append(body: Buffer): void {
    this.pruneToCap(this.max - 1);
    this.seq += 1;
    const name = `${process.hrtime.bigint().toString().padStart(20, '0')}-${String(
      this.seq,
    ).padStart(6, '0')}${WAL_EXT}`;
    const final = path.join(this.dir, name);
    const tmp = `${final}.tmp`;
    try {
      fs.writeFileSync(tmp, body, { mode: 0o600 });
      fs.renameSync(tmp, final);
    } catch (err) {
      try {
        fs.rmSync(tmp, { force: true });
      } catch {
        /* ignore */
      }
      console.warn(`wal: append failed: ${(err as Error).name}`);
    }
  }

  private pruneToCap(cap: number): void {
    if (cap < 0) cap = 0;
    const names = this.files();
    while (names.length > cap) {
      const oldest = names.shift() as string;
      console.warn('wal: capacity exceeded, dropping oldest buffered entry');
      try {
        fs.rmSync(path.join(this.dir, oldest), { force: true });
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * Replay buffered records oldest-first. `post(body)` resolves 'ok' or
   * 'permanent' to delete the record, 'retry' to keep it and stop this cycle
   * (backend likely still down).
   */
  async drain(post: (body: Buffer) => Promise<DeliverOutcome>): Promise<void> {
    for (const name of this.files()) {
      const p = path.join(this.dir, name);
      let body: Buffer;
      try {
        body = fs.readFileSync(p);
      } catch {
        continue;
      }
      let outcome: DeliverOutcome;
      try {
        outcome = await post(body);
      } catch {
        return; // a failing post must never crash the SDK; retry next drain
      }
      if (outcome === 'retry') {
        return;
      }
      try {
        fs.rmSync(p, { force: true });
      } catch {
        /* ignore */
      }
    }
  }
}
