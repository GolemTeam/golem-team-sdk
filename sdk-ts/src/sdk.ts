import * as crypto from 'crypto';
import * as cbor from 'cbor';
import { StorageBackend, LocalDiskStorage } from './storage';
import { WitnessClient, PolicyDecision, CheckExecutionArgs, ResultDecision } from './witness-client';
import { coseSign1 } from './cose';
import { RedactionPolicyCache, redactValue, countsMetadata } from './redaction';

/** Validation: identifiers must be alphanumeric with hyphens, underscores, dots, colons, slashes */
const SAFE_ID_PATTERN = /^[\w\-.:\/]{1,255}$/;

const MAX_TOOL_ARGS_KEYS = 100;
const MAX_CHUNKS = 500;
const MAX_STRING_LENGTH = 1024;
const MAX_SYSTEM_PROMPT_LENGTH = 100_000;

function validateIdField(name: string, value: any): void {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${name} must be a non-empty string`);
  }
  if (!SAFE_ID_PATTERN.test(value)) {
    throw new Error(
      `${name} contains invalid characters. ` +
      'Only alphanumeric, hyphens, underscores, dots, colons, and slashes are allowed.'
    );
  }
}

// Label keys the witness will accept and propagate into session
// metadata. The witness re-validates and silently drops anything
// outside this list, so adding a key here is necessary but not
// sufficient — keep in sync with the Golem agent's SDK handler.
const SDK_LABEL_KEYS = new Set([
  'branch',
  'worktree',
  'workspace',
  'env',
  'project',
  'purpose',
]);
const SDK_LABEL_VALUE_RE = /^[\w\-./@: ]{1,255}$/;

function validateSDKLabels(raw: Record<string, string> | undefined): Record<string, string> {
  if (!raw) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (!SDK_LABEL_KEYS.has(k)) continue;
    if (typeof v !== 'string' || !SDK_LABEL_VALUE_RE.test(v)) continue;
    out[k] = v;
  }
  return out;
}

export interface PreExecutionBOM {
  agent_id: string;
  mcp_server_id: string;
  tool_name: string;
  system_prompt?: string;
  retrieved_chunks?: Array<{
    id: string;
    content: string;
    source_uri?: string;
  }>;
  tool_arguments: Record<string, any>;
}

export interface RecordResultArgs {
  toolName: string;
  /** What the tool returned. Anything JSON can encode. */
  result: unknown;
  /** The toolUseId interceptExecution returned, so the result pairs with its call. */
  toolUseId?: string;
  agentId?: string;
  mcpServerId?: string;
  toolArguments?: Record<string, unknown>;
  isError?: boolean;
  durationMs?: number;
}

export interface RecordUsageArgs {
  model: string;
  inputTokens?: number;
  outputTokens?: number;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
  provider?: string;
  agentId?: string;
  durationMs?: number;
}

function count(n: number | undefined): number {
  return Number.isFinite(n) && (n as number) > 0 ? Math.floor(n as number) : 0;
}

export interface GolemSDKOptions {
  /** Golem Witness URL (enables witness mode). Default: undefined (legacy mode). */
  witnessUrl?: string;
  /**
   * @deprecated BYOB storage has moved to the Golem Relay.
   * Use witnessUrl instead — the Relay handles BYOB storage automatically.
   */
  storage?: StorageBackend;
  /** Session identifier (passed to witness). */
  sessionId?: string;
  /** Host identifier (passed to witness). */
  hostId?: string;
  /**
   * Friendly-name labels attached to every event from this SDK
   * instance. Useful when the witness can't infer them from the
   * environment (sidecar/docker deployments). Allowed keys: branch,
   * worktree, workspace, env, project, purpose. Values must match
   * `^[\w\-./@: ]{1,255}$`. Unknown keys / invalid values are dropped
   * silently. Example:
   *
   *     new GolemSDK({
   *       witnessUrl: 'http://localhost:39393',
   *       labels: { branch: 'feat/foo', env: 'dev' },
   *     })
   */
  labels?: Record<string, string>;
}

export class GolemSDK {
  private storage: StorageBackend | null;
  private identityPrivateKey: crypto.KeyObject;
  private mode: 'witness' | 'legacy';
  private witness: WitnessClient | null;
  private sessionId: string;
  private hostId: string;
  private labels: Record<string, string>;
  private redaction: RedactionPolicyCache | null = null;

  /**
   * Ask the Golem agent whether a call may run, before running it. Pass the
   * call's toolArguments and mcpServerId: rules and inventory decisions match on
   * them. Defaults to this SDK's session and host. Without a Golem agent
   * (legacy mode) the answer is allowed with reachable false.
   */
  checkExecution(args: CheckExecutionArgs): Promise<PolicyDecision> {
    if (!this.witness) {
      return Promise.resolve({ allowed: true, action: 'allow', reason: '', reachable: false, ruleName: '' });
    }
    return this.witness.checkExecution({
      ...args,
      sessionId: args.sessionId || this.sessionId,
      hostId: args.hostId || this.hostId,
    });
  }

  /**
   * Record what a tool returned, and learn whether the model should read it.
   * When the answer's action is `withhold`, give the model `notice` instead of
   * the result. The result is checked for prompt injection under the tool's
   * inventory name, `mcp__<mcpServerId>__<toolName>`. Fails open.
   */
  recordResult(args: RecordResultArgs): Promise<ResultDecision> {
    if (!this.witness) {
      return Promise.resolve({ action: 'allow', advice: '', notice: '', reachable: false });
    }
    const body: Record<string, unknown> = {
      tool_name: args.toolName,
      agent_id: args.agentId ?? '',
      mcp_server_id: args.mcpServerId ?? '',
      session_id: this.sessionId,
      host_id: this.hostId,
      tool_use_id: args.toolUseId ?? '',
      result: args.result === undefined ? null : args.result,
      is_error: Boolean(args.isError),
      duration_ms: count(args.durationMs),
    };
    if (args.toolArguments && Object.keys(args.toolArguments).length > 0) {
      body.tool_arguments = args.toolArguments;
    }
    return this.witness.recordResult(body);
  }

  /** Record one model call's tokens. Fire-and-forget. Golem does not price a model an SDK names. */
  recordUsage(args: RecordUsageArgs): void {
    if (!this.witness || !args.model) return;
    this.witness.recordUsage({
      model: args.model,
      provider: args.provider ?? '',
      agent_id: args.agentId ?? '',
      session_id: this.sessionId,
      host_id: this.hostId,
      input_tokens: count(args.inputTokens),
      output_tokens: count(args.outputTokens),
      cache_read_tokens: count(args.cacheReadTokens),
      cache_creation_tokens: count(args.cacheCreationTokens),
      duration_ms: count(args.durationMs),
    });
  }

  /** The session every call from this SDK is recorded under. */
  get session(): string {
    return this.sessionId;
  }

  constructor(options?: GolemSDKOptions) {
    // Without an id the Golem agent made a new session for every call, so a
    // run's calls never chained or read as one conversation.
    this.sessionId = options?.sessionId || `sdk-${crypto.randomBytes(10).toString('hex')}`;
    this.hostId = options?.hostId || '';
    this.labels = validateSDKLabels(options?.labels);

    // Ephemeral ECDSA P-256 signing key (non-repudiation from the caller)
    // Note: switched from secp256k1 to P-256 to match Python SDK and Witness
    const { privateKey } = crypto.generateKeyPairSync('ec', {
      namedCurve: 'prime256v1', // P-256 (same as Python SDK's NIST256p)
    });
    this.identityPrivateKey = privateKey;

    if (options?.witnessUrl) {
      // Witness mode: thin SDK -> Witness sidecar
      this.mode = 'witness';
      this.witness = new WitnessClient(options.witnessUrl);
      const witness = this.witness;
      this.redaction = new RedactionPolicyCache(() => witness.fetchRedactionPolicy());
      this.storage = null;
    } else if (options?.storage) {
      // Legacy mode: fat SDK with local storage
      this.mode = 'legacy';
      this.storage = options.storage;
      this.witness = null;
      console.warn(
        'GolemSDK: Legacy mode (storage) is deprecated. ' +
        'Use GolemSDK({ witnessUrl: "http://localhost:39393" }) instead. ' +
        'The Witness handles storage, embedding, and forwarding.'
      );
    } else {
      // Default: legacy mode with local storage (backward compatible)
      this.mode = 'legacy';
      this.storage = new LocalDiskStorage();
      this.witness = null;
    }
  }

  /**
   * Export the SDK's ephemeral public key in DER (SPKI) format.
   * Used by the Witness to verify signatures.
   */
  get publicKeyDer(): Buffer {
    const pubKey = crypto.createPublicKey(this.identityPrivateKey);
    return pubKey.export({ type: 'spki', format: 'der' }) as Buffer;
  }

  /**
   * Applies this machine's redaction policy to what will be signed, on a copy.
   * Never throws: with no policy, or if redaction fails, the event goes out as
   * written and its metadata says so.
   */
  private async redact(bom: PreExecutionBOM): Promise<[PreExecutionBOM, Record<string, string>]> {
    const current = this.redaction ? await this.redaction.current() : null;
    if (!current) return [bom, { redaction: 'policy_unavailable' }];
    if (!current.policy.enabled) return [bom, {}];
    const counts: Record<string, number> = {};
    try {
      const out: PreExecutionBOM = {
        ...bom,
        tool_arguments: redactValue(current.redactor, bom.tool_arguments, counts) as Record<string, any>,
      };
      if (bom.system_prompt !== undefined) {
        out.system_prompt = redactValue(current.redactor, bom.system_prompt, counts, 'system_prompt') as string;
      }
      if (bom.retrieved_chunks !== undefined) {
        out.retrieved_chunks = bom.retrieved_chunks.map((c) => ({
          ...c,
          content: redactValue(current.redactor, c.content, counts, 'content') as string,
          ...(c.source_uri === undefined
            ? {}
            : { source_uri: redactValue(current.redactor, c.source_uri, counts, 'source_uri') as string }),
        }));
      }
      return [out, Object.keys(counts).length ? { redaction: countsMetadata(counts) } : {}];
    } catch (e) {
      console.warn(`GolemSDK: redaction failed (${(e as Error)?.name ?? 'Error'}); event sent unredacted`);
      return [bom, { redaction: 'failed' }];
    }
  }

  /**
   * Intercepts an agent's execution intent, hashes the state, signs it,
   * and routes it to the appropriate destination.
   *
   * In witness mode: signs and POSTs to the Witness (localhost sidecar).
   * In legacy mode: signs, stores to BYOB.
   *
   * The returned toolUseId (generated unless one is passed) is what
   * recordResult pairs the call's result on.
   */
  async interceptExecution(bom: PreExecutionBOM, options?: { toolUseId?: string }): Promise<{
    toolUseId: string;
    bomHash: string;
    storageUri?: string;
    coseSign1: Buffer;
    cborPayload: Buffer;
  }> {
    // Input validation
    validateIdField('agent_id', bom.agent_id);
    validateIdField('mcp_server_id', bom.mcp_server_id);
    validateIdField('tool_name', bom.tool_name);
    const toolUseId = options?.toolUseId || `call-${crypto.randomBytes(8).toString('hex')}`;
    validateIdField('tool_use_id', toolUseId);

    if (typeof bom.tool_arguments !== 'object' || bom.tool_arguments === null || Array.isArray(bom.tool_arguments)) {
      throw new Error('tool_arguments must be a plain object');
    }
    if (Object.keys(bom.tool_arguments).length > MAX_TOOL_ARGS_KEYS) {
      throw new Error(`tool_arguments exceeds maximum of ${MAX_TOOL_ARGS_KEYS} keys`);
    }

    if (bom.system_prompt !== undefined) {
      if (typeof bom.system_prompt !== 'string') {
        throw new Error('system_prompt must be a string');
      }
      if (bom.system_prompt.length > MAX_SYSTEM_PROMPT_LENGTH) {
        throw new Error(`system_prompt exceeds maximum length of ${MAX_SYSTEM_PROMPT_LENGTH}`);
      }
    }

    if (bom.retrieved_chunks !== undefined) {
      if (!Array.isArray(bom.retrieved_chunks)) {
        throw new Error('retrieved_chunks must be an array');
      }
      if (bom.retrieved_chunks.length > MAX_CHUNKS) {
        throw new Error(`retrieved_chunks exceeds maximum of ${MAX_CHUNKS} entries`);
      }
    }

    // 1. Serialize deterministic payload (CBOR)
    // Use `encodeAsync`: synchronous `cbor.encode` is broken under Node.js >= 26
    // (NoFilter's sync read-back drops the payload, emitting only the type
    // prefix). The async path is byte-identical and works on all Node versions.
    let redactionMeta: Record<string, string> = {};
    if (this.mode === 'witness') {
      [bom, redactionMeta] = await this.redact(bom);
    }
    const cborPayload = await cbor.encodeAsync(bom);

    // 2. Cryptographic hash (SHA-256 of CBOR payload)
    const bomHash = crypto.createHash('sha256').update(cborPayload).digest('hex');

    // 3. COSE_Sign1 envelope: wraps CBOR payload with ES256 signature
    // Protected headers carry alg=ES256 and kid=SHA-256(DER pub key).
    // The Sig_structure is ["Signature1", protected, "", payload].
    // Signature is raw r||s (64 bytes) per RFC 9052 §8.1.
    const coseBytes = await coseSign1(cborPayload, this.identityPrivateKey, this.publicKeyDer);

    // 4. Route based on mode
    if (this.mode === 'witness' && this.witness) {
      // Witness mode: POST the protobuf LedgerEntry to the Witness sidecar.
      this.witness.record({
        agent_id: bom.agent_id,
        mcp_server_id: bom.mcp_server_id,
        tool_name: bom.tool_name,
        bom_hash: bomHash,
        cbor_payload: cborPayload,
        cose_sign1: coseBytes,
        public_key: this.publicKeyDer,
        session_id: this.sessionId,
        host_id: this.hostId,
        labels: this.labels,
        metadata: { ...redactionMeta, tool_use_id: toolUseId },
      });

      return {
        toolUseId,
        bomHash,
        coseSign1: coseBytes,
        cborPayload,
      };
    } else {
      // Legacy mode: store locally
      const storageUri = await this.storage!.storeRawPayload(bomHash, bom);

      return {
        toolUseId,
        bomHash,
        storageUri,
        coseSign1: coseBytes,
        cborPayload,
      };
    }
  }
}
