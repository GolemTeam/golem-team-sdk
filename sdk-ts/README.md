# Golem TypeScript SDK

Edge SDK for capturing AI agent execution state, creating cryptographic proofs, and routing sensitive data to customer-controlled storage (BYOB — Bring Your Own Bucket).

## What It Does

1. **Serializes** — Takes an agent's pre-execution state (tool name, arguments, system prompt, RAG chunks) and encodes it as deterministic CBOR.
2. **Hashes** — Computes a SHA-256 hash of the CBOR payload.
3. **Signs** — Signs the hash with ECDSA on the NIST **P-256** curve (`prime256v1`, COSE ES256 — matching the Python SDK and the Golem agent) using Node.js `crypto`.
4. **Stores** — Routes the raw payload to a pluggable storage backend and returns a URI pointer.

The hash and signature go into the transparency log. The raw data stays in the customer's storage.

## Installation

The SDK is not published to npm; build it from this directory. The cloud
backends' SDKs (`@aws-sdk/client-s3`, `@google-cloud/storage`,
`@azure/storage-blob`) are regular dependencies, installed with it:

```bash
cd sdk-ts
npm install
npm run build
```

## Quick Start

The recommended setup sends events to the Golem agent: `new GolemSDK({ witnessUrl: 'http://localhost:39393' })`
signs with COSE_Sign1 and POSTs a protobuf `LedgerEntry` to the local Golem agent,
which owns BYOB storage. The agent checks that signature, then signs the ledger entry
with the machine's key: the entry's `payload_hash` is the SDK's `bom_hash`, the signed
bytes go to your storage, and the SDK's own signature stays on the entry as
`sdk_cose_sign1`. Every call from one `GolemSDK` is recorded under one session
(`sdk.session`); pass `sessionId` to group a run under an id of your own. The example below is the deprecated legacy mode, in which
the SDK stores the payload itself:

```typescript
import { GolemSDK } from './src/sdk';
import { S3Storage } from './src/storage';

const sdk = new GolemSDK({
  storage: new S3Storage('my-bucket', 'boms/', 'us-west-2'),
});

const result = await sdk.interceptExecution({
  agent_id: 'support-bot-v2',
  mcp_server_id: 'srv-jira-prod',
  tool_name: 'SearchTickets',
  tool_arguments: { query: 'auth failures', max_results: 10 },
  system_prompt: 'You are an IT support assistant.',
  retrieved_chunks: [
    { id: 'c1', content: 'Refund policy text...', source_uri: 'rag://policies/refunds' },
  ],
});

console.log(result.bomHash);     // SHA-256 hex (64 chars)
console.log(result.storageUri);  // s3://my-bucket/boms/<hash>.json
console.log(result.coseSign1);   // COSE_Sign1 envelope, ES256 (Buffer)
```

## Before, after, and around a call

With the Golem agent (`witnessUrl`), an app can ask before a call, and record the
result and each model call's usage after:

```typescript
const sdk = new GolemSDK({ witnessUrl: 'http://localhost:39393' });

const args = { id: 'OPS-12' };
const decision = await sdk.checkExecution({ toolName: 'get_ticket', mcpServerId: 'jira', toolArguments: args });
if (!decision.allowed) throw new Error(`Blocked by Golem policy: ${decision.reason}`);

const call = await sdk.interceptExecution({ agent_id: 'support-bot', mcp_server_id: 'jira', tool_name: 'get_ticket', tool_arguments: args });
const output = await getTicket(args.id);
const verdict = await sdk.recordResult({ toolName: 'get_ticket', result: output, toolUseId: call.toolUseId, mcpServerId: 'jira' });
const reply = verdict.action === 'withhold' ? verdict.notice : output; // what the model reads

sdk.recordUsage({ model: 'claude-sonnet-5-5', inputTokens: 1200, outputTokens: 85, provider: 'anthropic' });
```

- `checkExecution` answers `allowed: false` only on a block or deny. It fails open: an
  unreachable agent answers allowed, with `reachable: false`.
- `recordResult` pairs the result with its call on `toolUseId`. The Golem agent
  checks the result for prompt injection as `mcp__<mcpServerId>__<toolName>`. When a
  rule blocks on that check, the action is `withhold` and `notice` is the text to
  give the model instead. When a rule advises, `advice` goes alongside the result. It
  also fails open.
- `recordUsage` sends one model call's tokens without waiting. Golem does not
  price a model an SDK names.

## Redaction before signing

The Golem agent cannot redact an SDK event, because the SDK signs it first. So with
`witnessUrl` set, the SDK asks the Golem agent on this machine what to remove
(`GET /api/v1/redaction-policy`, kept for 30 seconds) and removes it from the tool
arguments, the system prompt and the retrieved chunks before it signs. That is
what the redaction packs on this machine's profiles name: contact and financial
details, secrets and keys, and your own patterns. Names and addresses need the
relay's model, so the SDK does not find them.

The event says what was removed (`metadata.redaction`, counts per kind), or that
no policy could be fetched (`policy_unavailable`), in which case it is sent as
written. Your own arguments are never changed in place.

## Storage Backends

| Backend | Class | URI Format |
|---|---|---|
| Local Disk | `LocalDiskStorage` | `file:///tmp/golem-byob/{hash}.json` |
| AWS S3 | `S3Storage` | `s3://{bucket}/{prefix}{hash}.json` |
| Google Cloud | `GCSStorage` | `gs://{bucket}/{prefix}{hash}.json` |
| Azure Blob | `AzureBlobStorage` | `https://{account}.blob.core.windows.net/...` |

Cloud backends use lazy initialization — they only import the cloud SDK when first accessed. All four backends are deprecated along with legacy mode: BYOB storage has moved to the Relay.

## Files

| File | Purpose |
|---|---|
| `src/sdk.ts` | `GolemSDK` class: validate → CBOR → SHA-256 → ECDSA → store |
| `src/storage.ts` | `StorageBackend` interface + 4 implementations with path traversal protection |
| `src/cose.ts` | COSE_Sign1 (RFC 9052, ES256) envelope builder |
| `src/transport.ts` | Hand-rolled protobuf encoder for the `LedgerEntry` the Golem agent accepts |
| `src/witness-client.ts` | `WitnessClient`: witness-mode delivery to `/api/v1/sdk/tool-use` |
| `src/wal.ts` | On-disk buffer for entries the witness could not take (`GOLEM_SDK_WAL_PATH=off` disables it) |
| `src/__tests__/sdk.test.ts` | 19 tests: hashing, signing, storage, determinism, optional fields |
| `src/__tests__/storage.test.ts` | 4 tests: LocalDiskStorage CRUD |
| `src/__tests__/witness.test.ts` | 9 tests: witness mode |
| `src/__tests__/wal.test.ts` | 4 tests: the on-disk buffer |
| `package.json` | Dependencies and scripts |
| `tsconfig.json` | CommonJS, ES2020 target, strict mode |

## Building

```bash
npm run build   # Compiles to dist/
```

## Testing

```bash
npm install
npm test   # 36 tests via Jest, no external deps needed
```

## Security

- All `hashId` parameters validated as `^[a-f0-9]{64}$` before file operations
- File paths resolved and verified to stay within base directory (path traversal protection)
- Files written with `0o600` permissions; directories created with `0o700`
- Input validation on all public API fields (type, length, format)
- Payload size capped at 10 MB
- Error messages sanitized — log `error.constructor.name`, not full error details
- Top-level `import` statements used throughout, except the cloud backends' lazy `require()` of their SDKs
- Bucket and container names validated on construction
