# Golem Python SDK

Edge SDK for capturing AI agent execution state, creating cryptographic proofs, and routing sensitive data to customer-controlled storage (BYOB — Bring Your Own Bucket).

## What It Does

1. **Redacts** — Removes what this machine's redaction packs name (see [Redaction before signing](#redaction-before-signing)).
2. **Serializes** — Takes an agent's pre-execution state (tool name, arguments, system prompt, RAG chunks) and encodes it as deterministic CBOR.
3. **Hashes** — Computes a SHA-256 hash of the CBOR payload.
4. **Signs** — Signs the hash with ECDSA (NIST P-256) using an ephemeral workload identity key.
5. **Stores** — Routes the raw payload to a pluggable storage backend and returns a URI pointer.

The hash and signature go into the transparency log. The raw data stays in the customer's storage.

## Redaction before signing

The Golem agent redacts what it records itself, but it cannot redact an SDK event: the SDK signs the event before the Golem agent sees it, and changing it afterwards would break the signature. So the SDK removes what this machine's redaction packs name before it signs.

- **Where the policy comes from.** With `witness_url` set, the SDK asks the Golem agent on this machine (`GET /api/v1/redaction-policy`) which types to remove and how, and keeps the answer for 30 seconds.
- **What is redacted.** Every string in the tool arguments (keys and values, at any depth), the system prompt, the retrieved chunks, and the labels. The agent, server and tool names are left alone: they travel unsigned beside the event anyway.
- **What it finds.** What a pattern can find: email addresses, phone numbers, card numbers (checksum-checked), SSNs, IP addresses, IBANs, cloud and API keys and tokens, private keys, passwords in connection strings, and your workspace's own custom patterns.
- **What it does not find.** Names and street addresses. Those need a model, and the model runs in the Golem agent, which cannot reach inside a signed SDK event. If your packs name them, SDK events still carry them.
- **What replaces a value.** Whatever the policy says: a `<TYPE>` tag, a mask, a hash, or nothing. The SDK attaches how many of each type it removed as `redaction` metadata, e.g. `{"EMAIL_ADDRESS":2}` — never the values.
- **When the Golem agent cannot be reached.** The last policy it gave keeps applying. If it has never answered, the event goes out as written and is marked `redaction: policy_unavailable`. Recording never raises into your tool call.
- **Legacy mode** (no `witness_url`) has no Golem agent to ask and does not redact.

The recognizers are a port of the Golem agent's own (`pkg/redaction`), held to the same test vectors (`pkg/redaction/testdata/vectors.json`).

## Installation

```bash
pip install golem-team-sdk              # Core (LocalDisk only)
pip install golem-team-sdk[aws]         # + S3 backend (boto3)
pip install golem-team-sdk[gcp]         # + GCS backend (google-cloud-storage)
pip install golem-team-sdk[azure]       # + Azure Blob backend (azure-storage-blob)
pip install golem-team-sdk[all]         # All storage backends
pip install golem-team-sdk[langchain]   # + LangChain adapter (langchain-core)
pip install golem-team-sdk[openai-agents]  # + OpenAI Agents adapter
pip install golem-team-sdk[dev,all]     # Dev + all backends (for contributors)
```

The `crewai`, `llamaindex`, and `semantic-kernel` adapters intentionally have **no extras** — they duck-type the framework's callback payload so the adapter module imports cleanly even when the framework isn't installed. Only frameworks that require a hard import (LangChain's `BaseCallbackHandler`, OpenAI Agents' `TracingProcessor` ABC) ship as extras.

## Framework integrations

The `golem_sdk.integrations.*` modules adapt popular agent frameworks to Golem's `intercept_execution` call. Each is one-class, ~200 lines, fail-open (telemetry exceptions never break the user's chain).

| Framework | Module | Quickstart |
|---|---|---|
| **LangChain** | `golem_sdk.integrations.langchain.GolemCallbackHandler` | [docs/langchain-quickstart.md](docs/langchain-quickstart.md) |
| **OpenAI Agents SDK** | `golem_sdk.integrations.openai_agents.GolemTraceProcessor` | [docs/openai-agents-quickstart.md](docs/openai-agents-quickstart.md) |
| **CrewAI** | `golem_sdk.integrations.crewai.GolemStepListener`, `GolemTaskListener` | [docs/crewai-quickstart.md](docs/crewai-quickstart.md) |
| **LlamaIndex** | `golem_sdk.integrations.llama_index.GolemCallbackHandler` | [docs/llamaindex-quickstart.md](docs/llamaindex-quickstart.md) |
| **Semantic Kernel** | `golem_sdk.integrations.semantic_kernel.GolemFunctionInvocationFilter` | [docs/semantic-kernel-quickstart.md](docs/semantic-kernel-quickstart.md) |

All adapters share a few invariants, enforced in code: tool arguments are bounded (each string value capped at 64 KB, dict capped at 100 keys) before reaching the SDK, telemetry exceptions are logged at WARN and swallowed, and no adapter logs sensitive payload contents.

### Blocking a tool call before it runs

Recording is fire-and-forget; enforcement is not. When the SDK talks to the Golem agent
(`GolemSDK(witness_url=...)`), `WitnessClient.check_execution(tool_name, ...)` asks
the Golem agent on this machine synchronously whether a call may run, through the same CEL rules
and guard profile the MCP interceptor and the Claude Code hook evaluate. It returns a
`PolicyDecision` whose action is `allow`, `alert` or `block` (a synchronous call
cannot hold a human gate, so a `gate` rule comes back as `block`), and it **fails open**: a Golem agent that cannot
be reached lets the call proceed and logs why, because a Golem outage must not take a
customer's agent down.

Two adapters wire it in behind an `enforce` flag that is **off by default**, so
adding Golem never changes an agent's behaviour until you ask it to:

```python
from golem_sdk.integrations.langchain import GolemCallbackHandler
from golem_sdk.policy import GolemPolicyBlocked

handler = GolemCallbackHandler(sdk, enforce=True)
# A blocked call raises GolemPolicyBlocked(tool_name, reason) before the tool runs.
```

`GolemFunctionInvocationFilter(sdk, enforce=True)` does the same for Semantic
Kernel, by not invoking the next filter. The CrewAI, LlamaIndex and OpenAI Agents
adapters record only; call `check_execution` yourself if you need them to refuse.

Which rules block is decided in the console, not in your code: a rule authored by a
person blocks when its action is `block`; a rule the defender harness proposed blocks
only after the safety-net worker has measured it in shadow and promoted it, or a
human approved it.

## Quick Start

The recommended setup sends events to the Golem agent: `GolemSDK(witness_url="http://localhost:39393")`
signs with COSE_Sign1 and hands the entry to the local Golem agent, which owns BYOB
storage, and `intercept_execution` returns `bom_hash`, `cose_payload` and
`cbor_payload`. The example below is the deprecated legacy mode, in which the SDK
stores the payload itself:

```python
from golem_sdk import GolemSDK
from golem_sdk.storage import S3Storage

sdk = GolemSDK(storage=S3Storage('my-bucket', 'boms/', 'us-west-2'))

result = sdk.intercept_execution(
    agent_id='support-bot-v2',
    mcp_server_id='srv-jira-prod',
    tool_name='SearchTickets',
    tool_arguments={'query': 'auth failures', 'max_results': 10},
    system_prompt='You are an IT support assistant.',
    retrieved_chunks=[
        {'id': 'c1', 'content': 'Refund policy text...', 'source_uri': 'rag://policies/refunds'},
    ],
)

print(result['bom_hash'])      # SHA-256 hex (64 chars)
print(result['storage_uri'])   # s3://my-bucket/boms/<hash>.json
print(result['signature'])     # ECDSA signature (bytes)
```

## Storage Backends

| Backend | Class | URI Format |
|---|---|---|
| Local Disk | `LocalDiskStorage` | `file:///tmp/golem-byob/{hash}.json` |
| AWS S3 | `S3Storage` | `s3://{bucket}/{prefix}{hash}.json` |
| Google Cloud | `GCSStorage` | `gs://{bucket}/{prefix}{hash}.json` |
| Azure Blob | `AzureBlobStorage` | `https://{account}.blob.core.windows.net/...` |

Cloud backends use lazy initialization — they only import the cloud SDK when first accessed. The cloud backends are deprecated along with legacy mode: BYOB storage has moved to the Relay.

## Evidence CLI

The package includes a CLI for verifying transparency log integrity (requires a running PostgreSQL):

```bash
golem-evidence verify-chain --from 1 --to 100
golem-evidence verify --entry-hash <64-char-hex>
golem-evidence export --entry-hash <64-char-hex>
golem-evidence audit-report --from 1 --to 100 --output report.json
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `GOLEM_DB_HOST` | `localhost` | PostgreSQL host |
| `GOLEM_DB_PORT` | `5432` | PostgreSQL port |
| `GOLEM_DB_NAME` | `golem_ledger` | Database name |
| `GOLEM_DB_USER` | `golem` | Database user |
| `GOLEM_DB_PASSWORD` | *(warns if unset)* | Database password |

## Files

| File | Purpose |
|---|---|
| `src/golem_sdk/__init__.py` | Package exports (absolute imports) |
| `src/golem_sdk/sdk.py` | `GolemSDK` class: validate → CBOR → SHA-256 → ECDSA → store |
| `src/golem_sdk/storage.py` | `StorageBackend` ABC + 4 implementations with path traversal protection |
| `src/golem_sdk/witness_client.py` | `WitnessClient`: witness-mode delivery, the synchronous `check_execution`, and the redaction policy fetch |
| `src/golem_sdk/redaction.py` | Redaction before signing: the recognizers, operators and the cached machine policy |
| `src/golem_sdk/policy.py` | `GolemPolicyBlocked`, raised by an enforcing adapter |
| `src/golem_sdk/evidence_cli.py` | CLI with input validation, bounded queries, safe file output |
| `src/golem_sdk/integrations/` | The framework adapters above |
| `tests/test_sdk.py` | 25 tests: instantiation, hashing, signing, storage, optional fields |
| `tests/test_storage.py` | 7 tests: LocalDiskStorage CRUD + path traversal rejection |
| `tests/test_transparency_log.py` | 10 tests: hash chain integrity + tamper detection (no DB needed) |
| `tests/test_redaction.py` | The Go engine's redaction vectors, policy parsing and caching, and redaction before signing |

## Testing

```bash
cd sdk-py
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,all]"
pytest tests/ -v   # ~165 tests, no external deps needed
```

## Security

- All `hash_id` parameters validated as `^[a-f0-9]{64}$` before file operations
- File paths resolved and verified to stay within base directory (path traversal protection)
- Files written with `0o600` permissions; directories created with `0o700`
- Input validation on all public API fields (type, length, format)
- Payload size capped at 10 MB
- Error messages sanitized — log `type(e).__name__`, not full exception details
- Absolute imports used throughout (`from golem_sdk.storage import ...`)
