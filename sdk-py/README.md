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

The `crewai`, `llamaindex`, and `semantic-kernel` adapters intentionally have **no extras** — their modules import cleanly without the framework installed, and import it, if at all, only when you register or wrap something. Only frameworks that require a hard import (LangChain's `BaseCallbackHandler`, OpenAI Agents' `TracingProcessor` ABC) ship as extras.

## Framework integrations

The `golem_sdk.integrations.*` modules adapt popular agent frameworks to Golem's `intercept_execution` call, and those that can refuse a call to `check_execution`. Each is fail-open: telemetry exceptions never break the user's chain.

| Framework | Use | With `enforce=True`, a blocked call… | Results and usage | Quickstart |
|---|---|---|---|---|
| **LangChain** | `GolemToolMiddleware` for `create_agent` agents; `GolemCallbackHandler` for tools and chains | does not run. The middleware hands the model `Blocked by Golem policy: <reason>`; the handler raises `GolemPolicyBlocked` | Each result is recorded against its call; the middleware replaces a withheld one with the notice, the handler can only record it. Model usage from `usage_metadata` | [docs/langchain-quickstart.md](docs/langchain-quickstart.md) |
| **OpenAI Agents SDK** | `golem_guard_tools(tools, sdk, ...)`, tool input and output guardrails; `GolemRunHooks` for usage; `GolemTraceProcessor` records from traces, after each tool ran, and cannot block | does not run; the model gets `Blocked by Golem policy: <reason>` as the tool's output | Each result is recorded against its call by the output guardrail, which hands the model the notice for a withheld one (the trace processor only records). Model usage via `GolemRunHooks` | [docs/openai-agents-quickstart.md](docs/openai-agents-quickstart.md) |
| **CrewAI** | `GolemToolHooks`, CrewAI's process-wide before/after tool-call hooks; `GolemTaskListener` for task completions | does not run; the model gets `Blocked by Golem policy: <reason>` as the tool's result | Each result is recorded against its call by the after hook, which hands the model the notice for a withheld one. Model usage from CrewAI's `LLMCallCompletedEvent` | [docs/crewai-quickstart.md](docs/crewai-quickstart.md) |
| **LlamaIndex** | `golem_guard_tools(tools, sdk, ...)` wraps the tools; `GolemCallbackHandler` records queries | does not run; the tool raises `GolemPolicyBlocked`, whose message an agent hands the model | Each result is recorded against its call, and a withheld one is replaced by the notice. No usage: LlamaIndex reports tokens in no provider-neutral place | [docs/llamaindex-quickstart.md](docs/llamaindex-quickstart.md) |
| **Semantic Kernel** | `GolemFunctionInvocationFilter`, a function-invocation filter | does not run; its result, which `kernel.invoke` returns and the model reads, is `Blocked by Golem policy: <reason>` | Each result is recorded against its call, and a withheld one is replaced by the notice. No usage: the filter does not see the chat-completion calls that report it | [docs/semantic-kernel-quickstart.md](docs/semantic-kernel-quickstart.md) |

All adapters share a few invariants: telemetry exceptions are logged at WARN and swallowed, and no adapter logs sensitive payload contents.

`intercept_execution` bounds the tool arguments it signs rather than dropping the call: each string is cut at 64 KB and all of them together at 1 MB (a cut string ends in `...[truncated]`), a dict over 100 keys keeps its first 99 plus `_golem_truncated_keys` with the count dropped, a list keeps 1,000 items, nesting stops at 16 levels, and a value CBOR cannot encode (an object, bytes, a set) is recorded as its text. Strings are cut after redaction, so a cut never splits a secret the redaction would have found.

### Blocking a tool call before it runs

Recording is fire-and-forget; enforcement is not. When the SDK talks to the Golem agent
(`GolemSDK(witness_url=...)`), `WitnessClient.check_execution(tool_name, ...)` asks
the Golem agent on this machine synchronously whether a call may run, through the same CEL rules
and guard profile the MCP interceptor and the Claude Code hook evaluate. It returns a
`PolicyDecision` whose action is `none` (no rule matched), `allow`, `alert` or `block`, with `allowed` false only for `block` (a synchronous call
cannot hold a human gate, so a `gate` rule comes back as `block`), and it **fails open**: a Golem agent that cannot
be reached lets the call proceed and logs why, because a Golem outage must not take a
customer's agent down.

The adapters that can refuse a call wire it in behind an `enforce` flag that is
**off by default**, so adding Golem never changes an agent's behaviour until you ask
it to. They send the call's arguments and their server (`mcp_server_id`, the
framework's name unless you set it), and the table above says what the caller sees
when a call is blocked:

```python
from golem_sdk.integrations.langchain import GolemToolMiddleware

agent = create_agent(model, tools, middleware=[
    GolemToolMiddleware(sdk, agent_id="support-bot", enforce=True),
])
# A blocked call does not run; the model reads "Blocked by Golem policy: <reason>".
```

Outside the adapters, call `sdk.check_execution(tool_name, tool_arguments=...,
mcp_server_id=...)` yourself before a call you need to be able to refuse.

Which rules block is decided in the console, not in your code: a rule authored by a
person blocks when its action is `block`; a rule the defender harness proposed blocks
only after the safety-net worker has measured it in shadow and promoted it, or a
human approved it.

## Quick Start

The recommended setup sends events to the Golem agent: `GolemSDK(witness_url="http://localhost:39393")`
signs with COSE_Sign1 and hands the entry to the local Golem agent, which owns BYOB
storage. The agent checks that signature, then signs the ledger entry with the
machine's key: the entry's `payload_hash` is the SDK's `bom_hash`, the signed bytes go
to your storage, and the SDK's own signature stays on the entry as `sdk_cose_sign1`.
Every call from one `GolemSDK` is recorded under one session (`sdk.session_id`);
pass `session_id=` to group a run under an id of your own.
`intercept_execution` returns `bom_hash`, `cose_payload` and
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

## Results and model usage

A call is half the story. With the Golem agent, record what the tool returned and
what each model call used:

```python
sdk = GolemSDK(witness_url="http://localhost:39393")

call = sdk.intercept_execution("support-bot", "jira", "get_ticket", {"id": "OPS-12"})
output = get_ticket("OPS-12")
verdict = sdk.record_result("get_ticket", output, tool_use_id=call["tool_use_id"],
                            agent_id="support-bot", mcp_server_id="jira")
reply = verdict.notice if verdict.withheld else output   # what the model reads

sdk.record_usage("claude-sonnet-5-5", input_tokens=1200, output_tokens=85, provider="anthropic")
```

- `record_result` pairs the result with its call on `tool_use_id`, so the console
  shows them as one step. It waits for the answer, because the Golem agent checks the
  result for prompt injection as `mcp__<mcp_server_id>__<tool_name>`, the name the
  inventory gives the tool. When a rule blocks on that check, `withheld` is true and
  `notice` is the text to give the model instead. When a rule advises, `advice` is
  the text to give it alongside the result. It fails open: an unreachable agent
  answers allow, with `reachable=False`.
- `record_usage` sends one model call's tokens in the background. They count in
  the token views. Golem does not price a model an SDK names.

The [framework integrations](#framework-integrations) make both calls for you, where
the framework exposes what they need.

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
| `src/golem_sdk/policy.py` | `GolemPolicyBlocked`, raised by an enforcing adapter, and the refusal text an adapter hands the model |
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
