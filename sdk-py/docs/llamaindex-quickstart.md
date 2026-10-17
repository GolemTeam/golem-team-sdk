# Wrapping LlamaIndex with Golem

Capture every tool call and top-level query a LlamaIndex agent makes by registering a Golem callback handler with the global `CallbackManager`.

**Time to set up: ~1 minute.**

---

## Install

```bash
pip install golem-team-sdk
```

No extra needed: the LlamaIndex adapter has **no hard dependency on the `llama_index` package**. It duck-types the `BaseCallbackHandler` interface, so it works across llama-index versions that have shuffled internal class layout.

## Wire up the handler

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from golem_sdk import GolemSDK
from golem_sdk.integrations.llama_index import GolemCallbackHandler

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)

Settings.callback_manager = CallbackManager(
    [GolemCallbackHandler(sdk, agent_id="rag-bot")]
)
```

You can also attach the handler to a specific index, query engine, or agent if you want narrower scoping — just pass a fresh `CallbackManager([handler])` to its constructor.

## What gets captured

| LlamaIndex event | Golem event | Fires |
|---|---|---|
| `CBEventType.FUNCTION_CALL` | `intercept_execution(tool_name=<tool>)` | Per tool invocation |
| `CBEventType.QUERY` | `intercept_execution(tool_name="llamaindex.query")` | Per top-level query |
| `LLM`, `AGENT_STEP`, `RETRIEVE`, others | — | Silently ignored |

LLM round trips are **not** captured by this adapter — they're handled at the HTTP layer via the Golem agent's Anthropic / OpenAI proxies. Capturing them here would double-count every LLM call.

Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Robustness

LlamaIndex's `FUNCTION_CALL` payload has shifted shape across releases. The handler accepts all of these:

- `payload["tool"]` as `ToolMetadata` (Pydantic) or `dict`
- `payload["function_call"]` as a JSON string (`{"name": "...", "arguments": ...}`)
- `payload["function_call"]` as a dict
- `payload["arguments"]` as a top-level dict
- Nested `arguments` as JSON string, dict, or scalar (wrapped under `"input"`)

If LlamaIndex ships a new shape, the handler silently skips that event rather than crashing the agent.

## Failure model

Telemetry failures **never** propagate to the agent loop. `on_event_start` wraps the SDK call in `try/except` and logs at `WARN`. The handler always returns the original `event_id` so LlamaIndex's trace correlation keeps working.

## Verifying it works

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"rag-bot"}}}' \
  | python -m json.tool
```

You'll see one event per tool call plus one `llamaindex.query` event per top-level query.
