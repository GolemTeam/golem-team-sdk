# Wrapping Semantic Kernel with Golem

Capture every plugin function invocation a Semantic Kernel agent makes by registering a Golem filter on the kernel's function-invocation pipeline.

**Time to set up: ~1 minute.**

---

## Install

```bash
pip install golem-team-sdk
```

No extra needed: the Semantic Kernel adapter has **no hard dependency on the `semantic_kernel` package**. It duck-types the `FunctionInvocationContext`, so it works across SK versions without a version pin.

## Wire up the filter

```python
from semantic_kernel import Kernel
from golem_sdk import GolemSDK
from golem_sdk.integrations.semantic_kernel import (
    GolemFunctionInvocationFilter,
)

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)

kernel = Kernel()
kernel.add_filter(
    "function_invocation",
    GolemFunctionInvocationFilter(sdk, agent_id="sk-bot"),
)
```

That's it. From here on, every `kernel.invoke(...)` (and every nested plugin call inside it) generates a Golem record before the function runs.

## What gets captured

| Semantic Kernel event | Golem event | Fires |
|---|---|---|
| `function_invocation` (any plugin function) | `intercept_execution(tool_name="<plugin>.<function>")` | Once per function call |

LLM round trips are **not** captured by this adapter — they're handled at the HTTP layer via the Golem agent's Anthropic / OpenAI proxies, so emitting them here would double-count.

Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Tool naming

The Golem `tool_name` is the fully qualified `{plugin_name}.{function_name}` SK uses for addressing — for example, `Weather.get_current` — so you can slice the ledger by plugin in the console. When a function isn't grouped under a plugin, the bare function name is recorded.

## Robustness

The filter accepts both the modern SK context shape (`context.function.name` / `context.function.plugin_name`) and the older flat shape (`context.function_name` / `context.plugin_name`). It also accepts plain dicts in case SK ever changes the wire format.

## Failure model

Telemetry failures **never** propagate to the kernel pipeline. The filter wraps the SDK call in `try/except` and logs at `WARN`. Critically, `await next(context)` is **always** called even if the Golem record fails — so the user's plugin function runs regardless.

Exceptions raised by the user's plugin (via `next`) are intentionally **not** caught — they surface to the caller normally so kernel-level error handling keeps working.

## Verifying it works

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"sk-bot"}}}' \
  | python -m json.tool
```

You'll see one event per plugin function call.
