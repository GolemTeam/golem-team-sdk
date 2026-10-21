# Wrapping Semantic Kernel with Golem

Record every kernel function a Semantic Kernel agent calls, before it runs, and what it returned; refuse the calls your Golem policy blocks, and keep a result Golem withholds from the model. One filter on the kernel's function-invocation pipeline does it all.

**Time to set up: ~1 minute.**

Tested against `semantic-kernel` 1.36.

---

## Install

```bash
pip install golem-team-sdk
```

No extra needed: the adapter imports `semantic_kernel` only to build a result when it replaces one.

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

Semantic Kernel runs the filter around every kernel function: `kernel.invoke(...)`, nested plugin calls, and the functions the model calls through automatic function calling. Each call is recorded with its arguments before the function runs, under the server `semantic-kernel` unless you pass `mcp_server_id=`. After it runs, what it returned (or the error it raised) is recorded against the same call, with how long it took. A streamed invocation's result is not recorded: reading it would consume the stream.

## Blocking a call

Pass `enforce=True` (off by default). The filter then asks the Golem agent before each function runs, sending the function name, its arguments and the server. A call your policy blocks does not run: its result is `Blocked by Golem policy: <reason>`, which is what `kernel.invoke` returns and what the model reads when it called the function. A Golem agent that cannot be reached lets the call run (fail open).

## A result Golem withholds

The Golem agent checks each result for prompt injection as `mcp__semantic-kernel__<plugin>.<function>`. When a rule withholds it, the filter replaces the result with the agent's notice, so neither `kernel.invoke`'s caller nor the model sees the original. A `withhold` needs no `enforce`; it is decided by the rules on the Golem agent.

## Model usage

Not recorded by this adapter. The function-invocation filter does not see the chat-completion calls automatic function calling makes, which is where Semantic Kernel reports token usage. Model traffic through the Golem agent's Anthropic / OpenAI proxies is counted there.

Every call from one `GolemSDK` is recorded under one session (`sdk.session_id`). Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Tool naming

The Golem `tool_name` is the fully qualified `{plugin_name}.{function_name}` SK uses for addressing — for example, `Weather.get_current` — so you can slice the ledger by plugin in the console. When a function isn't grouped under a plugin, the bare function name is recorded.

## Failure model

Telemetry failures **never** propagate to the kernel pipeline: a failed record is logged at `WARN` and the function runs. Exceptions raised by the function itself are recorded and then re-raised, so kernel-level error handling keeps working. The only things that change a result are a policy block with `enforce=True` and a withheld result.

## Verifying it works

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"sk-bot"}}}' \
  | python -m json.tool
```

You'll see one event per kernel function call.
