# Wrapping LangChain with Golem

Record every tool call a LangChain agent makes, before it runs, and refuse the ones your Golem policy blocks.

**Time to set up: ~1 minute** (assuming you already have the Golem agent running).

Tested against `langchain-core` 1.6 and `langchain` 1.4.

---

## Install

```bash
pip install 'golem-team-sdk[langchain]'   # langchain-core, for the callback handler
pip install langchain                     # also needed for the agent middleware
```

## Agents built with `create_agent`: the middleware

```python
from langchain.agents import create_agent
from golem_sdk import GolemSDK
from golem_sdk.integrations.langchain import GolemToolMiddleware

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice-laptop", "env": "dev"},
)

agent = create_agent(
    model,
    tools=my_tools,
    middleware=[GolemToolMiddleware(sdk, agent_id="support-bot-v2")],
)
```

The middleware records each tool call, with its arguments, before the tool runs.

## Tools and chains run directly: the callback handler

```python
from golem_sdk.integrations.langchain import GolemCallbackHandler

handler = GolemCallbackHandler(sdk, agent_id="support-bot-v2")

chain.invoke({"input": "..."}, config={"callbacks": [handler]})   # per call
chain = chain.with_config(callbacks=[handler])                    # bound to a runnable
```

The handler records every `tool.invoke` / `ainvoke`, including ones nested in LCEL chains, before the tool runs.

Use the middleware or the handler for a run, not both: attached together they record each call twice.

## Blocking a call

Both take `enforce=True` (off by default). The SDK then asks the Golem agent before each tool runs, sending the tool name, its arguments and the server (`langchain` unless you pass `mcp_server_id=`), and a call your policy blocks does not run:

| | What the caller sees |
|---|---|
| `GolemToolMiddleware(sdk, agent_id=..., enforce=True)` | The model gets a tool message with status `error` and the content `Blocked by Golem policy: <reason>`, and the agent carries on, free to try something else. |
| `GolemCallbackHandler(sdk, agent_id=..., enforce=True)` | `GolemPolicyBlocked(tool_name, reason)` is raised from `tool.invoke` before the tool body runs. Inside an agent loop this ends the run, unless its `ToolNode` was built with `handle_tool_errors` covering `GolemPolicyBlocked`; use the middleware for agents. |

A Golem agent that cannot be reached lets the call run (fail open), as everywhere in the SDK.

## Results, and a result Golem withholds

After each tool runs, both record what it returned (or the error it raised) against the same call — the agent's tool-call id when there is one — with how long it took. The Golem agent checks the result for prompt injection as `mcp__langchain__<tool>`, and a rule can withhold it:

| | A withheld result… |
|---|---|
| `GolemToolMiddleware` | reaches the model as the Golem agent's notice instead of the result. No `enforce` needed: the rules on the Golem agent decide. |
| `GolemCallbackHandler` | is recorded, but still reaches the caller: a LangChain callback cannot change a tool's output. Use the middleware if a result must be kept from the model. |

## Model usage

Both record each model call's token counts (input, output, cache reads and writes) from the message's `usage_metadata`, under the model name the provider reported (`response_metadata["model_name"]`). A model that reports neither is not recorded; nothing is estimated. If your model traffic already goes through the Golem agent's Anthropic / OpenAI proxies, which count it there, pass `record_usage=False` so it is not counted twice.

## What gets captured

| Event | Default | Use case |
|---|---|---|
| Tool calls | ✅ on | Every tool call, with its arguments, before it runs |
| Tool results | ✅ on | What each call returned, paired with it |
| Model usage | ✅ on | Token counts per model call; `record_usage=False` turns it off |
| LLM calls (`on_llm_start`, handler only) | ❌ off | Already captured at the HTTP layer via the Golem agent's Anthropic / OpenAI proxy; turn on with `capture_llm_calls=True` only for pure-SDK deployments |

Every call from one `GolemSDK` is recorded under one session (`sdk.session_id`). Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Failure model

Telemetry failures **never** propagate up. If the Golem agent is unreachable, the SDK is misconfigured, or the Golem stack is having a bad day, the adapter logs at `WARN` and the agent continues. The only things that change what the agent sees are a policy block with `enforce=True` and, through the middleware, a withheld result.

## Verifying it works

After your agent runs once, hit the context-server MCP:

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"support-bot-v2"}}}' \
  | python -m json.tool
```

You should see one entry per tool call your agent made.
