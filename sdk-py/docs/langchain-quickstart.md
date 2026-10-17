# Wrapping LangChain with Golem

Capture every tool call a LangChain agent makes — across `AgentExecutor`, LCEL chains, and standalone `Tool.invoke` paths — and route the pre-execution state through the cryptographic chain.

**Time to set up: ~1 minute** (assuming you already have the Golem agent running).

---

## Install

```bash
pip install 'golem-team-sdk[langchain]'
```

Pulls `langchain-core` as a transitive dep. If you already have `langchain` installed, the resolver will reuse it.

## Wire up the callback handler

```python
from golem_sdk import GolemSDK
from golem_sdk.integrations.langchain import GolemCallbackHandler

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice-laptop", "env": "dev"},
)
handler = GolemCallbackHandler(
    sdk,
    agent_id="support-bot-v2",
    mcp_server_id="langchain",
)
```

## Three ways to attach it

### Per-call (recommended for ad-hoc instrumentation)

```python
chain.invoke({"input": "..."}, config={"callbacks": [handler]})
```

### Per-agent (recommended for production)

```python
from langchain.agents import AgentExecutor

agent = AgentExecutor.from_agent_and_tools(
    agent=my_agent,
    tools=my_tools,
    callbacks=[handler],
)
```

### Bound to a runnable

```python
chain = chain.with_config(callbacks=[handler])
```

Every later `invoke` / `stream` / `batch` on the returned runnable carries the handler.

## What gets captured

| Event | Default | Use case |
|---|---|---|
| Tool calls (`on_tool_start`) | ✅ on | Every `Tool.invoke` — including ones nested in LCEL chains |
| LLM calls (`on_llm_start`) | ❌ off | Already captured at the HTTP layer via the Golem agent's Anthropic / OpenAI proxy; turn on only for pure-SDK deployments |

Enable LLM capture explicitly:

```python
handler = GolemCallbackHandler(
    sdk, agent_id="bot", capture_llm_calls=True,
)
```

Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Failure model

Telemetry failures **never** propagate up. If the Golem agent is unreachable, the SDK is misconfigured, or the Golem stack is having a bad day, the handler logs at `WARN` and the customer's chain continues. There is no path where instrumenting your agent with Golem can break the agent.

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

You should see one entry per `Tool.invoke()` your chain made.
