# Wrapping LlamaIndex with Golem

Record every tool call a LlamaIndex agent makes, before it runs, and what it returned; refuse the calls your Golem policy blocks, and keep a result Golem withholds from the model.

**Time to set up: ~1 minute.**

Tested against `llama-index-core` 0.14 (workflow agents such as `ReActAgent` and `FunctionAgent`, and direct `tool.call` / `acall`).

---

## Install

```bash
pip install golem-team-sdk
```

No extra needed: the adapter imports `llama_index` only when you wrap tools.

## Wrap the tools

```python
from llama_index.core.agent.workflow import FunctionAgent
from golem_sdk import GolemSDK
from golem_sdk.integrations.llama_index import golem_guard_tools

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)

agent = FunctionAgent(
    tools=golem_guard_tools([search_tool, refund_tool], sdk, agent_id="rag-bot"),
    llm=llm,
)
```

`golem_guard_tools` returns wrapped copies and leaves your tools as they are. A `FunctionTool` stays a `FunctionTool`, so a tool that takes the workflow `Context` still gets it; any other tool (a `QueryEngineTool`, your own `BaseTool`) is wrapped in one that forwards to it.

Each call is recorded with its arguments before the tool runs, under the server `llamaindex` unless you pass `mcp_server_id=`. After it runs, what it returned (or the error it raised) is recorded against the same call, with how long it took.

Why wrap the tools rather than register a callback: `llama-index-core` 0.14 no longer emits the `FUNCTION_CALL` callback event, so a callback handler never sees a tool call.

## Blocking a call

Pass `enforce=True` (off by default). The wrapper then asks the Golem agent before each call, sending the tool name, its arguments and the server. A call your policy blocks does not run: the tool raises `GolemPolicyBlocked(tool_name, reason)`. A LlamaIndex agent turns a tool's exception into its output, so the model reads `Golem policy blocked <tool>: <reason>` and carries on; a direct `tool.call()` raises to your code. A Golem agent that cannot be reached lets the call run (fail open).

## A result Golem withholds

The Golem agent checks each result for prompt injection as `mcp__llamaindex__<tool>`. When a rule withholds it, the wrapper returns the agent's notice in place of the result, so the model reads the notice and never the result. A `withhold` needs no `enforce`; it is decided by the rules on the Golem agent.

## Queries

`GolemCallbackHandler` records each top-level query as `llamaindex.query`:

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from golem_sdk.integrations.llama_index import GolemCallbackHandler

Settings.callback_manager = CallbackManager([GolemCallbackHandler(sdk, agent_id="rag-bot")])
```

On LlamaIndex releases older than 0.14 it also records `FUNCTION_CALL` events, so there use it or the wrapped tools for tool calls, not both.

## Model usage

Not recorded by this adapter. LlamaIndex has no provider-neutral place that reports a model call's token counts (each LLM integration puts them in its own raw response), and estimating them would record numbers no provider reported. Model traffic through the Golem agent's Anthropic / OpenAI proxies is counted there.

Every call from one `GolemSDK` is recorded under one session (`sdk.session_id`). Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Failure model

Telemetry failures **never** propagate to the agent loop: recording failures are logged at `WARN` and the tool runs. The only things that change what the agent sees are a policy block with `enforce=True` and a withheld result.

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
