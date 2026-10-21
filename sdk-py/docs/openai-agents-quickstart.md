# Wrapping OpenAI Agents SDK with Golem

Record every function-tool call an OpenAI Agents SDK agent makes, before it runs, and refuse the ones your Golem policy blocks — across `Runner.run`, `Runner.run_sync`, and streamed runs.

**Time to set up: ~1 minute** (assuming the Golem agent is running).

Tested against `openai-agents` 0.24.

---

## Install

```bash
pip install 'golem-team-sdk[openai-agents]'
```

Pulls `openai-agents` as a transitive dep.

## Guard the tools

```python
from agents import Agent, Runner
from golem_sdk import GolemSDK
from golem_sdk.integrations.openai_agents import golem_guard_tools

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)

agent = Agent(
    name="support",
    tools=golem_guard_tools([lookup_order, refund], sdk, agent_id="support-bot"),
)
result = Runner.run_sync(agent, "Where is my order?")
```

`golem_guard_tools` adds a Golem [tool input guardrail and tool output guardrail](https://openai.github.io/openai-agents-python/guardrails/) to each function tool, in place, and returns the list. The input guardrail runs before the tool and records the call with its arguments; the output guardrail records what the tool returned, against the same call (the model's tool-call id), with how long it took. Hosted tools (web search, file search, code interpreter) run at OpenAI, take no guardrails and are left alone. For an MCP server's tools, pass the pair to the server yourself:

```python
from golem_sdk.integrations.openai_agents import golem_tool_guardrails

golem = golem_tool_guardrails(sdk, agent_id="support-bot", mcp_server_id="my-server")
server = MCPServerStdio(params, tool_input_guardrails=[golem.input],
                        tool_output_guardrails=[golem.output])
```

## A result Golem withholds

The Golem agent checks each result for prompt injection as `mcp__openai-agents__<tool>`. When a rule withholds it, the output guardrail rejects the result and the model receives the agent's notice as the tool's output instead. No `enforce` needed: the rules on the Golem agent decide. The output guardrail cannot tell an error the Agents SDK turned into text from a normal result, so results are recorded with `is_error` false.

## Model usage

```python
from golem_sdk.integrations.openai_agents import GolemRunHooks

result = Runner.run_sync(agent, "Where is my order?", hooks=GolemRunHooks(sdk, agent_id="support-bot"))
```

`GolemRunHooks` records each model call's input, output and cached tokens from the response's `usage`, under the model named on the `Agent` (a string, or a model object's `model`). An agent with no model name of its own is not recorded; nothing is estimated. If the traffic already goes through the Golem agent's OpenAI proxy, which counts it there, leave the hooks out.

## Blocking a call

Pass `enforce=True` (off by default). The guardrail then asks the Golem agent before each tool runs, sending the tool name, its arguments and the server (`openai-agents` unless you pass `mcp_server_id=`). A call your policy blocks does not run: the guardrail rejects it, the model receives `Blocked by Golem policy: <reason>` as the tool's output, and the run carries on, free to try something else. A Golem agent that cannot be reached lets the call run (fail open).

```python
tools = golem_guard_tools(my_tools, sdk, agent_id="support-bot", enforce=True)
```

## Recording from traces instead

`GolemTraceProcessor` records from the Agents SDK's tracing pipeline and needs no change to the tools, but it **cannot block or withhold**, and it records each call and its result **after** the tool has run: the SDK only fills in a function span's arguments once the tool returns, so the processor records when the span ends. The arguments are there only while the run traces sensitive data (`RunConfig.trace_include_sensitive_data`, on unless `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA` turns it off).

```python
from agents import add_trace_processor
from golem_sdk.integrations.openai_agents import GolemTraceProcessor

add_trace_processor(GolemTraceProcessor(sdk, agent_id="support-bot"))
```

`add_trace_processor` keeps the default OpenAI trace export; `set_trace_processors([...])` replaces it. Use the guardrail or the processor for the same tools, not both: together they record each call twice.

| Span type | Default | Notes |
|---|---|---|
| `FunctionSpanData` (tool calls) | ✅ on | The call and its output, recorded when the span ends, after the tool ran |
| `GenerationSpanData` / `ResponseSpanData` (model calls) | ❌ opt-in | Recorded as `llm.<model>`. Already captured at the HTTP layer by the Golem agent's OpenAI proxy; turn on with `capture_generations=True` only without it |
| Handoffs, guardrails, agent spans | ❌ off | Not mapped |

Every call from one `GolemSDK` is recorded under one session (`sdk.session_id`). Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Failure model

Telemetry failures **never** propagate up. The guardrails, the hooks and the processor wrap every record in `try/except` and log at `WARN`. A broken Golem agent can't crash `Runner.run`; the only things that change what the model sees are a policy block with `enforce=True` and a withheld result.

## Verifying it works

After a run, query the context-server:

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"support-bot"}}}' \
  | python -m json.tool
```

You should see one entry per function-tool the run invoked.
