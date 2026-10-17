# Wrapping OpenAI Agents SDK with Golem

Capture every function-tool call from an OpenAI Agents SDK agent — across `Runner.run`, `Runner.run_sync`, and streamed runs — by registering a `TracingProcessor` that forwards to Golem.

**Time to set up: ~1 minute** (assuming the Golem agent is running).

---

## Install

```bash
pip install 'golem-team-sdk[openai-agents]'
```

Pulls `openai-agents` as a transitive dep.

## Wire up the tracing processor

```python
from agents import Agent, Runner, set_trace_processors
from golem_sdk import GolemSDK
from golem_sdk.integrations.openai_agents import GolemTraceProcessor

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)
set_trace_processors([GolemTraceProcessor(sdk, agent_id="support-bot")])

agent = Agent(name="support", tools=[my_tool])
result = Runner.run_sync(agent, "How do I reset my password?")
```

`set_trace_processors([...])` **replaces** the default processors. If you want Golem alongside the default console / OpenAI trace sinks, use `add_trace_processor(...)` from the same module.

## What gets captured

| Span type | Default | Notes |
|---|---|---|
| `FunctionSpanData` (tool calls) | ✅ on | Pre-execution capture, matches Golem's PreToolUse semantics |
| `GenerationSpanData` / `ResponseSpanData` (LLM calls) | ❌ opt-in | Already captured at the HTTP layer by the Golem agent's OpenAI proxy; double-capture inflates the chain |
| Handoffs, guardrails, agent spans | ❌ off | Not currently mapped — out of scope for v1 |

Enable LLM capture explicitly when running without the HTTP proxy:

```python
processor = GolemTraceProcessor(
    sdk, agent_id="bot", capture_generations=True,
)
```

Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Failure model

Telemetry failures **never** propagate up. The processor wraps every emit in `try/except` and logs at `WARN`. A broken Golem agent can't crash `Runner.run`.

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
