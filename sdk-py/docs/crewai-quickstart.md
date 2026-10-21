# Wrapping CrewAI with Golem

Record every tool call a CrewAI crew makes, before it runs, and refuse the ones your Golem policy blocks, through CrewAI's tool-call hooks.

**Time to set up: ~1 minute.**

Tested against `crewai` 1.15, on both of its tool-calling paths: native function calling (the default for a model that supports it) and the ReAct text protocol.

---

## Install

```bash
pip install golem-team-sdk
```

No extra needed: the adapter imports `crewai` only when you register the hooks, so it adds no version pin. It needs a crewai that has `crewai.hooks`.

## Register the hooks

```python
from crewai import Crew
from golem_sdk import GolemSDK
from golem_sdk.integrations.crewai import GolemToolHooks, GolemTaskListener

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)

hooks = GolemToolHooks(sdk, agent_id="research-crew").register()

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    task_callback=GolemTaskListener(sdk, agent_id="research-crew"),
)
result = crew.kickoff()
```

CrewAI's tool-call hooks are **process-wide**: once registered, they see every crew's tool calls until `hooks.unregister()`. To scope them to one run:

```python
with GolemToolHooks(sdk, agent_id="research-crew"):
    crew.kickoff()
```

## Blocking a call

Pass `enforce=True` (off by default). The before-tool hook then asks the Golem agent before each tool runs, sending the tool name, its arguments and the server (`crewai` unless you pass `mcp_server_id=`). A call your policy blocks does not run: the model reads `Blocked by Golem policy: <reason>` as the tool's result, and the crew carries on. A Golem agent that cannot be reached lets the call run (fail open).

```python
hooks = GolemToolHooks(sdk, agent_id="research-crew", enforce=True).register()
```

## A result Golem withholds

The after-tool hook records what each tool returned, against the same call, with how long it took. The Golem agent checks the result for prompt injection as `mcp__crewai__<tool>`; when a rule withholds it, the hook replaces the result with the agent's notice, so the model reads the notice instead. No `enforce` needed: the rules on the Golem agent decide. CrewAI hands the hook the tool's result as text and does not say whether the tool failed, so results are recorded with `is_error` false.

## Model usage

The hooks also record each model call's tokens (prompt, completion, cached and cache-write tokens) from CrewAI's `LLMCallCompletedEvent`, under the model CrewAI names, for the LLMs that report usage there (CrewAI's built-in providers do). Nothing is estimated. If your model traffic already goes through the Golem agent's Anthropic / OpenAI proxies, which count it there, pass `record_usage=False`.

## What gets captured

| Event | Source | Fires |
|---|---|---|
| Tool call, with its arguments | `GolemToolHooks` | Once per tool call, before the tool runs |
| Tool result | `GolemToolHooks` | Once per tool call, after it runs |
| Model usage | `GolemToolHooks` | Once per model call that reports usage |
| Task completion (`crewai.task`) | `GolemTaskListener` | Once per task |

Every call from one `GolemSDK` is recorded under one session (`sdk.session_id`). Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## `GolemStepListener` is deprecated

The earlier `step_callback` listener misses most tool calls: CrewAI does not call `step_callback` for native function calling, and on the ReAct path calls it after the tool has already run. It still works, with a `DeprecationWarning`; replace it with `GolemToolHooks`, and don't use both, or ReAct calls are recorded twice.

## Failure model

Telemetry failures **never** propagate to the crew loop. The hooks and the task listener wrap SDK calls in `try/except` and log at `WARN`; the only things that change what the model sees are a policy block with `enforce=True` and a withheld result.

## Verifying it works

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"research-crew"}}}' \
  | python -m json.tool
```

You'll see one event per tool call plus one `crewai.task` event per completed task.
