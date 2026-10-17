# Wrapping CrewAI with Golem

Capture every step a CrewAI crew takes — tool invocations and task completions — by passing Golem listeners as `step_callback` and `task_callback`.

**Time to set up: ~1 minute.**

---

## Install

```bash
pip install golem-team-sdk
```

No extra needed: the CrewAI adapter has **no hard dependency on the `crewai` package**. It duck-types the callback payload, so it works across crewai versions without a version pin.

## Wire up the listeners

```python
from crewai import Crew, Agent, Task
from golem_sdk import GolemSDK
from golem_sdk.integrations.crewai import (
    GolemStepListener, GolemTaskListener,
)

sdk = GolemSDK(
    witness_url="http://localhost:39393",
    labels={"workspace": "alice", "env": "dev"},
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    step_callback=GolemStepListener(sdk, agent_id="research-crew"),
    task_callback=GolemTaskListener(sdk, agent_id="research-crew"),
)

result = crew.kickoff()
```

You can attach the step listener to individual agents instead of the whole crew if you want per-agent scoping:

```python
researcher = Agent(
    role="researcher",
    goal="…",
    step_callback=GolemStepListener(sdk, agent_id="researcher"),
)
```

## What gets captured

| Event | Listener | Fires |
|---|---|---|
| Tool invocation (during agent reasoning) | `GolemStepListener` | Once per `AgentAction` step |
| Task completion | `GolemTaskListener` | Once per task |
| Final answer (`AgentFinish`) | — | Silently skipped (no tool intent) |

LLM round trips are **not** captured by this adapter — they're handled at the HTTP layer via the Golem agent's Anthropic / OpenAI proxies.

Before an event is signed, the SDK removes what this machine's redaction packs name: email addresses, phone and card numbers, keys, tokens and the like. Names and street addresses need a model and are not found by the SDK. See [Redaction before signing](../README.md#redaction-before-signing).

## Robustness

CrewAI has shipped multiple callback payload shapes across minor releases. The adapter handles all of them:

- `step.tool` (modern flat shape)
- `step.tool_name` (alt naming)
- `step.action.tool` (older `AgentAction` wrapper)
- `step.tool_input` as `dict`, JSON string, or empty
- Step payload as Pydantic object **or** plain dict

If CrewAI ships a new shape, the listener silently skips that step rather than crashing.

## Failure model

Telemetry failures **never** propagate to the crew loop. Both listeners wrap SDK calls in `try/except` and log at `WARN`.

## Verifying it works

```bash
curl -s -X POST http://localhost:3003/ \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"get_recent_activity",
                 "arguments":{"minutes":10,"agent_id":"research-crew"}}}' \
  | python -m json.tool
```

You'll see one event per step (tool used) plus one `crewai.task` event per completed task.
