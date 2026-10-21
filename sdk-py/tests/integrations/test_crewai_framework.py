"""The CrewAI adapter driven through a real Crew against a fake Golem agent.

Both of CrewAI's tool-calling paths are covered: native function calling (the
default for a model that supports it) and the ReAct text protocol.
"""

import os

import pytest

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
pytest.importorskip("crewai")

from crewai import Agent, BaseLLM, Crew, Task  # noqa: E402
from crewai.events.types.llm_events import LLMCallType  # noqa: E402
from crewai.tools import tool  # noqa: E402

from tests.integrations.conftest import only_entry  # noqa: E402


class _FakeLLM(BaseLLM):
    def __init__(self, native: bool):
        super().__init__(model="fake-model")
        self._native = native
        self.calls = []

    def supports_function_calling(self) -> bool:
        return self._native

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs):
        self.calls.append(messages)
        first = len(self.calls) == 1
        self._emit_call_completed_event(
            response="", call_type=LLMCallType.LLM_CALL, from_task=from_task,
            from_agent=from_agent,
            usage={"prompt_tokens": 30, "completion_tokens": 6, "cached_prompt_tokens": 10},
        )
        if self._native:
            if first:
                return [{"id": "call-1", "type": "function",
                         "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}}]
            return "7"
        if first:
            return 'Thought: I should add.\nAction: add\nAction Input: {"a": 3, "b": 4}'
        return "Thought: I now know the final answer\nFinal Answer: 7"

    def saw(self, text: str) -> bool:
        return any(text in str(m) for m in self.calls[-1])


def _run_crew(golem, native: bool) -> _FakeLLM:
    @tool("add")
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        golem.ran("add")
        return a + b

    llm = _FakeLLM(native)
    agent = Agent(role="adder", goal="add numbers", backstory="b", tools=[add], llm=llm)
    task = Task(description="add 3 and 4", expected_output="a number", agent=agent)
    Crew(agents=[agent], tasks=[task]).kickoff()
    return llm


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
def test_hooks_record_the_real_arguments_before_the_tool_runs(golem, native):
    from golem_sdk.integrations.crewai import GolemToolHooks

    sdk = golem.sdk()
    with GolemToolHooks(sdk, agent_id="crew-a"):
        _run_crew(golem, native)

    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 3, "b": 4}
    assert entry["mcp_server_id"] == "crewai"
    assert entry["session_id"] == sdk.session_id
    assert golem.events == [("recorded", "add"), ("ran", "add"), ("result", "add")]
    [result] = golem.results
    assert result["tool_use_id"] == golem.call_ids[0]
    assert result["result"] == "7"
    assert result["mcp_server_id"] == "crewai"


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
def test_hooks_hand_the_model_a_withheld_results_notice(golem, native):
    from golem_sdk.integrations.crewai import GolemToolHooks

    golem.withhold("[Golem withheld this result]")
    with GolemToolHooks(golem.sdk(), agent_id="crew-a"):
        llm = _run_crew(golem, native)

    assert ("ran", "add") in golem.events
    assert llm.saw("[Golem withheld this result]")
    assert golem.results[0]["tool_use_id"] == golem.call_ids[0]


def test_hooks_record_model_usage(golem):
    from golem_sdk.integrations.crewai import GolemToolHooks

    with GolemToolHooks(golem.sdk(), agent_id="crew-a"):
        _run_crew(golem, native=True)

    usage = golem.wait_usage(2)
    assert len(usage) == 2
    assert usage[0]["model"] == "fake-model"
    assert (usage[0]["input_tokens"], usage[0]["output_tokens"],
            usage[0]["cache_read_tokens"]) == (30, 6, 10)
    assert usage[0]["agent_id"] == "crew-a"


@pytest.mark.parametrize("native", [True, False], ids=["native", "react"])
def test_hooks_block_stops_the_tool_and_tells_the_model(golem, native):
    from golem_sdk.integrations.crewai import GolemToolHooks

    golem.block("adding is not allowed here")
    with GolemToolHooks(golem.sdk(), agent_id="crew-a", enforce=True):
        llm = _run_crew(golem, native)

    assert ("ran", "add") not in golem.events
    assert llm.saw("Blocked by Golem policy: adding is not allowed here")
    assert golem.checks[0]["tool_arguments"] == {"a": 3, "b": 4}
    assert golem.checks[0]["mcp_server_id"] == "crewai"


def test_hooks_leave_later_crews_alone_once_unregistered(golem):
    from golem_sdk.integrations.crewai import GolemToolHooks

    golem.block("adding is not allowed here")
    with GolemToolHooks(golem.sdk(), agent_id="crew-a", enforce=True):
        pass
    _run_crew(golem, native=True)

    assert ("ran", "add") in golem.events
    assert golem.checks == []
