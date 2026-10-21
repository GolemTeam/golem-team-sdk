"""The OpenAI Agents adapter driven through a real Runner against a fake Golem agent."""

import time

import pytest

pytest.importorskip("agents")

from agents import Agent, Runner, function_tool, set_trace_processors  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.tracing import response_span  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from golem_sdk.integrations.openai_agents import GolemTraceProcessor  # noqa: E402
from tests.integrations.conftest import only_entry  # noqa: E402


class _FakeModel(Model):
    """Calls add(3, 4) on its first turn and answers on its second, like a real
    Responses model it opens a response span and sets the response on it."""

    model = "gpt-fake"

    def __init__(self):
        self.inputs = []

    async def get_response(self, system_instructions, input, *args, **kwargs):
        self.inputs.append(input)
        with response_span() as span:
            if len(self.inputs) == 1:
                out = [ResponseFunctionToolCall(
                    type="function_call", id="fc1", call_id="call-1", name="add",
                    arguments='{"a": 3, "b": 4}', status="completed")]
            else:
                out = [ResponseOutputMessage(
                    type="message", id="m1", role="assistant", status="completed",
                    content=[ResponseOutputText(type="output_text", text="done", annotations=[])])]
            span.span_data.response = Response.model_construct(
                id=f"resp-{len(self.inputs)}", model="gpt-fake", output=out)
        return ModelResponse(output=out, response_id=None,
                             usage=Usage(requests=1, input_tokens=20, output_tokens=5, total_tokens=25))

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def _add_tool(golem):
    @function_tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        golem.ran("add")
        return a + b

    return add


@pytest.fixture(autouse=True)
def processors():
    installed = []
    set_trace_processors(installed)

    def install(p):
        installed.append(p)
        set_trace_processors(installed)

    yield install
    set_trace_processors([])


def _tool_outputs(model):
    return [item["output"] for item in model.inputs[-1]
            if isinstance(item, dict) and item.get("type") == "function_call_output"]


def test_trace_processor_records_the_real_arguments(golem, processors):
    sdk = golem.sdk()
    processors(GolemTraceProcessor(sdk, agent_id="oa-agent"))
    agent = Agent(name="support", tools=[_add_tool(golem)], model=_FakeModel())

    result = Runner.run_sync(agent, "add 3 and 4")

    assert result.final_output == "done"
    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 3, "b": 4}
    assert entry["mcp_server_id"] == "openai-agents"
    assert entry["session_id"] == sdk.session_id
    for _ in range(100):
        if golem.results:
            break
        time.sleep(0.02)
    [result] = golem.results
    assert result["tool_use_id"] == golem.call_ids[0]
    assert result["result"] == 7


def test_trace_processor_names_the_generation_model(golem, processors):
    processors(GolemTraceProcessor(golem.sdk(), agent_id="oa-agent", capture_generations=True))
    agent = Agent(name="support", tools=[_add_tool(golem)], model=_FakeModel())

    Runner.run_sync(agent, "add 3 and 4")

    names = [e["tool_name"] for e in golem.wait_entries(3)]
    assert names.count("llm.gpt-fake") == 2
    assert "llm.unknown" not in names


def test_guardrail_records_before_the_tool_runs(golem):
    from golem_sdk.integrations.openai_agents import golem_guard_tools

    sdk = golem.sdk()
    tools = golem_guard_tools([_add_tool(golem)], sdk, agent_id="oa-agent", enforce=True)
    agent = Agent(name="support", tools=tools, model=_FakeModel())

    assert Runner.run_sync(agent, "add 3 and 4").final_output == "done"

    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 3, "b": 4}
    assert entry["mcp_server_id"] == "openai-agents"
    assert golem.events == [("recorded", "add"), ("ran", "add"), ("result", "add")]
    assert golem.checks[0]["tool_arguments"] == {"a": 3, "b": 4}
    assert golem.checks[0]["mcp_server_id"] == "openai-agents"
    assert golem.checks[0]["session_id"] == sdk.session_id
    assert golem.call_ids == ["call-1"]
    [result] = golem.results
    assert result["tool_use_id"] == "call-1"
    assert result["result"] == 7
    assert result["mcp_server_id"] == "openai-agents"


def test_guardrail_hands_the_model_a_withheld_results_notice(golem):
    from golem_sdk.integrations.openai_agents import golem_guard_tools

    golem.withhold("[Golem withheld this result]")
    model = _FakeModel()
    tools = golem_guard_tools([_add_tool(golem)], golem.sdk(), agent_id="oa-agent")
    agent = Agent(name="support", tools=tools, model=model)

    assert Runner.run_sync(agent, "add 3 and 4").final_output == "done"

    assert ("ran", "add") in golem.events
    assert _tool_outputs(model) == ["[Golem withheld this result]"]


def test_run_hooks_record_model_usage(golem):
    from golem_sdk.integrations.openai_agents import GolemRunHooks

    agent = Agent(name="support", tools=[_add_tool(golem)], model=_FakeModel())

    Runner.run_sync(agent, "add 3 and 4", hooks=GolemRunHooks(golem.sdk(), agent_id="oa-agent"))

    usage = golem.wait_usage(2)
    assert len(usage) == 2
    assert usage[0]["model"] == "gpt-fake"
    assert (usage[0]["input_tokens"], usage[0]["output_tokens"]) == (20, 5)
    assert usage[0]["agent_id"] == "oa-agent"


def test_guardrail_block_stops_the_tool_and_tells_the_model(golem):
    from golem_sdk.integrations.openai_agents import golem_guard_tools

    golem.block("adding is not allowed here")
    model = _FakeModel()
    tools = golem_guard_tools([_add_tool(golem)], golem.sdk(), agent_id="oa-agent",
                              enforce=True)
    agent = Agent(name="support", tools=tools, model=model)

    assert Runner.run_sync(agent, "add 3 and 4").final_output == "done"

    assert ("ran", "add") not in golem.events
    assert _tool_outputs(model) == ["Blocked by Golem policy: adding is not allowed here"]


def test_guarding_twice_adds_one_guardrail(golem):
    from golem_sdk.integrations.openai_agents import golem_guard_tools

    sdk = golem.sdk()
    tools = [_add_tool(golem)]
    golem_guard_tools(tools, sdk, agent_id="oa-agent")
    golem_guard_tools(tools, sdk, agent_id="oa-agent")

    assert len(tools[0].tool_input_guardrails) == 1
    assert len(tools[0].tool_output_guardrails) == 1
