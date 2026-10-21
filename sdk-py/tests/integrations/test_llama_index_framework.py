"""The LlamaIndex adapter driven through real LlamaIndex against a fake Golem agent."""

import asyncio
from typing import Any, List

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.agent.workflow import ReActAgent  # noqa: E402
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload  # noqa: E402
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata  # noqa: E402
from llama_index.core.llms.callbacks import llm_completion_callback  # noqa: E402
from llama_index.core.tools import FunctionTool  # noqa: E402
from llama_index.core.workflow import Context  # noqa: E402

from golem_sdk.integrations.llama_index import GolemCallbackHandler  # noqa: E402
from golem_sdk.policy import GolemPolicyBlocked  # noqa: E402
from tests.integrations.conftest import only_entry  # noqa: E402


class _ReActFake(CustomLLM):
    prompts: List[str] = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=False)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            text = 'Thought: I need to use a tool.\nAction: add\nAction Input: {"a": 3, "b": 4}'
        else:
            text = "Thought: I can answer without using any more tools.\nAnswer: 7"
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        text = self.complete(prompt, formatted, **kwargs).text
        yield CompletionResponse(text=text, delta=text)


def _add_tool(golem):
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        golem.ran("add")
        return a + b

    return FunctionTool.from_defaults(fn=add)


def _run_agent(tools) -> _ReActFake:
    llm = _ReActFake(prompts=[])
    agent = ReActAgent(tools=tools, llm=llm)

    async def go():
        return await agent.run("add 3 and 4")

    asyncio.run(go())
    return llm


def test_guarded_tool_records_the_real_arguments_before_running(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    sdk = golem.sdk()
    [add] = golem_guard_tools([_add_tool(golem)], sdk, agent_id="rag-bot")

    assert add.call(a=1, b=2).raw_output == 3

    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 1, "b": 2}
    assert entry["mcp_server_id"] == "llamaindex"
    assert entry["session_id"] == sdk.session_id
    assert golem.events == [("recorded", "add"), ("ran", "add"), ("result", "add")]
    [result] = golem.results
    assert result["tool_use_id"] == golem.call_ids[0]
    assert result["result"] == 3
    assert result["mcp_server_id"] == "llamaindex"
    assert result["is_error"] is False


def test_guarded_tool_block_raises_the_reason_without_running(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    golem.block("adding is not allowed here")
    [add] = golem_guard_tools([_add_tool(golem)], golem.sdk(), agent_id="rag-bot", enforce=True)

    with pytest.raises(GolemPolicyBlocked) as exc:
        asyncio.run(add.acall(a=1, b=2))

    assert exc.value.reason == "adding is not allowed here"
    assert ("ran", "add") not in golem.events
    assert golem.checks[0]["tool_arguments"] == {"a": 1, "b": 2}
    assert golem.checks[0]["mcp_server_id"] == "llamaindex"


def test_agent_records_a_guarded_call_before_it_runs(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    llm = _run_agent(golem_guard_tools([_add_tool(golem)], golem.sdk(), agent_id="rag-bot"))

    assert "Observation: 7" in llm.prompts[-1]
    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 3, "b": 4}
    assert golem.events == [("recorded", "add"), ("ran", "add"), ("result", "add")]


def test_agent_hands_the_model_a_withheld_results_notice(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    golem.withhold("[Golem withheld this result]")
    llm = _run_agent(golem_guard_tools([_add_tool(golem)], golem.sdk(), agent_id="rag-bot"))

    assert ("ran", "add") in golem.events
    assert "[Golem withheld this result]" in llm.prompts[-1]
    assert "Observation: 7" not in llm.prompts[-1]
    assert golem.results[0]["tool_use_id"] == golem.call_ids[0]


def test_a_raising_tool_is_recorded_as_an_error(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    def fail(a: int) -> int:
        """Fail."""
        raise RuntimeError("disk on fire")

    [tool] = golem_guard_tools([FunctionTool.from_defaults(fn=fail)], golem.sdk(), agent_id="rag-bot")
    with pytest.raises(RuntimeError):
        tool.call(a=1)

    [result] = golem.results
    assert result["is_error"] is True
    assert result["result"] == "RuntimeError: disk on fire"


def test_a_tool_that_is_not_a_function_tool_is_guarded_too(golem):
    from llama_index.core.tools import AsyncBaseTool, ToolMetadata, ToolOutput

    from golem_sdk.integrations.llama_index import golem_guard_tools

    class Lookup(AsyncBaseTool):
        @property
        def metadata(self):
            return ToolMetadata(name="lookup", description="Look something up.")

        def call(self, query: str) -> ToolOutput:
            golem.ran("lookup")
            return ToolOutput(content=f"found {query}", tool_name="lookup",
                              raw_input={"query": query}, raw_output=query)

        async def acall(self, query: str) -> ToolOutput:
            return self.call(query)

    golem.withhold("[Golem withheld this result]")
    [tool] = golem_guard_tools([Lookup()], golem.sdk(), agent_id="rag-bot")

    out = asyncio.run(tool.acall(query="golem"))

    assert out.content == "[Golem withheld this result]"
    assert only_entry(golem, "lookup")["bom"]["tool_arguments"] == {"query": "golem"}
    assert golem.events == [("recorded", "lookup"), ("ran", "lookup"), ("result", "lookup")]


def test_agent_block_stops_the_tool_and_tells_the_model(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    golem.block("adding is not allowed here")
    llm = _run_agent(golem_guard_tools([_add_tool(golem)], golem.sdk(), agent_id="rag-bot",
                                       enforce=True))

    assert ("ran", "add") not in golem.events
    assert "adding is not allowed here" in llm.prompts[-1]


def test_agent_still_hands_a_context_tool_its_context(golem):
    from golem_sdk.integrations.llama_index import golem_guard_tools

    seen = []

    async def add(ctx: Context, a: int, b: int) -> int:
        """Add two numbers."""
        seen.append(isinstance(ctx, Context))
        golem.ran("add")
        return a + b

    tools = golem_guard_tools([FunctionTool.from_defaults(async_fn=add)], golem.sdk(),
                              agent_id="rag-bot")
    _run_agent(tools)

    assert seen == [True]
    assert only_entry(golem, "add")["bom"]["tool_arguments"] == {"a": 3, "b": 4}


def test_callback_handler_reads_the_bare_arguments_of_a_function_call_event(golem):
    handler = GolemCallbackHandler(golem.sdk(), agent_id="rag-bot")
    manager = CallbackManager([handler])
    tool = _add_tool(golem)

    with manager.event(CBEventType.FUNCTION_CALL, payload={
        EventPayload.FUNCTION_CALL: '{"a": 1, "b": 2}',
        EventPayload.TOOL: tool.metadata,
    }) as event:
        event.on_end(payload={EventPayload.FUNCTION_OUTPUT: "3"})

    assert only_entry(golem, "add")["bom"]["tool_arguments"] == {"a": 1, "b": 2}
