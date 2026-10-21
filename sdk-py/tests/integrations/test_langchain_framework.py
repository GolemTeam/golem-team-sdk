"""The LangChain adapter driven through real LangChain against a fake Golem agent."""

import asyncio

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from golem_sdk.integrations.langchain import GolemCallbackHandler  # noqa: E402
from golem_sdk.policy import GolemPolicyBlocked  # noqa: E402
from tests.integrations.conftest import only_entry  # noqa: E402


def _add_tool(golem):
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        golem.ran("add")
        return a + b

    return add


class _ToolCallingFake(GenericFakeChatModel):
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


_USAGE = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
          "input_token_details": {"cache_read": 4}}


def _model_calling_add(args):
    return _ToolCallingFake(
        seen=[],
        messages=iter([
            AIMessage(content="", tool_calls=[{"name": "add", "args": args, "id": "call-1"}],
                      usage_metadata=_USAGE, response_metadata={"model_name": "fake-model"}),
            AIMessage(content="done", usage_metadata=_USAGE,
                      response_metadata={"model_name": "fake-model"}),
        ]),
    )


def test_tool_invoke_records_real_arguments_before_running(golem):
    sdk = golem.sdk()
    add = _add_tool(golem)
    handler = GolemCallbackHandler(sdk, agent_id="lc-agent")

    assert add.invoke({"a": 1, "b": 2}, config={"callbacks": [handler]}) == 3

    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 1, "b": 2}
    assert entry["mcp_server_id"] == "langchain"
    assert entry["session_id"] == sdk.session_id
    assert golem.events == [("recorded", "add"), ("ran", "add"), ("result", "add")]
    [result] = golem.results
    assert result["tool_use_id"] == golem.call_ids[0]
    assert result["result"] == 3
    assert result["mcp_server_id"] == "langchain"
    assert result["is_error"] is False


def test_handler_records_a_raising_tool_as_an_error(golem):
    @tool
    def explode(a: int) -> int:
        """Fail."""
        raise RuntimeError("disk on fire")

    handler = GolemCallbackHandler(golem.sdk(), agent_id="lc-agent")
    with pytest.raises(RuntimeError):
        explode.invoke({"a": 1}, config={"callbacks": [handler]})

    [result] = golem.results
    assert result["is_error"] is True
    assert result["result"] == "RuntimeError: disk on fire"


def test_handler_records_model_usage(golem):
    handler = GolemCallbackHandler(golem.sdk(), agent_id="lc-agent")
    model = _model_calling_add({"a": 1, "b": 2})

    model.invoke([HumanMessage("hi")], config={"callbacks": [handler]})

    [usage] = golem.wait_usage(1)
    assert usage["model"] == "fake-model"
    assert (usage["input_tokens"], usage["output_tokens"], usage["cache_read_tokens"]) == (12, 3, 4)
    assert usage["agent_id"] == "lc-agent"


def test_enforce_block_stops_the_tool_and_raises_the_reason(golem):
    golem.block("adding is not allowed here")
    add = _add_tool(golem)
    handler = GolemCallbackHandler(golem.sdk(), agent_id="lc-agent", enforce=True)

    with pytest.raises(GolemPolicyBlocked) as exc:
        add.invoke({"a": 5, "b": 6}, config={"callbacks": [handler]})

    assert exc.value.reason == "adding is not allowed here"
    assert ("ran", "add") not in golem.events
    assert golem.wait_entries(0, timeout=0) == []


def test_enforce_block_stops_the_tool_on_ainvoke(golem):
    golem.block("adding is not allowed here")
    add = _add_tool(golem)
    handler = GolemCallbackHandler(golem.sdk(), agent_id="lc-agent", enforce=True)

    with pytest.raises(GolemPolicyBlocked):
        asyncio.run(add.ainvoke({"a": 7, "b": 8}, config={"callbacks": [handler]}))

    assert ("ran", "add") not in golem.events


def test_policy_check_carries_arguments_and_server_not_a_repr(golem):
    sdk = golem.sdk()
    add = _add_tool(golem)
    handler = GolemCallbackHandler(sdk, agent_id="lc-agent", enforce=True)

    add.invoke({"a": 5, "b": 6}, config={"callbacks": [handler]})

    assert len(golem.checks) == 1
    check = golem.checks[0]
    assert check["tool_name"] == "add"
    assert check["tool_arguments"] == {"a": 5, "b": 6}
    assert check["mcp_server_id"] == "langchain"
    assert check["command"] == ""
    assert check["session_id"] == sdk.session_id
    only_entry(golem, "add")


def test_agent_middleware_hands_the_refusal_to_the_model(golem):
    pytest.importorskip("langchain.agents")
    from langchain.agents import create_agent

    from golem_sdk.integrations.langchain import GolemToolMiddleware

    golem.block("adding is not allowed here")
    add = _add_tool(golem)
    model = _model_calling_add({"a": 3, "b": 4})
    agent = create_agent(
        model, tools=[add],
        middleware=[GolemToolMiddleware(golem.sdk(), agent_id="lc-agent", enforce=True)],
    )

    out = agent.invoke({"messages": [HumanMessage("add 3 and 4")]})

    assert out["messages"][-1].content == "done"
    assert ("ran", "add") not in golem.events
    tool_msgs = [m for m in model.seen[-1] if m.type == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    assert "Blocked by Golem policy: adding is not allowed here" in tool_msgs[0].content
    assert golem.checks[0]["tool_arguments"] == {"a": 3, "b": 4}
    assert golem.checks[0]["mcp_server_id"] == "langchain"


def test_agent_middleware_records_an_allowed_call_before_it_runs(golem):
    pytest.importorskip("langchain.agents")
    from langchain.agents import create_agent

    from golem_sdk.integrations.langchain import GolemToolMiddleware

    add = _add_tool(golem)
    agent = create_agent(
        _model_calling_add({"a": 3, "b": 4}), tools=[add],
        middleware=[GolemToolMiddleware(golem.sdk(), agent_id="lc-agent", enforce=True)],
    )

    asyncio.run(agent.ainvoke({"messages": [HumanMessage("add 3 and 4")]}))

    entry = only_entry(golem, "add")
    assert entry["bom"]["tool_arguments"] == {"a": 3, "b": 4}
    assert golem.events == [("recorded", "add"), ("ran", "add"), ("result", "add")]
    assert len(golem.checks) == 1
    assert golem.call_ids == ["call-1"]
    assert golem.results[0]["tool_use_id"] == "call-1"
    assert golem.results[0]["result"] == "7"


def test_agent_middleware_hands_the_model_a_withheld_results_notice(golem):
    pytest.importorskip("langchain.agents")
    from langchain.agents import create_agent

    from golem_sdk.integrations.langchain import GolemToolMiddleware

    golem.withhold("[Golem withheld this result]")
    model = _model_calling_add({"a": 3, "b": 4})
    agent = create_agent(model, tools=[_add_tool(golem)],
                         middleware=[GolemToolMiddleware(golem.sdk(), agent_id="lc-agent")])

    agent.invoke({"messages": [HumanMessage("add 3 and 4")]})

    assert ("ran", "add") in golem.events
    [tool_msg] = [m for m in model.seen[-1] if m.type == "tool"]
    assert tool_msg.content == "[Golem withheld this result]"
    assert golem.results[0]["tool_use_id"] == golem.call_ids[0]


def test_agent_middleware_records_model_usage(golem):
    pytest.importorskip("langchain.agents")
    from langchain.agents import create_agent

    from golem_sdk.integrations.langchain import GolemToolMiddleware

    agent = create_agent(_model_calling_add({"a": 3, "b": 4}), tools=[_add_tool(golem)],
                         middleware=[GolemToolMiddleware(golem.sdk(), agent_id="lc-agent")])

    agent.invoke({"messages": [HumanMessage("add 3 and 4")]})

    usage = golem.wait_usage(2)
    assert len(usage) == 2
    assert {u["model"] for u in usage} == {"fake-model"}
    assert usage[0]["input_tokens"] == 12 and usage[0]["output_tokens"] == 3
