"""The Semantic Kernel adapter driven through a real Kernel against a fake Golem agent."""

import asyncio

import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel  # noqa: E402
from semantic_kernel.functions import KernelArguments, kernel_function  # noqa: E402

from golem_sdk.integrations.semantic_kernel import GolemFunctionInvocationFilter  # noqa: E402
from tests.integrations.conftest import only_entry  # noqa: E402


class _Opaque:
    def __str__(self) -> str:
        return "opaque-value"


def _invoke(golem, sdk, enforce: bool, **args):
    class Math:
        @kernel_function(name="add", description="Add two numbers")
        def add(self, a: int, b: int) -> int:
            golem.ran("Math.add")
            return a + b

    kernel = Kernel()
    kernel.add_plugin(Math(), plugin_name="Math")
    kernel.add_filter("function_invocation",
                      GolemFunctionInvocationFilter(sdk, agent_id="sk-bot", enforce=enforce))
    return asyncio.run(kernel.invoke(plugin_name="Math", function_name="add",
                                     arguments=KernelArguments(**args)))


def test_filter_records_the_real_arguments_before_running(golem):
    sdk = golem.sdk()

    assert _invoke(golem, sdk, False, a=1, b=2).value == 3

    entry = only_entry(golem, "Math.add")
    assert entry["bom"]["tool_arguments"] == {"a": 1, "b": 2}
    assert entry["mcp_server_id"] == "semantic-kernel"
    assert entry["session_id"] == sdk.session_id
    assert golem.events == [("recorded", "Math.add"), ("ran", "Math.add"), ("result", "Math.add")]
    [result] = golem.results
    assert result["tool_use_id"] == golem.call_ids[0]
    assert result["result"] == 3
    assert result["mcp_server_id"] == "semantic-kernel"
    assert result["is_error"] is False


def test_a_withheld_result_is_replaced_by_the_notice(golem):
    golem.withhold("[Golem withheld this result]")

    result = _invoke(golem, golem.sdk(), False, a=1, b=2)

    assert ("ran", "Math.add") in golem.events
    assert str(result) == "[Golem withheld this result]"
    assert golem.results[0]["tool_use_id"] == golem.call_ids[0]


def test_a_raising_function_is_recorded_as_an_error(golem):
    class Broken:
        @kernel_function(name="explode", description="Fails")
        def explode(self) -> str:
            raise RuntimeError("disk on fire")

    kernel = Kernel()
    kernel.add_plugin(Broken(), plugin_name="Broken")
    kernel.add_filter("function_invocation",
                      GolemFunctionInvocationFilter(golem.sdk(), agent_id="sk-bot"))
    with pytest.raises(Exception):
        asyncio.run(kernel.invoke(plugin_name="Broken", function_name="explode"))

    [result] = golem.results
    assert result["is_error"] is True
    assert "disk on fire" in result["result"]


def test_filter_block_returns_the_reason_as_the_result(golem):
    golem.block("adding is not allowed here")

    result = _invoke(golem, golem.sdk(), True, a=5, b=6)

    assert result is not None
    assert str(result) == "Blocked by Golem policy: adding is not allowed here"
    assert ("ran", "Math.add") not in golem.events
    assert golem.checks[0]["tool_arguments"] == {"a": 5, "b": 6}
    assert golem.checks[0]["mcp_server_id"] == "semantic-kernel"


def test_an_argument_cbor_cannot_encode_still_records_the_call(golem):
    _invoke(golem, golem.sdk(), False, a=1, b=2, extra=_Opaque())

    entry = only_entry(golem, "Math.add")
    assert entry["bom"]["tool_arguments"] == {"a": 1, "b": 2, "extra": "opaque-value"}
