"""Tests for the Semantic Kernel function-invocation filter.

The adapter has no hard dependency on ``semantic_kernel`` — it
duck-types the FunctionInvocationContext shape — so these tests
run without any ``importorskip``. We use ``asyncio.run`` to drive
the async filter rather than pulling in ``pytest-asyncio``.
"""

import asyncio
from unittest.mock import MagicMock

from golem_sdk import GolemSDK
from golem_sdk.integrations.semantic_kernel import (
    GolemFunctionInvocationFilter,
    _extract_tool_name,
    _extract_arguments,
)


def _mock_sdk():
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution = MagicMock()
    return sdk


class FakeFunction:
    """Stand-in for SK's KernelFunctionMetadata."""

    def __init__(self, name: str, plugin_name: str = ""):
        self.name = name
        self.plugin_name = plugin_name


class FakeContext:
    """Stand-in for SK's FunctionInvocationContext."""

    def __init__(self, function=None, arguments=None, **kwargs):
        self.function = function
        self.arguments = arguments
        for k, v in kwargs.items():
            setattr(self, k, v)


def _run(coro):
    return asyncio.run(coro)


def _async_noop():
    """Return an async no-op that records it was awaited."""
    state = {"called": False, "context": None}

    async def _next(ctx):
        state["called"] = True
        state["context"] = ctx

    return _next, state


# ----------------------------------------------------------------------
# Filter behavior
# ----------------------------------------------------------------------


def test_filter_records_intercept_then_invokes_next():
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, state = _async_noop()
    ctx = FakeContext(
        function=FakeFunction(name="get_weather", plugin_name="Weather"),
        arguments={"city": "Berlin"},
    )
    _run(f(ctx, nxt))
    assert state["called"] is True
    assert state["context"] is ctx
    sdk.intercept_execution.assert_called_once()
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "Weather.get_weather"
    assert call.kwargs["tool_arguments"] == {"city": "Berlin"}
    assert call.kwargs["agent_id"] == "sk-bot"
    assert call.kwargs["mcp_server_id"] == "semantic-kernel"


def test_filter_uses_bare_function_name_when_no_plugin():
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, _ = _async_noop()
    ctx = FakeContext(
        function=FakeFunction(name="get_time"),
        arguments={},
    )
    _run(f(ctx, nxt))
    assert sdk.intercept_execution.call_args.kwargs["tool_name"] == "get_time"


def test_filter_supports_older_top_level_field_names():
    """Older SK releases exposed function_name/plugin_name on the
    context itself, not nested under .function. Verify fallback."""
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, _ = _async_noop()
    ctx = FakeContext(
        function=None,
        arguments={"x": 1},
        function_name="get_time",
        plugin_name="Clock",
    )
    _run(f(ctx, nxt))
    assert (
        sdk.intercept_execution.call_args.kwargs["tool_name"] == "Clock.get_time"
    )


def test_filter_with_no_function_name_skips_intercept_but_still_calls_next():
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, state = _async_noop()
    ctx = FakeContext(function=None, arguments={})
    _run(f(ctx, nxt))
    sdk.intercept_execution.assert_not_called()
    assert state["called"] is True


def test_filter_handles_kernelarguments_mapping():
    """SK's KernelArguments is a Mapping subclass, not a plain dict.
    Verify dict(args) conversion works."""
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, _ = _async_noop()

    class FakeKernelArgs:
        def __init__(self, d):
            self._d = d

        def keys(self):
            return self._d.keys()

        def __getitem__(self, k):
            return self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    ctx = FakeContext(
        function=FakeFunction(name="echo"),
        arguments=FakeKernelArgs({"msg": "hi"}),
    )
    _run(f(ctx, nxt))
    assert sdk.intercept_execution.call_args.kwargs["tool_arguments"] == {"msg": "hi"}


def test_filter_swallows_sdk_exceptions_and_still_calls_next():
    """Telemetry failures must never short-circuit the function call."""
    sdk = _mock_sdk()
    sdk.intercept_execution.side_effect = ValueError("boom")
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, state = _async_noop()
    ctx = FakeContext(function=FakeFunction(name="x"), arguments={})
    # Must not raise.
    _run(f(ctx, nxt))
    assert state["called"] is True


def test_filter_propagates_exception_from_next():
    """We don't catch exceptions raised by next() — the original
    function's failure must surface to the caller normally."""
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")

    async def boom(ctx):
        raise RuntimeError("plugin crashed")

    ctx = FakeContext(function=FakeFunction(name="x"), arguments={})
    raised = False
    try:
        _run(f(ctx, boom))
    except RuntimeError:
        raised = True
    assert raised


def test_filter_with_dict_context_works():
    """A future SK version handing us a dict instead of a Pydantic
    model should still extract correctly."""
    sdk = _mock_sdk()
    f = GolemFunctionInvocationFilter(sdk, agent_id="sk-bot")
    nxt, _ = _async_noop()
    ctx = {
        "function": {"name": "fs_read", "plugin_name": "FS"},
        "arguments": {"path": "/tmp"},
    }
    _run(f(ctx, nxt))
    assert sdk.intercept_execution.call_args.kwargs["tool_name"] == "FS.fs_read"
    assert (
        sdk.intercept_execution.call_args.kwargs["tool_arguments"]
        == {"path": "/tmp"}
    )


# ----------------------------------------------------------------------
# _extract_tool_name
# ----------------------------------------------------------------------


def test_extract_tool_name_prefers_function_plugin_format():
    ctx = FakeContext(function=FakeFunction(name="f", plugin_name="P"))
    assert _extract_tool_name(ctx) == "P.f"


def test_extract_tool_name_returns_none_when_function_missing():
    assert _extract_tool_name(FakeContext()) is None
    assert _extract_tool_name(None) is None


def test_extract_tool_name_bare_when_plugin_absent():
    ctx = FakeContext(function=FakeFunction(name="f"))
    assert _extract_tool_name(ctx) == "f"


# ----------------------------------------------------------------------
# _extract_arguments
# ----------------------------------------------------------------------


def test_extract_arguments_handles_dict():
    ctx = FakeContext(arguments={"a": 1})
    assert _extract_arguments(ctx) == {"a": 1}


def test_extract_arguments_returns_empty_when_absent():
    assert _extract_arguments(FakeContext()) == {}
    assert _extract_arguments(None) == {}


def test_extract_arguments_handles_non_iterable_gracefully():
    """Defensive — if SK ever hands us a value that can't be
    dict()'d, we return empty rather than crash."""
    ctx = FakeContext(arguments=42)
    assert _extract_arguments(ctx) == {}
