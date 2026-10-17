"""Tests for the LangChain callback handler adapter."""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

# Skip the entire module if langchain-core isn't installed.
pytest.importorskip("langchain_core")

from golem_sdk import GolemSDK  # noqa: E402
from golem_sdk.integrations.langchain import (  # noqa: E402
    GolemCallbackHandler,
    _safe_tool_name,
    _input_str_to_args,
)


@pytest.fixture
def mock_sdk():
    """A SDK with intercept_execution mocked so tests don't need network."""
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution = MagicMock()
    return sdk


def test_on_tool_start_calls_intercept_execution(mock_sdk):
    handler = GolemCallbackHandler(
        mock_sdk, agent_id="agent-A", mcp_server_id="langchain-test"
    )
    handler.on_tool_start(
        serialized={"name": "search_tickets"},
        input_str='{"query": "auth failures"}',
        run_id=uuid4(),
        inputs={"query": "auth failures", "max_results": 10},
    )
    mock_sdk.intercept_execution.assert_called_once()
    call = mock_sdk.intercept_execution.call_args
    assert call.kwargs["agent_id"] == "agent-A"
    assert call.kwargs["mcp_server_id"] == "langchain-test"
    assert call.kwargs["tool_name"] == "search_tickets"
    assert call.kwargs["tool_arguments"] == {"query": "auth failures", "max_results": 10}


def test_on_tool_start_falls_back_to_input_str_when_inputs_missing(mock_sdk):
    handler = GolemCallbackHandler(mock_sdk, agent_id="agent-A")
    handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="2 + 2",
        run_id=uuid4(),
        # No inputs= kwarg
    )
    call = mock_sdk.intercept_execution.call_args
    assert call.kwargs["tool_arguments"] == {"input": "2 + 2"}


def test_sdk_exception_does_not_break_callback(mock_sdk):
    """Telemetry failures must never propagate up — they would kill
    the customer's chain run."""
    mock_sdk.intercept_execution.side_effect = ValueError("network down")
    handler = GolemCallbackHandler(mock_sdk, agent_id="agent-A")
    # Must not raise.
    handler.on_tool_start(
        serialized={"name": "tool"},
        input_str="",
        run_id=uuid4(),
    )


def test_on_tool_end_clears_run_tracking(mock_sdk):
    handler = GolemCallbackHandler(mock_sdk, agent_id="agent-A")
    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "tool"}, input_str="", run_id=run_id
    )
    assert run_id in handler._run_starts
    handler.on_tool_end(output="result", run_id=run_id)
    assert run_id not in handler._run_starts


def test_on_tool_error_clears_run_tracking(mock_sdk):
    handler = GolemCallbackHandler(mock_sdk, agent_id="agent-A")
    run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "tool"}, input_str="", run_id=run_id
    )
    handler.on_tool_error(error=RuntimeError("boom"), run_id=run_id)
    assert run_id not in handler._run_starts


def test_capture_llm_calls_off_by_default(mock_sdk):
    handler = GolemCallbackHandler(mock_sdk, agent_id="agent-A")
    handler.on_llm_start(
        serialized={"name": "ChatAnthropic"},
        prompts=["Hello"],
        run_id=uuid4(),
    )
    mock_sdk.intercept_execution.assert_not_called()


def test_capture_llm_calls_on_when_enabled(mock_sdk):
    handler = GolemCallbackHandler(
        mock_sdk, agent_id="agent-A", capture_llm_calls=True
    )
    handler.on_llm_start(
        serialized={"name": "ChatAnthropic"},
        prompts=["What is 2 + 2?"],
        run_id=uuid4(),
    )
    mock_sdk.intercept_execution.assert_called_once()
    call = mock_sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "ChatAnthropic"
    assert call.kwargs["system_prompt"] == "What is 2 + 2?"


def test_run_starts_bounded(mock_sdk):
    """A runaway agent shouldn't accumulate run-start entries forever."""
    handler = GolemCallbackHandler(mock_sdk, agent_id="agent-A")
    handler._MAX_RUN_STARTS = 5  # tighten for the test
    for _ in range(20):
        handler.on_tool_start(
            serialized={"name": "tool"}, input_str="", run_id=uuid4()
        )
    assert len(handler._run_starts) <= 5


def test_safe_tool_name_prefers_name_key():
    assert _safe_tool_name({"name": "search"}) == "search"


def test_safe_tool_name_falls_back_to_id_list():
    # Older LangChain releases serialised tool identity as a class path list.
    assert _safe_tool_name({"id": ["langchain", "tools", "Calculator"]}) == "Calculator"


def test_safe_tool_name_falls_back_to_id_string():
    assert _safe_tool_name({"id": "OldStyleName"}) == "OldStyleName"


def test_safe_tool_name_default_when_unrecognised():
    assert _safe_tool_name({}) == "unknown"
    assert _safe_tool_name({"name": ""}) == "unknown"
    assert _safe_tool_name(None) == "unknown"  # type: ignore[arg-type]


def test_input_str_to_args_handles_non_string():
    # Defensive: shouldn't crash if LangChain ever passes a non-str.
    assert _input_str_to_args(42) == {"input": "42"}  # type: ignore[arg-type]
