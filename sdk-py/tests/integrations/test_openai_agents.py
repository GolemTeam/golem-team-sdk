"""Tests for the OpenAI Agents SDK tracing processor adapter."""

import pytest
from unittest.mock import MagicMock

pytest.importorskip("agents.tracing")

from golem_sdk import GolemSDK  # noqa: E402
from golem_sdk.integrations.openai_agents import (  # noqa: E402
    GolemTraceProcessor,
    _parse_tool_input,
    _is_generation_span,
)


@pytest.fixture
def mock_sdk():
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution = MagicMock()
    return sdk


def _make_function_span_data(name: str, raw_input: str):
    """Construct a FunctionSpanData with the public constructor — keeps
    the test resilient to internal field renames."""
    from agents.tracing.span_data import FunctionSpanData

    return FunctionSpanData(name=name, input=raw_input, output=None)


def _make_span(span_data):
    """The processor only reads ``.span_data``; everything else on
    Span is irrelevant for our tests. A MagicMock is the cheapest
    way to satisfy the type."""
    span = MagicMock()
    span.span_data = span_data
    return span


def test_on_span_start_emits_for_function_call(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    span = _make_span(
        _make_function_span_data(
            "search_tickets", '{"query": "auth", "max_results": 10}'
        )
    )
    processor.on_span_start(span)
    mock_sdk.intercept_execution.assert_called_once()
    call = mock_sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "search_tickets"
    assert call.kwargs["agent_id"] == "agent-A"
    assert call.kwargs["tool_arguments"] == {"query": "auth", "max_results": 10}


def test_on_span_start_ignores_non_function_spans(mock_sdk):
    """The processor should be quiet for trace types it doesn't handle."""
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    # A span with non-Function span_data (use a plain object).
    span = _make_span(MagicMock(name="GuardrailSpanData"))
    processor.on_span_start(span)
    mock_sdk.intercept_execution.assert_not_called()


def test_capture_generations_off_by_default(mock_sdk):
    """LLM generation spans default off because the witness HTTP
    proxy usually catches them already — double capture inflates."""
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    fake_gen = MagicMock()
    fake_gen.__class__.__name__ = "GenerationSpanData"
    fake_gen.model = "gpt-4o"
    span = _make_span(fake_gen)
    processor.on_span_start(span)
    mock_sdk.intercept_execution.assert_not_called()


def test_capture_generations_on_when_enabled(mock_sdk):
    """Pure-SDK deployments can opt in to LLM capture."""
    processor = GolemTraceProcessor(
        mock_sdk, agent_id="agent-A", capture_generations=True
    )
    fake_gen = MagicMock()
    fake_gen.__class__.__name__ = "ResponseSpanData"
    fake_gen.model = "gpt-4o-mini"
    span = _make_span(fake_gen)
    processor.on_span_start(span)
    # Capture only fires for fake span types when _is_generation_span
    # matches the class name. We supply a MagicMock whose __class__.__name__
    # is one of the recognised names.
    mock_sdk.intercept_execution.assert_called_once()
    assert mock_sdk.intercept_execution.call_args.kwargs["tool_name"] == "llm.gpt-4o-mini"


def test_sdk_exception_does_not_break_tracer(mock_sdk):
    """Telemetry failures must NEVER propagate into the agents
    runtime — Runner.run would surface that to the user."""
    mock_sdk.intercept_execution.side_effect = ValueError("network down")
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    span = _make_span(_make_function_span_data("tool", ""))
    # Must not raise.
    processor.on_span_start(span)


def test_on_trace_lifecycle_no_ops_dont_crash(mock_sdk):
    """Trace-level callbacks intentionally do nothing. Verify they
    don't accidentally raise when called with the real trace shape."""
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    fake_trace = MagicMock()
    processor.on_trace_start(fake_trace)
    processor.on_trace_end(fake_trace)
    processor.shutdown()
    processor.force_flush()
    mock_sdk.intercept_execution.assert_not_called()


# ----------------------------------------------------------------------
# _parse_tool_input
# ----------------------------------------------------------------------


def test_parse_tool_input_json_string():
    assert _parse_tool_input('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_tool_input_empty_string():
    assert _parse_tool_input("") == {}


def test_parse_tool_input_none():
    assert _parse_tool_input(None) == {}


def test_parse_tool_input_already_dict():
    assert _parse_tool_input({"foo": "bar"}) == {"foo": "bar"}


def test_parse_tool_input_non_dict_json():
    # If the tool input is a JSON list or scalar, wrap it under "input".
    assert _parse_tool_input("[1, 2, 3]") == {"input": [1, 2, 3]}
    assert _parse_tool_input('"hello"') == {"input": "hello"}


def test_parse_tool_input_malformed_string():
    """A non-JSON string falls back to wrapping under "input" — the
    SDK's intercept_execution requires a dict for tool_arguments."""
    assert _parse_tool_input("not json {") == {"input": "not json {"}


# ----------------------------------------------------------------------
# _is_generation_span
# ----------------------------------------------------------------------


def test_is_generation_span_recognises_both_names():
    """openai-agents has used GenerationSpanData and ResponseSpanData
    in different releases; we accept both for forward/backward compat."""
    a = MagicMock()
    a.__class__.__name__ = "GenerationSpanData"
    b = MagicMock()
    b.__class__.__name__ = "ResponseSpanData"
    c = MagicMock()
    c.__class__.__name__ = "FunctionSpanData"
    assert _is_generation_span(a) is True
    assert _is_generation_span(b) is True
    assert _is_generation_span(c) is False


# ----------------------------------------------------------------------
# Fail-open contract on the internal emit helpers
#
# An openai-agents upgrade that renames .name / .input / .model on
# its SpanData types MUST NOT crash the customer's agent. The
# internal emitters swallow exceptions and log a warning instead.
# ----------------------------------------------------------------------


def _processor_with_broken_sdk():
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution.side_effect = RuntimeError("framework drift")
    return GolemTraceProcessor(sdk, agent_id="test-agent")


def test_emit_function_call_swallows_intercept_exceptions():
    proc = _processor_with_broken_sdk()
    data = _make_function_span_data("search", '{"q": "x"}')
    proc._emit_function_call(data)
    # No exception raised → fail-open contract honored.


def test_emit_generation_swallows_intercept_exceptions():
    proc = _processor_with_broken_sdk()
    fake_data = MagicMock()
    fake_data.model = "gpt-5"
    proc._emit_generation(fake_data)


def test_emit_function_call_survives_garbage_data():
    # Future versions may pass an object that doesn't have .name /
    # .input at all. Same fail-open.
    proc = _processor_with_broken_sdk()
    proc._emit_function_call(object())  # type: ignore[arg-type]
    proc._emit_function_call(None)  # type: ignore[arg-type]


def test_emit_generation_survives_garbage_data():
    proc = _processor_with_broken_sdk()
    proc._emit_generation(object())
    proc._emit_generation(None)
