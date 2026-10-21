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
    from agents.tracing.span_data import FunctionSpanData

    return FunctionSpanData(name=name, input=raw_input, output=None)


def _make_span(span_data):
    span = MagicMock()
    span.span_data = span_data
    return span


def test_on_span_end_emits_for_function_call(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    span = _make_span(
        _make_function_span_data(
            "search_tickets", '{"query": "auth", "max_results": 10}'
        )
    )
    processor.on_span_end(span)
    mock_sdk.intercept_execution.assert_called_once()
    call = mock_sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "search_tickets"
    assert call.kwargs["agent_id"] == "agent-A"
    assert call.kwargs["tool_arguments"] == {"query": "auth", "max_results": 10}


def test_on_span_start_records_nothing(mock_sdk):
    """At span start the Agents SDK has not set the arguments yet."""
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A", capture_generations=True)
    processor.on_span_start(_make_span(_make_function_span_data("search", None)))
    mock_sdk.intercept_execution.assert_not_called()


def test_on_span_end_ignores_non_function_spans(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    span = _make_span(MagicMock(name="GuardrailSpanData"))
    processor.on_span_end(span)
    mock_sdk.intercept_execution.assert_not_called()


def test_capture_generations_off_by_default(mock_sdk):
    """LLM generation spans default off because the witness HTTP
    proxy usually catches them already — double capture inflates."""
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    fake_gen = MagicMock()
    fake_gen.__class__.__name__ = "GenerationSpanData"
    fake_gen.model = "gpt-4o"
    processor.on_span_end(_make_span(fake_gen))
    mock_sdk.intercept_execution.assert_not_called()


def test_capture_generations_reads_the_model_of_a_generation_span(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A", capture_generations=True)
    from agents.tracing.span_data import GenerationSpanData

    processor.on_span_end(_make_span(GenerationSpanData(model="gpt-4o-mini")))
    assert mock_sdk.intercept_execution.call_args.kwargs["tool_name"] == "llm.gpt-4o-mini"


def test_capture_generations_reads_the_model_of_a_response_span(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A", capture_generations=True)
    from agents.tracing.span_data import ResponseSpanData

    response = MagicMock()
    response.model = "gpt-5"
    processor.on_span_end(_make_span(ResponseSpanData(response=response)))
    assert mock_sdk.intercept_execution.call_args.kwargs["tool_name"] == "llm.gpt-5"


def test_sdk_exception_does_not_break_tracer(mock_sdk):
    """Telemetry failures must NEVER propagate into the agents
    runtime — Runner.run would surface that to the user."""
    mock_sdk.intercept_execution.side_effect = ValueError("network down")
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A", capture_generations=True)
    processor.on_span_end(_make_span(_make_function_span_data("tool", "")))
    fake_gen = MagicMock()
    fake_gen.__class__.__name__ = "ResponseSpanData"
    processor.on_span_end(_make_span(fake_gen))


def test_on_span_end_survives_garbage_spans(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A", capture_generations=True)
    processor.on_span_end(_make_span(object()))
    processor.on_span_end(_make_span(None))
    processor.on_span_end(object())  # type: ignore[arg-type]


def test_on_trace_lifecycle_no_ops_dont_crash(mock_sdk):
    processor = GolemTraceProcessor(mock_sdk, agent_id="agent-A")
    fake_trace = MagicMock()
    processor.on_trace_start(fake_trace)
    processor.on_trace_end(fake_trace)
    processor.shutdown()
    processor.force_flush()
    mock_sdk.intercept_execution.assert_not_called()


def test_parse_tool_input_json_string():
    assert _parse_tool_input('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_tool_input_empty_string():
    assert _parse_tool_input("") == {}


def test_parse_tool_input_none():
    assert _parse_tool_input(None) == {}


def test_parse_tool_input_already_dict():
    assert _parse_tool_input({"foo": "bar"}) == {"foo": "bar"}


def test_parse_tool_input_non_dict_json():
    assert _parse_tool_input("[1, 2, 3]") == {"input": [1, 2, 3]}
    assert _parse_tool_input('"hello"') == {"input": "hello"}


def test_parse_tool_input_malformed_string():
    assert _parse_tool_input("not json {") == {"input": "not json {"}


def test_is_generation_span_recognises_both_names():
    a = MagicMock()
    a.__class__.__name__ = "GenerationSpanData"
    b = MagicMock()
    b.__class__.__name__ = "ResponseSpanData"
    c = MagicMock()
    c.__class__.__name__ = "FunctionSpanData"
    assert _is_generation_span(a) is True
    assert _is_generation_span(b) is True
    assert _is_generation_span(c) is False
