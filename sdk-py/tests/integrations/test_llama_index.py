"""Tests for the LlamaIndex callback handler adapter.

The adapter has no hard dependency on ``llama_index`` — it
duck-types the BaseCallbackHandler interface — so these tests
run without any ``importorskip``. We construct fake payloads
with the shapes LlamaIndex emits.
"""

from unittest.mock import MagicMock

from golem_sdk import GolemSDK
from golem_sdk.integrations.llama_index import (
    GolemCallbackHandler,
    _event_type_value,
    _extract_tool_name,
    _extract_tool_args,
)


def _mock_sdk():
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution = MagicMock()
    return sdk


class FakeEnum:
    """Stand-in for CBEventType enum members."""

    def __init__(self, value: str):
        self.value = value


class FakeToolMetadata:
    """Stand-in for LlamaIndex's ToolMetadata Pydantic model."""

    def __init__(self, name: str):
        self.name = name


# ----------------------------------------------------------------------
# on_event_start — FUNCTION_CALL routing
# ----------------------------------------------------------------------


def test_function_call_with_tool_metadata_fires_intercept():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    payload = {
        "tool": FakeToolMetadata(name="web_search"),
        "function_call": '{"name": "web_search", "arguments": {"q": "golem"}}',
    }
    h.on_event_start(FakeEnum("function_call"), payload=payload, event_id="e1")
    sdk.intercept_execution.assert_called_once()
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "web_search"
    assert call.kwargs["tool_arguments"] == {"q": "golem"}


def test_function_call_with_dict_tool():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    payload = {"tool": {"name": "calculator"}, "function_call": ""}
    h.on_event_start(FakeEnum("function_call"), payload=payload, event_id="e1")
    assert sdk.intercept_execution.call_args.kwargs["tool_name"] == "calculator"


def test_function_call_with_only_function_call_string():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    payload = {
        "function_call": '{"name": "fs.read", "arguments": "{\\"path\\":\\"/etc/hosts\\"}"}'
    }
    h.on_event_start(FakeEnum("function_call"), payload=payload, event_id="e1")
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "fs.read"
    assert call.kwargs["tool_arguments"] == {"path": "/etc/hosts"}


def test_function_call_missing_name_skips():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(FakeEnum("function_call"), payload={}, event_id="e1")
    sdk.intercept_execution.assert_not_called()


# ----------------------------------------------------------------------
# on_event_start — QUERY routing
# ----------------------------------------------------------------------


def test_query_event_records_llamaindex_query():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(
        FakeEnum("query"),
        payload={"query_str": "what is golem?"},
        event_id="q1",
    )
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "llamaindex.query"
    assert call.kwargs["tool_arguments"] == {"query": "what is golem?"}


def test_query_event_truncates_long_query_strings():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(
        FakeEnum("query"),
        payload={"query_str": "x" * 5000},
        event_id="q1",
    )
    q = sdk.intercept_execution.call_args.kwargs["tool_arguments"]["query"]
    assert len(q) == 1024


def test_query_event_handles_missing_string():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(FakeEnum("query"), payload={}, event_id="q1")
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_arguments"] == {"query": ""}


# ----------------------------------------------------------------------
# Other event types are silently ignored
# ----------------------------------------------------------------------


def test_llm_event_is_ignored():
    """LLM events are double-counted with the witness HTTP proxy."""
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(FakeEnum("llm"), payload={"messages": []}, event_id="l1")
    sdk.intercept_execution.assert_not_called()


def test_agent_step_event_is_ignored():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(
        FakeEnum("agent_step"), payload={"messages": []}, event_id="a1"
    )
    sdk.intercept_execution.assert_not_called()


def test_retrieve_event_is_ignored():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_start(
        FakeEnum("retrieve"), payload={"query_str": "x"}, event_id="r1"
    )
    sdk.intercept_execution.assert_not_called()


# ----------------------------------------------------------------------
# Return value contract + lifecycle methods
# ----------------------------------------------------------------------


def test_on_event_start_returns_event_id():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    out = h.on_event_start(FakeEnum("function_call"), payload={}, event_id="abc")
    assert out == "abc"


def test_on_event_end_is_noop():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.on_event_end(FakeEnum("function_call"), payload={}, event_id="e1")
    sdk.intercept_execution.assert_not_called()


def test_start_and_end_trace_are_noops():
    sdk = _mock_sdk()
    h = GolemCallbackHandler(sdk, agent_id="rag")
    h.start_trace("t1")
    h.end_trace("t1", trace_map={})
    sdk.intercept_execution.assert_not_called()


def test_ignore_lists_are_present():
    """LlamaIndex's CallbackManager checks for these attrs."""
    h = GolemCallbackHandler(_mock_sdk(), agent_id="rag")
    assert isinstance(h.event_starts_to_ignore, list)
    assert isinstance(h.event_ends_to_ignore, list)


# ----------------------------------------------------------------------
# Failure isolation
# ----------------------------------------------------------------------


def test_sdk_exception_does_not_propagate():
    """Telemetry failures must never propagate to LlamaIndex's loop."""
    sdk = _mock_sdk()
    sdk.intercept_execution.side_effect = ValueError("boom")
    h = GolemCallbackHandler(sdk, agent_id="rag")
    # Must not raise.
    out = h.on_event_start(
        FakeEnum("function_call"),
        payload={"tool": FakeToolMetadata(name="x")},
        event_id="e1",
    )
    assert out == "e1"


# ----------------------------------------------------------------------
# _event_type_value
# ----------------------------------------------------------------------


def test_event_type_value_extracts_enum_value():
    assert _event_type_value(FakeEnum("function_call")) == "function_call"


def test_event_type_value_normalizes_string():
    assert _event_type_value("FUNCTION_CALL") == "function_call"


def test_event_type_value_returns_empty_when_absent():
    assert _event_type_value(None) == ""
    assert _event_type_value(object()) == ""


# ----------------------------------------------------------------------
# _extract_tool_name
# ----------------------------------------------------------------------


def test_extract_tool_name_prefers_tool_metadata():
    p = {"tool": FakeToolMetadata(name="t1")}
    assert _extract_tool_name(p) == "t1"


def test_extract_tool_name_falls_back_to_function_call_string():
    p = {"function_call": '{"name": "t2"}'}
    assert _extract_tool_name(p) == "t2"


def test_extract_tool_name_returns_none_when_absent():
    assert _extract_tool_name({}) is None
    assert _extract_tool_name(None) is None


def test_extract_tool_name_handles_malformed_json():
    assert _extract_tool_name({"function_call": "not json"}) is None


# ----------------------------------------------------------------------
# _extract_tool_args
# ----------------------------------------------------------------------


def test_extract_tool_args_handles_direct_arguments_key():
    assert _extract_tool_args({"arguments": {"a": 1}}) == {"a": 1}


def test_extract_tool_args_decodes_nested_json_string():
    p = {"function_call": '{"name": "t", "arguments": "{\\"a\\": 1}"}'}
    assert _extract_tool_args(p) == {"a": 1}


def test_extract_tool_args_handles_nested_dict_arguments():
    p = {"function_call": {"name": "t", "arguments": {"a": 1}}}
    assert _extract_tool_args(p) == {"a": 1}


def test_extract_tool_args_empty_when_absent():
    assert _extract_tool_args({}) == {}
    assert _extract_tool_args(None) == {}


def test_extract_tool_args_wraps_non_dict_decoded_value():
    """A scalar argument gets wrapped under "input"."""
    p = {"function_call": '{"name": "t", "arguments": "42"}'}
    assert _extract_tool_args(p) == {"input": 42}


# ----------------------------------------------------------------------
# Fail-open contract on the internal dispatchers
#
# A LlamaIndex upgrade that changes payload shapes or breaks the
# downstream sdk.intercept_execution call MUST NOT crash the
# customer's agent. The internal dispatchers swallow exceptions
# and log a warning instead.
# ----------------------------------------------------------------------


def _handler_with_broken_sdk():
    """Build a GolemCallbackHandler whose intercept_execution raises.
    The handler's dispatchers must survive."""
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution.side_effect = RuntimeError("framework drift")
    return GolemCallbackHandler(sdk, agent_id="test-agent")


def test_record_function_call_swallows_intercept_exceptions():
    h = _handler_with_broken_sdk()
    # Payload resolves a tool name so we reach intercept_execution.
    h._record_function_call({"function_call": {"name": "search", "arguments": {}}})
    # No exception raised → fail-open contract honored.


def test_record_query_swallows_intercept_exceptions():
    h = _handler_with_broken_sdk()
    h._record_query({"query_str": "what is the meaning of life?"})


def test_record_function_call_survives_garbage_payload():
    # A future framework version might pass a fundamentally
    # different shape (e.g. an object, a list, None). Same fail-open.
    h = _handler_with_broken_sdk()
    for bad in (None, [], "string-not-dict", 42, object()):
        h._record_function_call(bad)  # type: ignore[arg-type]


def test_record_query_survives_garbage_payload():
    h = _handler_with_broken_sdk()
    for bad in (None, [], "string-not-dict", 42, object()):
        h._record_query(bad)  # type: ignore[arg-type]
