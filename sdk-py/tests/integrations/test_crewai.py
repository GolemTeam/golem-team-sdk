"""Tests for the CrewAI step / task listener adapter.

The adapter has no hard dependency on the ``crewai`` package — it
duck-types the callback payload — so these tests run without any
``importorskip``. We construct fake payloads with the shapes CrewAI
hands its callbacks.
"""

from unittest.mock import MagicMock

import pytest

from golem_sdk import GolemSDK
from golem_sdk.integrations.crewai import (
    GolemStepListener,
    GolemTaskListener,
    _extract_tool_name,
    _extract_tool_args,
)

pytestmark = pytest.mark.filterwarnings("ignore:GolemStepListener:DeprecationWarning")


# ----------------------------------------------------------------------
# GolemStepListener
# ----------------------------------------------------------------------


def _mock_sdk():
    sdk = MagicMock(spec=GolemSDK)
    sdk.intercept_execution = MagicMock()
    return sdk


@pytest.mark.filterwarnings("default")
def test_step_listener_is_deprecated():
    with pytest.warns(DeprecationWarning, match="GolemToolHooks"):
        GolemStepListener(_mock_sdk(), agent_id="crew-A")


class FakeStep:
    """Stand-in for crewai's AgentAction / AgentFinish payload.

    Real CrewAI emits Pydantic models; we just need attribute access.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_step_with_tool_modern_shape_fires_intercept():
    sdk = _mock_sdk()
    listener = GolemStepListener(sdk, agent_id="crew-A")
    step = FakeStep(tool="web_search", tool_input={"query": "golem"})
    listener(step)
    sdk.intercept_execution.assert_called_once()
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "web_search"
    assert call.kwargs["tool_arguments"] == {"query": "golem"}


def test_step_with_action_wrapper_older_shape():
    sdk = _mock_sdk()
    listener = GolemStepListener(sdk, agent_id="crew-A")
    step = FakeStep(
        action=FakeStep(tool="calculator", tool_input='{"a": 1, "b": 2}')
    )
    listener(step)
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "calculator"
    assert call.kwargs["tool_arguments"] == {"a": 1, "b": 2}


def test_step_with_no_tool_does_not_fire():
    """AgentFinish (final answer) has no tool — silently skip."""
    sdk = _mock_sdk()
    listener = GolemStepListener(sdk, agent_id="crew-A")
    step = FakeStep(return_values={"output": "the answer is 42"})
    listener(step)
    sdk.intercept_execution.assert_not_called()


def test_step_listener_swallows_sdk_exceptions():
    """Telemetry failures must never propagate to CrewAI's loop."""
    sdk = _mock_sdk()
    sdk.intercept_execution.side_effect = ValueError("boom")
    listener = GolemStepListener(sdk, agent_id="crew-A")
    step = FakeStep(tool="x", tool_input={})
    # Must not raise.
    listener(step)


def test_step_with_dict_payload():
    """CrewAI has shipped releases that pass dicts instead of Pydantic
    objects to callbacks. Verify dict-style access works."""
    sdk = _mock_sdk()
    listener = GolemStepListener(sdk, agent_id="crew-A")
    listener({"tool": "fs.read", "tool_input": {"path": "/etc/hosts"}})
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "fs.read"
    assert call.kwargs["tool_arguments"] == {"path": "/etc/hosts"}


# ----------------------------------------------------------------------
# GolemTaskListener
# ----------------------------------------------------------------------


class FakeTaskOutput:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_task_listener_records_task_completion():
    sdk = _mock_sdk()
    listener = GolemTaskListener(sdk, agent_id="crew-A")
    output = FakeTaskOutput(
        description="Research Golem use cases",
        raw_output="Long answer goes here..." * 50,
    )
    listener(output)
    sdk.intercept_execution.assert_called_once()
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_name"] == "crewai.task"
    assert "Research Golem use cases" in call.kwargs["tool_arguments"]["description"]
    assert call.kwargs["tool_arguments"]["output_length"] > 0


def test_task_listener_alt_field_names():
    """Newer CrewAI versions renamed `raw_output` → `raw` and use
    `name` instead of `description` in some paths. Verify fallback."""
    sdk = _mock_sdk()
    listener = GolemTaskListener(sdk, agent_id="crew-A")
    output = FakeTaskOutput(name="task-1", raw="x" * 10)
    listener(output)
    call = sdk.intercept_execution.call_args
    assert call.kwargs["tool_arguments"]["description"] == "task-1"
    assert call.kwargs["tool_arguments"]["output_length"] == 10


def test_task_listener_swallows_sdk_exceptions():
    sdk = _mock_sdk()
    sdk.intercept_execution.side_effect = RuntimeError("network down")
    listener = GolemTaskListener(sdk, agent_id="crew-A")
    listener(FakeTaskOutput(description="x", raw_output=""))


def test_task_listener_truncates_long_descriptions():
    """Description is bounded at 1024 chars so a giant task title
    doesn't bloat every event in the chain."""
    sdk = _mock_sdk()
    listener = GolemTaskListener(sdk, agent_id="crew-A")
    listener(FakeTaskOutput(description="x" * 5000, raw_output=""))
    desc = sdk.intercept_execution.call_args.kwargs["tool_arguments"]["description"]
    assert len(desc) == 1024


# ----------------------------------------------------------------------
# _extract_tool_name / _extract_tool_args
# ----------------------------------------------------------------------


def test_extract_tool_name_prefers_top_level():
    assert _extract_tool_name(FakeStep(tool="t1")) == "t1"
    assert _extract_tool_name(FakeStep(tool_name="t2")) == "t2"


def test_extract_tool_name_falls_back_to_action_wrapper():
    assert _extract_tool_name(FakeStep(action=FakeStep(tool="inner"))) == "inner"


def test_extract_tool_name_returns_none_when_absent():
    assert _extract_tool_name(FakeStep()) is None
    assert _extract_tool_name(None) is None
    assert _extract_tool_name({}) is None


def test_extract_tool_args_handles_dict_directly():
    assert _extract_tool_args(FakeStep(tool_input={"a": 1})) == {"a": 1}


def test_extract_tool_args_decodes_json_string():
    assert _extract_tool_args(FakeStep(tool_input='{"a": 1}')) == {"a": 1}


def test_extract_tool_args_wraps_non_dict_json():
    """A JSON array or scalar input gets wrapped under "input" so the
    downstream SDK contract (dict tool_arguments) is satisfied."""
    assert _extract_tool_args(FakeStep(tool_input='[1, 2]')) == {"input": [1, 2]}


def test_extract_tool_args_wraps_malformed_string():
    assert _extract_tool_args(FakeStep(tool_input="not json")) == {"input": "not json"}


def test_extract_tool_args_empty_when_absent():
    assert _extract_tool_args(FakeStep()) == {}
