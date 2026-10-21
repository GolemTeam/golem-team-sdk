"""OpenAI Agents SDK integration for the Golem SDK.

Three entry points:

* ``golem_guard_tools`` adds a Golem tool input guardrail and output guardrail
  to each function tool. The input guardrail runs before the tool: it records
  the call with its arguments and, with ``enforce=True``, rejects a call your
  policy blocks. A rejected call does not run; the model receives
  ``"Blocked by Golem policy: <reason>"`` as the tool's output and the run
  carries on. The output guardrail records what the tool returned and, when
  the Golem agent withholds it, hands the model the agent's notice instead.
* ``GolemRunHooks`` records each model call's token usage. Pass it as
  ``Runner.run(..., hooks=GolemRunHooks(sdk, agent_id=...))``.
* ``GolemTraceProcessor`` records from the tracing pipeline. A function span's
  arguments are only set once the tool has run, so it records when the span
  ends, after the tool ran, and it cannot refuse or withhold anything. Its
  arguments are only there when the run traces sensitive data
  (``RunConfig.trace_include_sensitive_data``, on by default).

Use the guardrails or the trace processor for the same tools, not both:
together they record each call twice.

Usage:

    from agents import Agent, Runner
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.openai_agents import GolemRunHooks, golem_guard_tools

    sdk = GolemSDK(witness_url="http://localhost:39393")
    agent = Agent(
        name="support",
        tools=golem_guard_tools([lookup_order, refund], sdk,
                                agent_id="support", enforce=True),
    )
    result = Runner.run_sync(agent, "hello", hooks=GolemRunHooks(sdk, agent_id="support"))

Install via ``pip install 'golem-team-sdk[openai-agents]'``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, NamedTuple

try:  # pragma: no cover — covered by tests that import this module
    from agents import FunctionTool, RunHooks
    from agents.tool_guardrails import (
        ToolGuardrailFunctionOutput,
        ToolInputGuardrail,
        ToolInputGuardrailData,
        ToolOutputGuardrail,
        ToolOutputGuardrailData,
    )
    from agents.tracing import TracingProcessor, Span, Trace
    from agents.tracing.span_data import FunctionSpanData
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "golem_sdk.integrations.openai_agents requires the openai-agents "
        "package. Install with: pip install 'golem-team-sdk[openai-agents]' or "
        "pip install openai-agents."
    ) from exc

from golem_sdk import GolemSDK
from golem_sdk.integrations._calls import ToolCall
from golem_sdk.policy import blocked_message

logger = logging.getLogger("golem_sdk.integrations.openai_agents")

_INPUT_GUARDRAIL = "golem_policy"
_OUTPUT_GUARDRAIL = "golem_result"
_MAX_OPEN_CALLS = 1024


class GolemGuardrails(NamedTuple):
    input: ToolInputGuardrail
    output: ToolOutputGuardrail


def golem_tool_guardrails(
    sdk: GolemSDK,
    agent_id: str,
    mcp_server_id: str = "openai-agents",
    enforce: bool = False,
) -> GolemGuardrails:
    """The input and output guardrails ``golem_guard_tools`` attaches.

    Attach them yourself where ``golem_guard_tools`` cannot reach, such as an
    MCP server's ``tool_input_guardrails`` and ``tool_output_guardrails``. The
    two share state, so use them as a pair.
    """
    open_calls: Dict[str, ToolCall] = {}
    lock = threading.Lock()

    async def golem_policy(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        call_id = getattr(ctx, "tool_call_id", None)
        call = ToolCall(sdk, agent_id, mcp_server_id, getattr(ctx, "tool_name", None) or "unknown",
                        _parse_tool_input(getattr(ctx, "tool_arguments", None)), call_id=call_id)
        if enforce:
            decision = await asyncio.to_thread(call.check)
            if not decision.allowed:
                return ToolGuardrailFunctionOutput.reject_content(
                    blocked_message(decision.reason),
                    output_info={"golem_rule": decision.rule_name},
                )
        await asyncio.to_thread(call.record)
        with lock:
            if call_id and len(open_calls) < _MAX_OPEN_CALLS:
                open_calls[call_id] = call
        return ToolGuardrailFunctionOutput.allow()

    async def golem_result(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        with lock:
            call = open_calls.pop(getattr(data.context, "tool_call_id", None) or "", None)
        if call is None:
            return ToolGuardrailFunctionOutput.allow()
        notice = await asyncio.to_thread(call.finish, data.output)
        if notice is None:
            return ToolGuardrailFunctionOutput.allow()
        return ToolGuardrailFunctionOutput.reject_content(notice)

    return GolemGuardrails(
        input=ToolInputGuardrail(guardrail_function=golem_policy, name=_INPUT_GUARDRAIL),
        output=ToolOutputGuardrail(guardrail_function=golem_result, name=_OUTPUT_GUARDRAIL),
    )


def golem_guard_tools(
    tools: List[Any],
    sdk: GolemSDK,
    agent_id: str,
    mcp_server_id: str = "openai-agents",
    enforce: bool = False,
) -> List[Any]:
    """Add the Golem guardrails to every function tool in ``tools``, in place.

    Returns ``tools``. Hosted tools (web search, file search, …) run at OpenAI
    and take no guardrails, so they are left as they are.
    """
    guardrails = golem_tool_guardrails(sdk, agent_id, mcp_server_id, enforce)
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            continue
        inputs = list(tool.tool_input_guardrails or [])
        if any(g.get_name() == _INPUT_GUARDRAIL for g in inputs):
            continue
        tool.tool_input_guardrails = [*inputs, guardrails.input]
        tool.tool_output_guardrails = [*(tool.tool_output_guardrails or []), guardrails.output]
    return tools


class GolemRunHooks(RunHooks):
    """Run hooks that record each model call's token usage, as the model reported it."""

    def __init__(self, sdk: GolemSDK, agent_id: str) -> None:
        super().__init__()
        self._sdk = sdk
        self._agent_id = agent_id

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        try:
            usage = getattr(response, "usage", None)
            model = _model_name(getattr(agent, "model", None))
            if usage is None or not model:
                return
            details = getattr(usage, "input_tokens_details", None)
            self._sdk.record_usage(
                model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(details, "cached_tokens", 0) or 0,
                agent_id=self._agent_id,
            )
        except Exception as e:  # noqa: BLE001 — never break the run
            logger.warning("golem on_llm_end failed: %s", type(e).__name__)


class GolemTraceProcessor(TracingProcessor):
    """Records function-tool calls and their results (and, opt-in, model
    generations) from the tracer.

    Records when a span ends, because that is when the Agents SDK has filled in
    its arguments, output and model: after the tool ran. To record before a
    tool runs, to refuse it, or to withhold its result, use ``golem_guard_tools``.
    """

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "openai-agents",
        capture_generations: bool = False,
    ) -> None:
        """Args:
        sdk: A ``GolemSDK`` instance (ideally witness-mode).
        agent_id: Identifier for the agent that owns this processor.
        mcp_server_id: The server the calls are recorded under. Defaults to
            ``"openai-agents"``.
        capture_generations: If True, also record each model response as
            ``llm.<model>``. Off by default — generation traffic is usually
            captured at the HTTP layer.
        """
        super().__init__()
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._capture_generations = capture_generations

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        try:
            data = span.span_data
            if isinstance(data, FunctionSpanData):
                call = ToolCall(self._sdk, self._agent_id, self._mcp_server_id,
                                getattr(data, "name", None) or "unknown",
                                _parse_tool_input(getattr(data, "input", None)))
                call.record()
                error = getattr(span, "error", None)
                # The tool already ran and nothing here can withhold its
                # output, so its result is not worth holding up the run for.
                threading.Thread(target=call.finish, args=(getattr(data, "output", None),
                                                           bool(error)), daemon=True).start()
            elif self._capture_generations and _is_generation_span(data):
                model = _generation_model(data)
                self._sdk.intercept_execution(
                    agent_id=self._agent_id,
                    mcp_server_id=self._mcp_server_id,
                    tool_name=f"llm.{model}",
                    tool_arguments={"model": model},
                )
        except Exception as e:  # noqa: BLE001 — never break the run
            logger.warning("golem on_span_end failed: %s", type(e).__name__)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


def _model_name(model: Any) -> str:
    if isinstance(model, str):
        return model
    name = getattr(model, "model", None)
    return name if isinstance(name, str) else ""


def _parse_tool_input(raw: Any) -> dict:
    """Tool arguments arrive as a JSON string; anything else is kept as ``input``."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            return {"input": raw}
        return decoded if isinstance(decoded, dict) else {"input": decoded}
    return {"input": str(raw)}


def _is_generation_span(data: Any) -> bool:
    return type(data).__name__ in {"GenerationSpanData", "ResponseSpanData"}


def _generation_model(data: Any) -> str:
    model = getattr(data, "model", None) or getattr(getattr(data, "response", None), "model", None)
    return str(model) if model else "unknown"
