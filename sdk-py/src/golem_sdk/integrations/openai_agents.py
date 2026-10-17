"""OpenAI Agents SDK integration for the Golem SDK.

The OpenAI Agents SDK (``openai-agents``) exposes a ``TracingProcessor``
interface that fires per-span lifecycle events for every agent run,
tool call, and handoff. This adapter implements that interface and
forwards function-tool calls through ``GolemSDK.intercept_execution``.

Why a tracing processor and not a callback:
    Unlike LangChain, the OpenAI Agents SDK doesn't have a separate
    callbacks system. Tracing IS the observability surface. The
    ``TracingProcessor`` ABC is the public hook point, and is
    documented as stable.

Usage:

    from agents import Runner, set_trace_processors
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.openai_agents import GolemTraceProcessor

    sdk = GolemSDK(witness_url="http://localhost:39393")
    set_trace_processors([GolemTraceProcessor(sdk, agent_id="my-agent")])

    result = Runner.run_sync(my_agent, "hello")

This module is import-only when ``openai-agents`` is installed.
Install via ``pip install 'golem-team-sdk[openai-agents]'``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:  # pragma: no cover — covered by tests that import this module
    from agents.tracing import TracingProcessor, Span, Trace
    from agents.tracing.span_data import FunctionSpanData
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "golem_sdk.integrations.openai_agents requires the openai-agents "
        "package. Install with: pip install 'golem-team-sdk[openai-agents]' or "
        "pip install openai-agents."
    ) from exc

from golem_sdk import GolemSDK

logger = logging.getLogger("golem_sdk.integrations.openai_agents")


class GolemTraceProcessor(TracingProcessor):
    """OpenAI Agents tracing processor that forwards tool calls to Golem.

    Implements the ``TracingProcessor`` ABC. Only ``FunctionSpanData``
    spans (i.e. tool calls) produce Golem events — other span types
    (handoffs, guardrail checks, generation spans) are observed but
    not relayed today. Generation spans are usually captured at the
    HTTP layer by the witness's OpenAI proxy, so double-capture would
    inflate the chain.
    """

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "openai-agents",
        capture_generations: bool = False,
    ) -> None:
        """Wire a Golem SDK instance to the OpenAI Agents tracer.

        Args:
            sdk: A ``GolemSDK`` instance (ideally witness-mode).
            agent_id: Identifier for the agent that owns this
                processor. Surfaces in the Golem console.
            mcp_server_id: Conceptual "server" identifier. Defaults
                to ``"openai-agents"``; override when the run uses a
                specific named toolset you want to slice on.
            capture_generations: If True, also emit Golem events for
                LLM generation spans. Off by default — generation
                traffic is usually captured at the HTTP layer.
        """
        super().__init__()
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._capture_generations = capture_generations

    # ------------------------------------------------------------------
    # TracingProcessor ABC
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Trace) -> None:
        """No-op: we don't emit a Golem event for the trace itself.
        Each tool span will produce its own event with the trace's
        agent_id as a label.
        """

    def on_trace_end(self, trace: Trace) -> None:
        """No-op."""

    def on_span_start(self, span: Span[Any]) -> None:
        """Fired when a span (tool call, generation, handoff) begins.
        We capture FunctionSpanData here so the Golem event lands
        BEFORE the tool executes — matching the pre-execution semantics
        used elsewhere in the stack.
        """
        try:
            data = span.span_data
            if isinstance(data, FunctionSpanData):
                self._emit_function_call(data)
            # Generation spans are captured elsewhere (HTTP proxy)
            # unless opt-in.
            elif self._capture_generations and _is_generation_span(data):
                self._emit_generation(data)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "golem on_span_start failed: %s", type(e).__name__
            )

    def on_span_end(self, span: Span[Any]) -> None:
        """No-op: outcome capture lives in a future commit.
        PreToolUse-style records the intent; PostToolUse-style outcome
        capture (success/failure/duration) deferred for now.
        """

    def shutdown(self) -> None:
        """Called when the tracer is shutting down. Nothing to flush
        — GolemSDK calls are fire-and-forget in witness mode."""

    def force_flush(self) -> None:
        """Same — nothing to flush."""

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------

    def _emit_function_call(self, data: FunctionSpanData) -> None:
        """Forward a function-tool call as a Golem intercept event.

        Fail-open: a future version of openai-agents may rename
        `.name` or `.input` on FunctionSpanData. The customer's
        agent MUST keep running even if our telemetry shape is
        wrong; any exception is logged and swallowed.
        """
        try:
            tool_name = getattr(data, "name", None) or "unknown"
            raw_input = getattr(data, "input", None)
            args = _parse_tool_input(raw_input)
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name=tool_name,
                tool_arguments=args,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "golem openai_agents._emit_function_call failed: %s",
                type(e).__name__,
            )

    def _emit_generation(self, data: Any) -> None:
        """Optional: emit a Golem event for an LLM generation span.

        Same fail-open contract as _emit_function_call.
        """
        try:
            model = getattr(data, "model", None) or "unknown"
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name=f"llm.{model}",
                tool_arguments={"model": model},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "golem openai_agents._emit_generation failed: %s",
                type(e).__name__,
            )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _parse_tool_input(raw: Any) -> dict:
    """The Agents SDK passes the tool input as a JSON string in
    ``FunctionSpanData.input``. Try to decode it for a structured
    record; fall back to wrapping the raw string so the SDK's strict
    ``tool_arguments: dict`` contract is satisfied regardless.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        # Empty string is a common no-arg tool call.
        if not raw:
            return {}
        try:
            import json

            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                return decoded
            return {"input": decoded}
        except (ValueError, TypeError):
            return {"input": raw}
    return {"input": str(raw)}


def _is_generation_span(data: Any) -> bool:
    """Best-effort detection of an LLM generation span across openai-
    agents minor versions — the class name has shifted between
    ``GenerationSpanData`` and ``ResponseSpanData`` historically.
    """
    cls_name = type(data).__name__
    return cls_name in {"GenerationSpanData", "ResponseSpanData"}
