"""CrewAI integration for the Golem SDK.

``GolemToolHooks`` registers CrewAI's global before/after tool-call hooks
(``crewai.hooks``). The before hook sees the tool name and its arguments before
the tool runs, on both of CrewAI's tool-calling paths (native function calling
and the ReAct text protocol). It records the call and, with ``enforce=True``,
blocks a call your policy refuses: the tool does not run and the model reads
``"Blocked by Golem policy: <reason>"`` as the tool's result. The after hook
records what the tool returned and, when the Golem agent withholds it, hands
the model the agent's notice instead. The hooks also record each model call's
token usage from CrewAI's ``LLMCallCompletedEvent``.

Usage:

    from crewai import Crew
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.crewai import GolemToolHooks, GolemTaskListener

    sdk = GolemSDK(witness_url="http://localhost:39393")
    hooks = GolemToolHooks(sdk, agent_id="research-crew", enforce=True).register()

    crew = Crew(agents=[...], tasks=[...],
                task_callback=GolemTaskListener(sdk, agent_id="research-crew"))
    crew.kickoff()

CrewAI's hooks are process-wide: once registered they see every crew's tool
calls until ``hooks.unregister()``. ``with GolemToolHooks(...):`` scopes them
to a block.

This module imports without ``crewai``; ``register`` needs a crewai that has
``crewai.hooks`` (tested on 1.15).
"""

from __future__ import annotations

import logging
import threading
import warnings
from typing import Any, Dict, Optional, Tuple, Union

from golem_sdk import GolemSDK
from golem_sdk.integrations._calls import ToolCall
from golem_sdk.policy import blocked_message

logger = logging.getLogger("golem_sdk.integrations.crewai")


class GolemToolHooks:
    """CrewAI before/after tool-call hooks that record each call and its result,
    and can refuse the call or withhold the result."""

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "crewai",
        enforce: bool = False,
        record_usage: bool = True,
    ) -> None:
        """Args:
        sdk: A ``GolemSDK`` instance.
        agent_id: Identifier for the crew these hooks record.
        mcp_server_id: The server the calls are judged and recorded under.
            Defaults to ``"crewai"``.
        enforce: If True, ask the Golem agent before each tool runs and block
            the call when policy refuses it. Off by default.
        record_usage: Record each model call's token counts as CrewAI reports
            them. Turn off when the model traffic already goes through the
            Golem agent's proxies, which count it there.
        """
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._enforce = enforce
        self._record_usage = record_usage
        # The before and after hooks get different context objects, so a call
        # is paired by the thread that runs it, which runs both hooks.
        self._open: Dict[Tuple[int, str], Union[ToolCall, str]] = {}
        self._lock = threading.Lock()
        self._before_hook = self._before
        self._after_hook = self._after
        self._usage_handler = self._on_llm_completed

    def register(self) -> "GolemToolHooks":
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.llm_events import LLMCallCompletedEvent
        from crewai.hooks import (
            register_after_tool_call_hook,
            register_before_tool_call_hook,
        )

        register_before_tool_call_hook(self._before_hook)
        register_after_tool_call_hook(self._after_hook)
        if self._record_usage:
            crewai_event_bus.register_handler(LLMCallCompletedEvent, self._usage_handler)
        return self

    def unregister(self) -> None:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.llm_events import LLMCallCompletedEvent
        from crewai.hooks import (
            unregister_after_tool_call_hook,
            unregister_before_tool_call_hook,
        )

        unregister_before_tool_call_hook(self._before_hook)
        unregister_after_tool_call_hook(self._after_hook)
        if self._record_usage:
            crewai_event_bus.off(LLMCallCompletedEvent, self._usage_handler)

    def __enter__(self) -> "GolemToolHooks":
        return self.register()

    def __exit__(self, *exc: Any) -> None:
        self.unregister()

    def _before(self, context: Any) -> Optional[bool]:
        name = getattr(context, "tool_name", None) or "unknown"
        raw = getattr(context, "tool_input", None)
        args = dict(raw) if isinstance(raw, dict) else ({} if raw is None else {"input": raw})
        call = ToolCall(self._sdk, self._agent_id, self._mcp_server_id, name, args)
        key = (threading.get_ident(), name)
        if self._enforce:
            decision = call.check()
            if not decision.allowed:
                with self._lock:
                    self._open[key] = decision.reason
                return False
        call.record()
        with self._lock:
            self._open[key] = call
        return None

    def _after(self, context: Any) -> Optional[str]:
        name = getattr(context, "tool_name", None) or "unknown"
        with self._lock:
            opened = self._open.pop((threading.get_ident(), name), None)
        # CrewAI's own text for a blocked call names no reason.
        if isinstance(opened, str):
            return blocked_message(opened)
        if opened is None:
            return None
        return opened.finish(getattr(context, "tool_result", None))

    def _on_llm_completed(self, source: Any, event: Any) -> None:
        usage = getattr(event, "usage", None)
        model = getattr(event, "model", None)
        if not isinstance(usage, dict) or not model:
            return
        try:
            self._sdk.record_usage(
                str(model),
                input_tokens=usage.get("prompt_tokens") or 0,
                output_tokens=usage.get("completion_tokens") or 0,
                cache_read_tokens=usage.get("cached_prompt_tokens") or 0,
                cache_creation_tokens=usage.get("cache_creation_tokens") or 0,
                agent_id=self._agent_id,
            )
        except Exception as e:  # noqa: BLE001 — never break the crew
            logger.warning("golem crewai usage handler failed: %s", type(e).__name__)


class GolemStepListener:
    """Deprecated: use ``GolemToolHooks``.

    CrewAI does not call ``step_callback`` for native function calling, its
    default for a model that supports it, so this records nothing there; on
    the ReAct path it is called after the tool has already run.
    """

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "crewai",
    ) -> None:
        warnings.warn(
            "GolemStepListener misses CrewAI's native tool calls and records "
            "after the tool ran; use GolemToolHooks.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id

    def __call__(self, step: Any) -> None:
        try:
            tool_name = _extract_tool_name(step)
            if not tool_name:
                return
            args = _extract_tool_args(step)
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name=tool_name,
                tool_arguments=args,
            )
        except Exception as e:  # noqa: BLE001 — never break the crew
            logger.warning(
                "golem crewai step listener failed: %s", type(e).__name__
            )


class GolemTaskListener:
    """Callable for ``Crew(task_callback=...)`` / ``Task(callback=...)``.

    Records one ``crewai.task`` event per completed task, with its description
    and the length of its output.
    """

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "crewai",
    ) -> None:
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id

    def __call__(self, task_output: Any) -> None:
        try:
            description = _get_attr(task_output, "description") or _get_attr(
                task_output, "name"
            ) or "task"
            raw = _get_attr(task_output, "raw_output") or _get_attr(
                task_output, "raw"
            ) or ""
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name="crewai.task",
                tool_arguments={
                    "description": str(description)[:1024],
                    "output_length": len(raw) if isinstance(raw, str) else 0,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "golem crewai task listener failed: %s", type(e).__name__
            )


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or dict key: CrewAI hands over Pydantic models and plain dicts."""
    if obj is None:
        return default
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _extract_tool_name(step: Any) -> Optional[str]:
    for path in ("tool", "tool_name"):
        v = _get_attr(step, path)
        if isinstance(v, str) and v:
            return v
    action = _get_attr(step, "action")
    if action is not None:
        v = _get_attr(action, "tool")
        if isinstance(v, str) and v:
            return v
    return None


def _extract_tool_args(step: Any) -> Dict[str, Any]:
    candidates = (
        _get_attr(step, "tool_input"),
        _get_attr(_get_attr(step, "action"), "tool_input"),
    )
    for raw in candidates:
        if raw is None:
            continue
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
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
    return {}
