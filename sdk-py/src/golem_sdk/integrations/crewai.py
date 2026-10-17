"""CrewAI integration for the Golem SDK.

CrewAI's observability surface is a pair of callable hooks on
``Crew`` (and per-``Agent``): ``step_callback`` fires after each
agent step, ``task_callback`` fires when a task completes. Both
accept a plain callable; there is no ABC to subclass, which is
why this module has **no hard dependency on ``crewai``**. We
produce callables that match the documented signatures.

Usage:

    from crewai import Crew, Agent
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.crewai import (
        GolemStepListener, GolemTaskListener,
    )

    sdk = GolemSDK(witness_url="http://localhost:39393")
    listener = GolemStepListener(sdk, agent_id="research-crew")

    crew = Crew(
        agents=[...],
        tasks=[...],
        step_callback=listener,           # per-step callback
        task_callback=GolemTaskListener(sdk, agent_id="research-crew"),
    )

Because we duck-type the callback payload, this adapter is stable
across crewai versions that have shuffled the internal Pydantic
model names (``AgentAction``, ``ToolUsage``, ``StepOutput`` etc.).
We read attributes if they exist, fall back gracefully, and never
crash on schema drift.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from golem_sdk import GolemSDK

logger = logging.getLogger("golem_sdk.integrations.crewai")


class GolemStepListener:
    """Callable for ``Crew(step_callback=...)``.

    Fires once per agent step. Records the step's tool invocation
    (when present) to Golem as a pre-execution intent. CrewAI uses
    different shapes across versions for the step payload — we
    duck-type and pick up whatever's there.
    """

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "crewai",
    ) -> None:
        """Args:
        sdk: A ``GolemSDK`` instance.
        agent_id: Identifier for the crew that owns this listener.
        mcp_server_id: Conceptual "server" identifier. Defaults to
            ``"crewai"``; override when a single crew talks to a
            specific named toolset you want to slice on.
        """
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id

    def __call__(self, step: Any) -> None:
        """Step-callback entry point. Called by CrewAI per step.

        CrewAI passes either an ``AgentAction`` (tool used) or an
        ``AgentFinish`` (no tool, final answer). We capture the
        action case and silently ignore the finish case — final
        answers don't represent tool intent.
        """
        try:
            tool_name = _extract_tool_name(step)
            if not tool_name:
                # No tool invocation in this step — silent skip.
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

    Fires once per task completion. CrewAI passes a ``TaskOutput``
    (or similar) with a ``raw_output`` / ``raw`` text field. We
    treat the task itself as a Golem event so a multi-task crew
    shows up as N records in the timeline.
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
            # Description goes in tool_arguments to keep tool_name short;
            # the raw output text is what an investigator wants to read
            # but it lives in BYOB-bound storage rather than the chain.
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


# ----------------------------------------------------------------------
# Duck-typed extractors
# ----------------------------------------------------------------------


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe attribute or dict-key getter. CrewAI sometimes hands us a
    Pydantic model (attribute access) and sometimes a plain dict
    (key access) — try both. Avoids a heap of ``hasattr`` /
    ``isinstance`` branches at each call site.
    """
    if obj is None:
        return default
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _extract_tool_name(step: Any) -> Optional[str]:
    """Pick the tool name out of a step payload.

    Across crewai versions we've seen:
      * ``step.tool`` — a string, modern API
      * ``step.action.tool`` — older AgentAction wrapper
      * ``step.tool_name`` — alt naming
    """
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
    """Pick the tool input out of a step payload.

    CrewAI shapes seen:
      * ``step.tool_input`` — dict
      * ``step.tool_input`` — JSON string
      * ``step.action.tool_input`` — older wrapper
    """
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
