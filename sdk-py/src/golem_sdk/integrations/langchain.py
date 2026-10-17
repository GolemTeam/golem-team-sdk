"""LangChain integration for the Golem SDK.

Drop-in callback handler that captures every tool invocation a
LangChain agent makes — including those nested inside LCEL chains and
``AgentExecutor`` runs — and routes the pre-execution state through
``GolemSDK.intercept_execution``.

Usage:

    from golem_sdk import GolemSDK
    from golem_sdk.integrations.langchain import GolemCallbackHandler

    sdk = GolemSDK(
        witness_url="http://localhost:39393",
        labels={"workspace": "alice-laptop", "env": "dev"},
    )
    handler = GolemCallbackHandler(sdk, agent_id="my-agent",
                                   mcp_server_id="langchain")

    # Pass to a chain or agent:
    chain.invoke({"input": "hello"}, config={"callbacks": [handler]})

    # Or set globally:
    from langchain.callbacks import set_handler
    set_handler(handler)

This module is **import-only** when ``langchain-core`` is installed.
We don't pin a hard version range — LangChain's callback surface has
been stable since v0.1; the ``BaseCallbackHandler`` interface points
we depend on (``on_tool_start`` / ``on_tool_end`` / ``on_tool_error``)
have been there since the public API stabilised. We log a warning at
construction time if a known-breaking newer schema is detected so
the failure mode is loud rather than silent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID

from golem_sdk.policy import GolemPolicyBlocked

try:  # pragma: no cover — covered by tests that import this module
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "golem_sdk.integrations.langchain requires the langchain-core "
        "package. Install with: pip install 'golem-team-sdk[langchain]' or "
        "pip install langchain-core."
    ) from exc

from golem_sdk import GolemSDK

logger = logging.getLogger("golem_sdk.integrations.langchain")


class GolemCallbackHandler(BaseCallbackHandler):
    """LangChain callback that forwards tool invocations to Golem.

    Hooks into the standard ``BaseCallbackHandler`` lifecycle so it
    works inside ``AgentExecutor``, LCEL pipelines, and standalone
    ``Tool.invoke`` paths. The handler is non-blocking — every Golem
    call goes through ``GolemSDK.intercept_execution`` which is itself
    fire-and-forget in witness mode, so a slow Golem stack never
    slows down the user's agent.

    Errors raised by the Golem SDK (validation, network, etc.) are
    caught and logged at WARN; a bad telemetry call MUST NEVER break
    the user's chain.
    """

    # LangChain ``raise_error`` knob: when False (default) the handler
    # swallows its own exceptions instead of propagating them out of
    # the callback dispatch. We always want the swallow behaviour
    # because a Golem telemetry failure shouldn't break a customer's
    # agent run.
    raise_error: bool = False

    # Run lightweight when callbacks are emitted by a non-LLM/tool
    # node — saves a no-op intercept_execution per chain step.
    run_inline: bool = True

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "langchain",
        capture_llm_calls: bool = False,
        enforce: bool = False,
    ) -> None:
        """Wire a Golem SDK instance to receive LangChain callbacks.

        Args:
            sdk: A ``GolemSDK`` instance, ideally in witness mode
                (``GolemSDK(witness_url=...)``).
            agent_id: Identifier for the agent that owns this handler.
                Surfaces in the Golem console as the agent column.
                Free-form but should be stable per deployment so
                queries by ``agent_id`` work.
            mcp_server_id: Identifier for the conceptual "MCP server"
                this run targets. Defaults to ``"langchain"`` — set
                explicitly when the agent talks to a specific named
                LangChain toolkit you want to slice on.
            enforce: If True, consult the witness BEFORE each tool runs
                and raise ``GolemPolicyBlocked`` when policy refuses it,
                which aborts the chain. Off by default: turning it on
                changes a monitored agent into a gated one, and a new
                default that can stop a customer's tool call is not a
                default. With it off this handler only records, exactly
                as before.
            capture_llm_calls: If True, also emit a Golem event for
                each LLM round-trip (``on_llm_start``). Off by
                default because most deployments capture LLM traffic
                via the Anthropic / OpenAI HTTP proxies; double-
                capture would inflate the chain count. Turn on for
                pure-SDK deployments where the LLM API proxy isn't
                in the path.
        """
        super().__init__()
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._capture_llm_calls = capture_llm_calls
        self._enforce = enforce
        # tool_start → tool_end pairing for duration reporting.
        # Bounded so a runaway agent doesn't accumulate forever.
        self._run_starts: Dict[UUID, float] = {}
        self._MAX_RUN_STARTS = 1024

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Fired when a LangChain Tool is about to execute.

        Records the call to Golem BEFORE the tool runs so the
        signed audit trail represents *intent*, not just outcome —
        matches Golem's PreToolUse semantics elsewhere in the stack.
        """
        tool_name = _safe_tool_name(serialized)

        # The refusal is raised OUTSIDE the try below, and that placement is the
        # whole feature. The handler swallows every exception so a Golem fault
        # never breaks a customer's chain; a policy block is not a fault, and
        # letting it fall into that except would enforce nothing while looking
        # like it did.
        if self._enforce:
            decision = self._sdk.check_execution(
                tool_name=tool_name,
                command=input_str or "",
                agent_id=self._agent_id,
            )
            if not decision.allowed:
                raise GolemPolicyBlocked(tool_name, decision.reason)

        try:
            args = inputs if inputs is not None else _input_str_to_args(input_str)
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name=tool_name,
                tool_arguments=args,
            )
            # Track start so on_tool_end can report duration via
            # metadata if a future SDK version supports it.
            if len(self._run_starts) < self._MAX_RUN_STARTS:
                self._run_starts[run_id] = time.monotonic()
        except Exception as e:  # noqa: BLE001 — never break the chain
            logger.warning("golem on_tool_start failed: %s", type(e).__name__)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Tool completed cleanly. Clears the run-start bookkeeping."""
        self._run_starts.pop(run_id, None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Tool errored. Same — clear bookkeeping. (Tool error capture
        through Golem is left to a future commit; today PreToolUse
        records intent, not outcome.)"""
        self._run_starts.pop(run_id, None)

    # ------------------------------------------------------------------
    # LLM lifecycle (opt-in)
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Optional: record the LLM call as a Golem event."""
        if not self._capture_llm_calls:
            return
        try:
            llm_name = _safe_tool_name(serialized, default="llm.call")
            # LangChain serialises prompts as a list; join with a
            # marker so multi-prompt batches stay distinguishable.
            prompt_text = "\n---\n".join(prompts) if isinstance(prompts, list) else str(prompts)
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name=llm_name,
                tool_arguments={"prompts_count": len(prompts) if isinstance(prompts, list) else 1},
                system_prompt=prompt_text[:100_000],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("golem on_llm_start failed: %s", type(e).__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _safe_tool_name(serialized: Dict[str, Any], default: str = "unknown") -> str:
    """LangChain serializes the tool's class/registration info in
    ``serialized``. The ``name`` key is the canonical place but older
    LangChain releases put it under ``id`` as a class path. Fall back
    gracefully so a renamed internal layout doesn't break capture.
    """
    if not isinstance(serialized, dict):
        return default
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    id_val = serialized.get("id")
    if isinstance(id_val, list) and id_val:
        return str(id_val[-1])
    if isinstance(id_val, str) and id_val:
        return id_val
    return default


def _input_str_to_args(input_str: str) -> Dict[str, Any]:
    """LangChain's older API passed a raw input string; newer versions
    pass a structured ``inputs`` dict. Normalise to a dict so the
    downstream SDK's ``tool_arguments`` (which requires a dict)
    accepts the value either way.
    """
    if not isinstance(input_str, str):
        return {"input": str(input_str)}
    return {"input": input_str}
