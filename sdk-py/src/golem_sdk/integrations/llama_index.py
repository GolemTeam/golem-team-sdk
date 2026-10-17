"""LlamaIndex integration for the Golem SDK.

LlamaIndex's primary observability surface is the
``BaseCallbackHandler`` ABC in ``llama_index.core.callbacks``.
A ``CallbackManager`` fans events out to all registered handlers
via ``on_event_start`` / ``on_event_end``, keyed by a
``CBEventType`` enum (FUNCTION_CALL, AGENT_STEP, QUERY, LLM, …).

We attach as a callback handler and emit a Golem
``intercept_execution`` record per ``FUNCTION_CALL`` event (the
tool-invocation boundary) and per ``QUERY`` event (the
top-level RAG retrieval boundary). LLM events are intentionally
skipped — they are captured at the HTTP layer by the witness's
Anthropic / OpenAI proxies, so emitting them here would
double-count.

Usage:

    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.llama_index import GolemCallbackHandler

    sdk = GolemSDK(witness_url="http://localhost:39393")
    handler = GolemCallbackHandler(sdk, agent_id="rag-bot")
    Settings.callback_manager = CallbackManager([handler])

This adapter **does not hard-import** ``llama_index``. The
LlamaIndex callback manager duck-types its handlers — it
iterates the registered list and calls ``on_event_start`` /
``on_event_end`` / ``start_trace`` / ``end_trace`` without an
``isinstance`` check. So this class conforms to the interface
by method signature alone, and importing it works on machines
without ``llama_index`` installed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from golem_sdk import GolemSDK

logger = logging.getLogger("golem_sdk.integrations.llama_index")

# Event types we care about. We compare against the string values
# of ``CBEventType`` rather than importing the enum, so we don't
# pin a llama-index version.
_EVENT_FUNCTION_CALL = "function_call"
_EVENT_QUERY = "query"


class GolemCallbackHandler:
    """Implements the ``BaseCallbackHandler`` interface for Golem.

    Duck-typed: matches the documented signatures of
    ``BaseCallbackHandler.on_event_start`` / ``on_event_end`` /
    ``start_trace`` / ``end_trace`` without inheriting from it,
    so this module imports cleanly without ``llama_index``
    installed.
    """

    # BaseCallbackHandler exposes these two lists so individual
    # handlers can opt out of specific event types. Some versions
    # of LlamaIndex's CallbackManager check that these attrs exist
    # before dispatching, so we set safe defaults.
    event_starts_to_ignore: list = []
    event_ends_to_ignore: list = []

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "llamaindex",
    ) -> None:
        """Args:
        sdk: A ``GolemSDK`` instance.
        agent_id: Identifier for the agent that owns this handler.
        mcp_server_id: Conceptual "server" identifier. Defaults to
            ``"llamaindex"``.
        """
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id

    # ------------------------------------------------------------------
    # BaseCallbackHandler surface
    # ------------------------------------------------------------------

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Record a Golem event for tool calls and top-level queries.

        Returns ``event_id`` unchanged. LlamaIndex's CallbackManager
        uses the return value for trace correlation; passing the id
        through is the documented contract.
        """
        try:
            etype = _event_type_value(event_type)
            if etype == _EVENT_FUNCTION_CALL:
                self._record_function_call(payload)
            elif etype == _EVENT_QUERY:
                self._record_query(payload)
            # All other event types (LLM, AGENT_STEP, RETRIEVE, …) are
            # silently ignored — they're either too low-level or
            # already captured by the witness HTTP proxies.
        except Exception as e:  # noqa: BLE001 — never break the chain
            logger.warning(
                "golem llama_index on_event_start failed: %s",
                type(e).__name__,
            )
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """No-op. We capture intent at event start, not end."""
        return None

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """No-op. The Golem session is keyed by ``agent_id``, not
        by LlamaIndex's per-query trace id."""
        return None

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _record_function_call(self, payload: Optional[Dict[str, Any]]) -> None:
        # Fail-open: a framework upgrade that changes the payload
        # shape MUST NOT crash the customer's agent. Any exception
        # here is logged and swallowed; the agent's LLM call
        # continues unobserved rather than dying.
        try:
            tool_name = _extract_tool_name(payload)
            if not tool_name:
                return
            args = _extract_tool_args(payload)
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name=tool_name,
                tool_arguments=args,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "golem llamaindex._record_function_call failed: %s",
                type(e).__name__,
            )

    def _record_query(self, payload: Optional[Dict[str, Any]]) -> None:
        # Same fail-open contract as _record_function_call.
        try:
            query_str = ""
            if isinstance(payload, dict):
                for key in ("query_str", "query"):
                    v = payload.get(key)
                    if isinstance(v, str) and v:
                        query_str = v
                        break
            self._sdk.intercept_execution(
                agent_id=self._agent_id,
                mcp_server_id=self._mcp_server_id,
                tool_name="llamaindex.query",
                tool_arguments={"query": query_str[:1024]},
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "golem llamaindex._record_query failed: %s",
                type(e).__name__,
            )


# ----------------------------------------------------------------------
# Duck-typed extractors
# ----------------------------------------------------------------------


def _event_type_value(event_type: Any) -> str:
    """Normalize a ``CBEventType`` enum value to its string form.

    LlamaIndex's enum values are lowercase strings like
    ``"function_call"`` and ``"query"``. We don't import the
    enum to avoid pinning a llama-index version, so callers may
    pass:
      * a ``CBEventType`` instance (has ``.value``)
      * a plain string (used by some custom integrations)
    """
    if event_type is None:
        return ""
    val = getattr(event_type, "value", None)
    if isinstance(val, str):
        return val.lower()
    if isinstance(event_type, str):
        return event_type.lower()
    return ""


def _extract_tool_name(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pick the tool name out of a FUNCTION_CALL payload.

    LlamaIndex shapes seen across versions:
      * ``payload["tool"]`` — a ``ToolMetadata`` model with ``.name``
      * ``payload["tool"]`` — a dict with ``"name"`` key
      * ``payload["function_call"]`` — a JSON string containing
        ``{"name": "...", "arguments": "..."}``
      * ``payload["function_call"]`` — a dict with ``"name"``
    """
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    if tool is not None:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            return name
        if isinstance(tool, dict):
            name = tool.get("name")
            if isinstance(name, str) and name:
                return name
    fc = payload.get("function_call")
    if isinstance(fc, dict):
        name = fc.get("name")
        if isinstance(name, str) and name:
            return name
    if isinstance(fc, str):
        try:
            decoded = json.loads(fc)
            if isinstance(decoded, dict):
                name = decoded.get("name")
                if isinstance(name, str) and name:
                    return name
        except (ValueError, TypeError):
            return None
    return None


def _extract_tool_args(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the tool arguments out of a FUNCTION_CALL payload.

    Shapes seen:
      * ``payload["function_call"]`` JSON string with
        ``{"name": "...", "arguments": {...}}`` or
        ``{"name": "...", "arguments": "<json>"}``
      * ``payload["function_call"]`` dict with ``"arguments"``
      * ``payload["arguments"]`` — direct dict
    """
    if not isinstance(payload, dict):
        return {}
    args = payload.get("arguments")
    if isinstance(args, dict):
        return args
    fc = payload.get("function_call")
    if isinstance(fc, dict):
        a = fc.get("arguments")
        if isinstance(a, dict):
            return a
        if isinstance(a, str):
            return _try_json_dict(a)
    if isinstance(fc, str):
        try:
            decoded = json.loads(fc)
        except (ValueError, TypeError):
            return {}
        if isinstance(decoded, dict):
            a = decoded.get("arguments")
            if isinstance(a, dict):
                return a
            if isinstance(a, str):
                return _try_json_dict(a)
    return {}


def _try_json_dict(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        decoded = json.loads(s)
    except (ValueError, TypeError):
        return {"input": s}
    if isinstance(decoded, dict):
        return decoded
    return {"input": decoded}
