"""LlamaIndex integration for the Golem SDK.

Two entry points:

* ``golem_guard_tools(tools, sdk, ...)`` returns the tools wrapped so that each
  call is recorded, with its arguments, before it runs, and its result after.
  With ``enforce=True`` a call your policy blocks raises ``GolemPolicyBlocked``
  instead of running; a LlamaIndex agent hands the exception's message to the
  model as the tool's output. A result Golem withholds is replaced by its
  notice. llama-index-core 0.14 no longer emits a ``FUNCTION_CALL`` callback
  event, so wrapping the tools is the only way to see tool calls.
* ``GolemCallbackHandler`` records each top-level ``QUERY`` and, on LlamaIndex
  releases that still emit them, ``FUNCTION_CALL`` events. Use it alongside
  the wrapped tools only on 0.14 or later, or calls are recorded twice.

Usage:

    from llama_index.core.agent.workflow import FunctionAgent
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.llama_index import golem_guard_tools

    sdk = GolemSDK(witness_url="http://localhost:39393")
    agent = FunctionAgent(
        tools=golem_guard_tools(tools, sdk, agent_id="rag-bot", enforce=True),
        llm=llm,
    )

This module imports without ``llama_index``; ``golem_guard_tools`` imports it.
"""

from __future__ import annotations

import asyncio
import copy
import functools
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from golem_sdk import GolemSDK
from golem_sdk.integrations._calls import ToolCall
from golem_sdk.policy import GolemPolicyBlocked

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

    LlamaIndex's own agents put the bare arguments JSON in
    ``payload["function_call"]``; a ``{"name", "arguments"}`` wrapper and a
    direct ``payload["arguments"]`` dict are accepted too.
    """
    if not isinstance(payload, dict):
        return {}
    args = payload.get("arguments")
    if isinstance(args, dict):
        return args
    fc = payload.get("function_call")
    if isinstance(fc, str):
        fc = _try_json_dict(fc)
    if not isinstance(fc, dict):
        return {}
    if "arguments" in fc and set(fc) <= {"name", "arguments"}:
        a = fc["arguments"]
        if isinstance(a, dict):
            return a
        return _try_json_dict(a) if isinstance(a, str) else {}
    return fc


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


def golem_guard_tools(
    tools: List[Any],
    sdk: GolemSDK,
    agent_id: str,
    mcp_server_id: str = "llamaindex",
    enforce: bool = False,
) -> List[Any]:
    """The tools, each wrapped to record its calls and results and, with
    ``enforce``, to refuse what policy blocks. The originals are left as they are."""
    return [golem_guard_tool(t, sdk, agent_id, mcp_server_id, enforce) for t in tools]


def golem_guard_tool(
    tool: Any,
    sdk: GolemSDK,
    agent_id: str,
    mcp_server_id: str = "llamaindex",
    enforce: bool = False,
) -> Any:
    from llama_index.core.tools import FunctionTool

    guard = _Guard(sdk, agent_id, mcp_server_id, enforce, tool.metadata.get_name())
    if isinstance(tool, FunctionTool):
        # A copy rather than a wrapper: agents inject a workflow Context only
        # into a FunctionTool, so the guarded tool must stay one.
        guarded = copy.copy(tool)
        ctx_param = tool.ctx_param_name if tool.requires_context else None
        guarded._fn = guard.wrap_sync(tool._fn, ctx_param)
        guarded._async_fn = guard.wrap_async(tool._async_fn, ctx_param)
        return guarded
    return _guarded_tool_class()(tool, guard)


class _Guard:
    def __init__(self, sdk: GolemSDK, agent_id: str, mcp_server_id: str, enforce: bool,
                 tool_name: str) -> None:
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._enforce = enforce
        self._tool_name = tool_name

    def start(self, args: tuple, kwargs: Dict[str, Any], ctx_param: Optional[str]) -> ToolCall:
        arguments = {k: v for k, v in kwargs.items() if k != ctx_param}
        if args:
            arguments = {"args": list(args), **arguments}
        call = ToolCall(self._sdk, self._agent_id, self._mcp_server_id, self._tool_name, arguments)
        if self._enforce:
            decision = call.check()
            if not decision.allowed:
                raise GolemPolicyBlocked(self._tool_name, decision.reason)
        call.record()
        return call

    def wrap_sync(self, fn: Callable[..., Any], ctx_param: Optional[str] = None) -> Callable[..., Any]:
        @functools.wraps(fn)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            call = self.start(args, kwargs, ctx_param)
            try:
                out = fn(*args, **kwargs)
            except Exception as e:
                call.finish(f"{type(e).__name__}: {e}", is_error=True)
                raise
            notice = call.finish(out)
            return out if notice is None else notice

        return guarded

    def wrap_async(self, fn: Callable[..., Any], ctx_param: Optional[str] = None) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def guarded(*args: Any, **kwargs: Any) -> Any:
            call = await asyncio.to_thread(self.start, args, kwargs, ctx_param)
            try:
                out = await fn(*args, **kwargs)
            except Exception as e:
                await asyncio.to_thread(call.finish, f"{type(e).__name__}: {e}", True)
                raise
            notice = await asyncio.to_thread(call.finish, out)
            return out if notice is None else notice

        return guarded


@functools.lru_cache(maxsize=None)
def _guarded_tool_class() -> type:
    from llama_index.core.tools import AsyncBaseTool, ToolOutput

    class GolemGuardedTool(AsyncBaseTool):
        def __init__(self, tool: Any, guard: _Guard) -> None:
            self._tool = tool
            self._guard = guard

        @property
        def metadata(self) -> Any:
            return self._tool.metadata

        def call(self, *args: Any, **kwargs: Any) -> Any:
            return self._output(self._guard.wrap_sync(self._tool.__call__)(*args, **kwargs))

        async def acall(self, *args: Any, **kwargs: Any) -> Any:
            if isinstance(self._tool, AsyncBaseTool):
                inner = self._tool.acall
            else:
                async def inner(*a: Any, **k: Any) -> Any:
                    return await asyncio.to_thread(self._tool, *a, **k)
            return self._output(await self._guard.wrap_async(inner)(*args, **kwargs))

        def _output(self, value: Any) -> Any:
            if isinstance(value, ToolOutput):
                return value
            return ToolOutput(content=str(value), tool_name=self.metadata.get_name(),
                              raw_input={}, raw_output=value)

    return GolemGuardedTool
