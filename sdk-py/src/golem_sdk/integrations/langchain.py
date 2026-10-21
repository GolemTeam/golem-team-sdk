"""LangChain integration for the Golem SDK.

Two entry points, for the two ways LangChain runs a tool:

* ``GolemToolMiddleware`` for an agent built with ``langchain.agents.create_agent``.
  It records each tool call before it runs and its result after, and each model
  call's token usage. With ``enforce=True`` it answers a refused call with a
  ``ToolMessage`` carrying the reason, so the model reads the refusal and the
  run carries on. A result the Golem agent withholds reaches the model as the
  agent's notice instead.
* ``GolemCallbackHandler`` for tools and chains run directly (``tool.invoke``,
  LCEL). It records each call before it runs, its result after, and each model
  call's token usage. With ``enforce=True`` it raises ``GolemPolicyBlocked``
  before the tool body runs. Inside an agent loop that exception ends the run
  unless the ``ToolNode`` was built with ``handle_tool_errors`` covering it,
  which is why agents should use the middleware. A callback cannot change a
  tool's output, so the handler records a withheld result but cannot keep it
  from the model.

Use one or the other for a given run: attached together they record each call
twice.

Usage:

    from golem_sdk import GolemSDK
    from golem_sdk.integrations.langchain import (
        GolemCallbackHandler, GolemToolMiddleware,
    )

    sdk = GolemSDK(witness_url="http://localhost:39393")

    agent = create_agent(model, tools, middleware=[
        GolemToolMiddleware(sdk, agent_id="my-agent", enforce=True),
    ])

    handler = GolemCallbackHandler(sdk, agent_id="my-agent", enforce=True)
    chain.invoke({"input": "hello"}, config={"callbacks": [handler]})

The handler needs ``langchain-core``; the middleware also needs ``langchain``
1.0 or later.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from golem_sdk.integrations._calls import ToolCall
from golem_sdk.policy import GolemPolicyBlocked, blocked_message

try:  # pragma: no cover — covered by tests that import this module
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, ToolMessage
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "golem_sdk.integrations.langchain requires the langchain-core "
        "package. Install with: pip install 'golem-team-sdk[langchain]' or "
        "pip install langchain-core."
    ) from exc

try:
    from langchain.agents.middleware import AgentMiddleware as _AgentMiddleware
except ImportError:
    _AgentMiddleware = object  # type: ignore[assignment,misc]

from golem_sdk import GolemSDK

logger = logging.getLogger("golem_sdk.integrations.langchain")

_MAX_OPEN_CALLS = 1024


class GolemCallbackHandler(BaseCallbackHandler):
    """LangChain callback that records tool calls, their results and model usage
    to Golem, and can refuse a call.

    Errors raised by the Golem SDK (validation, network, etc.) are caught and
    logged at WARN; a bad telemetry call never breaks the user's chain.
    """

    run_inline: bool = True

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "langchain",
        capture_llm_calls: bool = False,
        enforce: bool = False,
        record_usage: bool = True,
    ) -> None:
        """Wire a Golem SDK instance to receive LangChain callbacks.

        Args:
            sdk: A ``GolemSDK`` instance, ideally in witness mode
                (``GolemSDK(witness_url=...)``).
            agent_id: Identifier for the agent that owns this handler.
            mcp_server_id: The server the call is judged and recorded under.
                Defaults to ``"langchain"``.
            enforce: If True, consult the Golem agent BEFORE each tool runs and
                raise ``GolemPolicyBlocked`` when policy refuses it. Off by
                default: turning it on changes a monitored agent into a gated
                one.
            capture_llm_calls: If True, also emit a Golem event for each LLM
                round-trip (``on_llm_start``). Off by default because most
                deployments capture LLM traffic via the Golem agent's HTTP
                proxies.
            record_usage: Record each model call's token counts, as the model
                reported them. Turn off when the model traffic already goes
                through the Golem agent's proxies, which count it there.
        """
        super().__init__()
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._capture_llm_calls = capture_llm_calls
        self._enforce = enforce
        self._record_usage = record_usage
        # LangChain's callback manager logs and swallows a handler's exception
        # unless raise_error is set, and a swallowed refusal lets the tool run.
        self.raise_error = enforce
        self._calls: Dict[UUID, ToolCall] = {}

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
        call = ToolCall(
            self._sdk, self._agent_id, self._mcp_server_id, _safe_tool_name(serialized),
            inputs if isinstance(inputs, dict) else _input_str_to_args(input_str),
            call_id=kwargs.get("tool_call_id"),
        )

        # Raised outside any except: a Golem fault never breaks the chain, and
        # a policy refusal is not a fault.
        if self._enforce:
            decision = call.check()
            if not decision.allowed:
                raise GolemPolicyBlocked(call.tool_name, decision.reason)

        call.record()
        if len(self._calls) < _MAX_OPEN_CALLS:
            self._calls[run_id] = call

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        call = self._calls.pop(run_id, None)
        if call is not None:
            call.finish(_tool_output(output), is_error=getattr(output, "status", "") == "error")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        call = self._calls.pop(run_id, None)
        if call is not None:
            call.finish(f"{type(error).__name__}: {error}", is_error=True)

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
        if not self._capture_llm_calls:
            return
        try:
            llm_name = _safe_tool_name(serialized, default="llm.call")
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

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if not self._record_usage:
            return
        llm_output = getattr(response, "llm_output", None) or {}
        for generations in getattr(response, "generations", None) or []:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is not None:
                    _record_usage(self._sdk, self._agent_id, message, llm_output)


class GolemToolMiddleware(_AgentMiddleware):  # type: ignore[misc,valid-type]
    """``create_agent`` middleware that records each tool call before it runs,
    its result after, and each model call's token usage.

    With ``enforce=True`` a refused call is not run: the model receives a
    ``ToolMessage`` with status ``error`` and the content
    ``"Blocked by Golem policy: <reason>"``, and the agent loop continues. A
    result the Golem agent withholds reaches the model as its notice.
    """

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "langchain",
        enforce: bool = False,
        record_usage: bool = True,
    ) -> None:
        if _AgentMiddleware is object:
            raise ImportError(
                "GolemToolMiddleware requires langchain 1.0 or later "
                "(pip install langchain)."
            )
        super().__init__()
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._enforce = enforce
        self._record_usage = record_usage

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        call, refusal = self._start(request.tool_call)
        if refusal is not None:
            return refusal
        try:
            response = handler(request)
        except Exception as e:
            call.finish(f"{type(e).__name__}: {e}", is_error=True)
            raise
        return self._finish(call, response)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        call, refusal = await asyncio.to_thread(self._start, request.tool_call)
        if refusal is not None:
            return refusal
        try:
            response = await handler(request)
        except Exception as e:
            await asyncio.to_thread(call.finish, f"{type(e).__name__}: {e}", True)
            raise
        return await asyncio.to_thread(self._finish, call, response)

    def after_model(self, state: Any, runtime: Any) -> None:
        if not self._record_usage:
            return None
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        if messages and isinstance(messages[-1], AIMessage):
            _record_usage(self._sdk, self._agent_id, messages[-1], {})
        return None

    def _start(self, tool_call: Dict[str, Any]) -> tuple:
        args = tool_call.get("args")
        call = ToolCall(
            self._sdk, self._agent_id, self._mcp_server_id, tool_call.get("name") or "unknown",
            args if isinstance(args, dict) else {"input": args},
            call_id=tool_call.get("id"),
        )
        if self._enforce:
            decision = call.check()
            if not decision.allowed:
                return call, ToolMessage(
                    content=blocked_message(decision.reason),
                    name=call.tool_name,
                    tool_call_id=tool_call.get("id") or "",
                    status="error",
                )
        call.record()
        return call, None

    def _finish(self, call: ToolCall, response: Any) -> Any:
        if not isinstance(response, ToolMessage):
            return response
        notice = call.finish(_tool_output(response), is_error=response.status == "error")
        if notice is None:
            return response
        return ToolMessage(content=notice, name=response.name,
                           tool_call_id=response.tool_call_id, status=response.status)


def _tool_output(output: Any) -> Any:
    return output.content if isinstance(output, ToolMessage) else output


def _record_usage(sdk: GolemSDK, agent_id: str, message: Any, llm_output: Dict[str, Any]) -> None:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return
    meta = getattr(message, "response_metadata", None) or {}
    model = (meta.get("model_name") or meta.get("model")
             or llm_output.get("model_name") or llm_output.get("model"))
    if not model:
        return
    details = usage.get("input_token_details") or {}
    try:
        sdk.record_usage(
            str(model),
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            cache_read_tokens=details.get("cache_read") or 0,
            cache_creation_tokens=details.get("cache_creation") or 0,
            provider=str(meta.get("model_provider") or ""),
            agent_id=agent_id,
        )
    except Exception as e:  # noqa: BLE001 — never break the agent
        logger.warning("golem: recording model usage failed: %s", type(e).__name__)


def _safe_tool_name(serialized: Dict[str, Any], default: str = "unknown") -> str:
    """The tool's ``name``, else the last part of its ``id`` class path."""
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


def _input_str_to_args(input_str: Any) -> Dict[str, Any]:
    """Arguments from the raw input string, when LangChain passes no ``inputs``."""
    if not isinstance(input_str, str):
        return {"input": str(input_str)}
    try:
        decoded = json.loads(input_str)
    except ValueError:
        return {"input": input_str}
    return decoded if isinstance(decoded, dict) else {"input": input_str}
