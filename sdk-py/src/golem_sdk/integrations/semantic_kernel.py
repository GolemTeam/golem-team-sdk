"""Microsoft Semantic Kernel integration for the Golem SDK.

``GolemFunctionInvocationFilter`` is a function-invocation filter: Semantic
Kernel runs it around every kernel function, whether you call
``kernel.invoke`` or the model calls the function through automatic function
calling. It records each call, with its arguments, before the function runs and
its result after. With ``enforce=True`` a call your policy blocks does not run
and its result is ``"Blocked by Golem policy: <reason>"``; a result the Golem
agent withholds is replaced by its notice. Either way the replacement is what
``kernel.invoke`` returns and what the model reads.

Usage:

    from semantic_kernel import Kernel
    from golem_sdk import GolemSDK
    from golem_sdk.integrations.semantic_kernel import (
        GolemFunctionInvocationFilter,
    )

    sdk = GolemSDK(witness_url="http://localhost:39393")
    kernel = Kernel()
    kernel.add_filter(
        "function_invocation",
        GolemFunctionInvocationFilter(sdk, agent_id="sk-bot", enforce=True),
    )

This module imports without ``semantic_kernel``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from golem_sdk import GolemSDK
from golem_sdk.integrations._calls import ToolCall
from golem_sdk.policy import blocked_message

logger = logging.getLogger("golem_sdk.integrations.semantic_kernel")


class GolemFunctionInvocationFilter:
    """Async callable with Semantic Kernel's function-invocation filter signature."""

    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str = "semantic-kernel",
        enforce: bool = False,
    ) -> None:
        """Args:
        sdk: A ``GolemSDK`` instance.
        agent_id: Identifier for the agent that owns this filter.
        mcp_server_id: The server the calls are judged and recorded under.
            Defaults to ``"semantic-kernel"``.
        enforce: If True, ask the Golem agent before each function runs and
            answer a refused call with a refusal result instead of running it.
            Off by default.
        """
        self._sdk = sdk
        self._agent_id = agent_id
        self._mcp_server_id = mcp_server_id
        self._enforce = enforce

    async def __call__(
        self,
        context: Any,
        next: Callable[[Any], Awaitable[None]],
    ) -> None:
        name = _extract_tool_name(context)
        if not name:
            await next(context)
            return
        call = ToolCall(self._sdk, self._agent_id, self._mcp_server_id, name,
                        _extract_arguments(context))

        # The refusal is a result, not an exception: an exception would reach
        # the caller as a kernel fault, and the model would never read why.
        if self._enforce:
            decision = await asyncio.to_thread(call.check)
            if not decision.allowed:
                logger.warning("golem policy blocked a kernel function; not running it")
                _set_result(context, blocked_message(decision.reason))
                return

        await asyncio.to_thread(call.record)
        try:
            await next(context)
        except Exception as e:
            await asyncio.to_thread(call.finish, f"{type(e).__name__}: {e}", True)
            raise
        if _get(context, "is_streaming"):
            return
        result = _get(context, "result")
        notice = await asyncio.to_thread(call.finish, getattr(result, "value", result))
        if notice is not None:
            _set_result(context, notice)


def _set_result(context: Any, text: str) -> None:
    try:
        from semantic_kernel.functions.function_result import FunctionResult

        context.result = FunctionResult(function=context.function.metadata, value=text)
    except Exception as e:  # noqa: BLE001 — the function still does not run
        logger.warning("golem could not set the kernel function's result: %s", type(e).__name__)


def _extract_tool_name(context: Any) -> Optional[str]:
    """``{plugin_name}.{function_name}``, the canonical SK address, or just the
    function name when it has no plugin."""
    if context is None:
        return None

    function = _get(context, "function")
    fn_name = _get(function, "name") or _get(context, "function_name")
    plugin_name = _get(function, "plugin_name") or _get(context, "plugin_name")

    if isinstance(fn_name, str) and fn_name:
        if isinstance(plugin_name, str) and plugin_name:
            return f"{plugin_name}.{fn_name}"
        return fn_name
    return None


def _extract_arguments(context: Any) -> Dict[str, Any]:
    if context is None:
        return {}
    args = _get(context, "arguments")
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    try:
        return dict(args)
    except (TypeError, ValueError):
        return {}


def _get(obj: Any, name: str) -> Any:
    """Attribute or dict key: SK hands over Pydantic models."""
    if obj is None:
        return None
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name)
    return None
