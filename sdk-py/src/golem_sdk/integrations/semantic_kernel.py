"""Microsoft Semantic Kernel integration for the Golem SDK.

Semantic Kernel's observability surface is its **filter system**.
A filter is an async callable that wraps function invocations:

    async def my_filter(context, next):
        # pre-invocation
        await next(context)
        # post-invocation

Filters are registered via ``kernel.add_filter("function_invocation",
my_filter)``. The framework feeds in a ``FunctionInvocationContext``
with attributes ``context.function`` (has ``.name`` and
``.plugin_name``) and ``context.arguments`` (a ``KernelArguments``
mapping).

This adapter **does not hard-import** ``semantic_kernel``. We
duck-type the context — read attributes if they exist, fall back
gracefully. That way the module imports cleanly on a machine
without semantic-kernel installed.

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
        GolemFunctionInvocationFilter(sdk, agent_id="sk-bot"),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from golem_sdk import GolemSDK

logger = logging.getLogger("golem_sdk.integrations.semantic_kernel")


class GolemFunctionInvocationFilter:
    """Async callable that conforms to SK's function-invocation filter signature.

    Records a Golem ``intercept_execution`` event before each
    Semantic Kernel function (a.k.a. "plugin function") runs.
    The original function is then invoked via ``await next(context)``.

    SK's filter pipeline calls this object's ``__call__`` — there
    is no ABC to subclass. Filters are pure async callables.
    """

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
        mcp_server_id: Conceptual "server" identifier. Defaults to
            ``"semantic-kernel"``.
        enforce: If True, consult the witness before each function and
            SKIP the pipeline when policy refuses it. Off by default:
            turning a monitored agent into a gated one is not a default.
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
        # The gate is NOT awaiting next, rather than raising. Semantic Kernel
        # filters are a pipeline: declining to continue it is how this surface
        # expresses a refusal, and an exception here would surface to the caller
        # as a kernel fault instead of a policy decision.
        #
        # Checked outside the try below for the same reason the LangChain
        # adapter raises outside its own: that except exists so a Golem fault
        # never breaks the kernel, and a policy block is not a fault.
        if self._enforce:
            name = _extract_tool_name(context)
            if name:
                decision = self._sdk.check_execution(
                    tool_name=name, agent_id=self._agent_id)
                if not decision.allowed:
                    logger.warning("golem policy blocked %s; not continuing the pipeline", name)
                    return

        try:
            tool_name = _extract_tool_name(context)
            if tool_name:
                args = _extract_arguments(context)
                self._sdk.intercept_execution(
                    agent_id=self._agent_id,
                    mcp_server_id=self._mcp_server_id,
                    tool_name=tool_name,
                    tool_arguments=args,
                )
        except Exception as e:  # noqa: BLE001 — never break the kernel
            logger.warning(
                "golem semantic_kernel filter failed: %s",
                type(e).__name__,
            )
        # Always continue the pipeline — telemetry failure must
        # never short-circuit the user's function call.
        await next(context)


# ----------------------------------------------------------------------
# Duck-typed extractors
# ----------------------------------------------------------------------


def _extract_tool_name(context: Any) -> Optional[str]:
    """Build a fully qualified tool name from the SK context.

    Format: ``{plugin_name}.{function_name}`` when both are
    present (the canonical SK addressing scheme), else just the
    function name. Returns ``None`` when neither is available.

    Looks at:
      * ``context.function.name`` / ``context.function.plugin_name``
      * ``context.function_name`` / ``context.plugin_name`` (older API)
      * dict access for both, in case SK ever hands a plain dict
    """
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
    """Pull the function arguments out of the SK context.

    ``context.arguments`` is a ``KernelArguments`` — a Mapping
    subclass. We try a few paths:
      * ``dict(context.arguments)`` if mapping-like
      * ``context.arguments`` if already a dict
      * empty dict if not present
    """
    if context is None:
        return {}
    args = _get(context, "arguments")
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    # KernelArguments implements Mapping — try dict() conversion.
    try:
        return dict(args)
    except (TypeError, ValueError):
        return {}


def _get(obj: Any, name: str) -> Any:
    """Attribute-or-dict-key getter. SK normally hands us a Pydantic
    model (attribute access); we also accept dicts for forward
    compatibility and easier testing."""
    if obj is None:
        return None
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(name)
    return None
