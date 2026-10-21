"""One tool call as every adapter reports it: checked, recorded, then its result."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, Optional

from golem_sdk import GolemSDK
from golem_sdk.witness_client import PolicyDecision, ResultDecision, _RESULT_ALLOW

logger = logging.getLogger("golem_sdk.integrations")

_SAFE_ID = re.compile(r"^[\w\-.:/]{1,255}$")
_WITHHELD = "[Golem withheld this tool result]"


class ToolCall:
    def __init__(
        self,
        sdk: GolemSDK,
        agent_id: str,
        mcp_server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> None:
        self.sdk = sdk
        self.agent_id = agent_id
        self.mcp_server_id = mcp_server_id
        self.tool_name = tool_name or "unknown"
        self.arguments = arguments
        self.tool_use_id = call_id if isinstance(call_id, str) and _SAFE_ID.match(call_id) else ""
        self._started = time.monotonic()

    def check(self) -> PolicyDecision:
        return self.sdk.check_execution(
            tool_name=self.tool_name,
            agent_id=self.agent_id,
            tool_arguments=self.arguments,
            mcp_server_id=self.mcp_server_id,
        )

    def record(self) -> None:
        try:
            out = self.sdk.intercept_execution(
                agent_id=self.agent_id,
                mcp_server_id=self.mcp_server_id,
                tool_name=self.tool_name,
                tool_arguments=self.arguments,
                tool_use_id=self.tool_use_id,
            )
            if isinstance(out, dict) and isinstance(out.get("tool_use_id"), str):
                self.tool_use_id = out["tool_use_id"]
        except Exception as e:  # noqa: BLE001 — never break the agent
            logger.warning("golem: recording a tool call failed: %s", type(e).__name__)
        self._started = time.monotonic()

    def finish(self, result: Any, is_error: bool = False) -> Optional[str]:
        """Record what the tool returned. Returns what the model should read in
        its place when Golem withholds it, else None."""
        decision = self._record_result(result, is_error)
        if not decision.withheld:
            return None
        return decision.notice or _WITHHELD

    def _record_result(self, result: Any, is_error: bool) -> ResultDecision:
        try:
            decision = self.sdk.record_result(
                tool_name=self.tool_name,
                result=result,
                tool_use_id=self.tool_use_id,
                agent_id=self.agent_id,
                mcp_server_id=self.mcp_server_id,
                is_error=is_error,
                duration_ms=int((time.monotonic() - self._started) * 1000),
            )
        except Exception as e:  # noqa: BLE001 — never break the agent
            logger.warning("golem: recording a tool result failed: %s", type(e).__name__)
            return _RESULT_ALLOW
        return decision if isinstance(decision, ResultDecision) else _RESULT_ALLOW
