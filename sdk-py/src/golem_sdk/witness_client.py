"""WitnessClient: Fire-and-forget HTTP client for the Golem Witness.

Sends COSE_Sign1 signed BOM entries to the Witness as protobuf LedgerEntry
messages over ``application/x-protobuf``.

Durable delivery: a POST that fails after a few live retries is written to an
on-disk buffer (~/.golem/sdk-wal) and replayed on the next SDK run, so a
witness that is momentarily down/restarting doesn't cost the ledger an entry.
Disable with GOLEM_SDK_WAL_PATH=off.
"""

import json
import logging
import os
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, NamedTuple, Optional

from golem_sdk.gen.transport_pb import encode_ledger_entry
from golem_sdk.wal import SDKBuffer

logger = logging.getLogger("golem_sdk.witness_client")

# HTTP statuses that mean "stop retrying" on replay: delivered (201), accepted
# (202), already-have-it (409), or a permanent client error (4xx).
_POST_MAX_TRIES = 3


def _resolve_wal_dir() -> Optional[str]:
    """Return the SDK buffer directory, or None when disabled/undeterminable."""
    v = os.environ.get("GOLEM_SDK_WAL_PATH", "").strip()
    if v.lower() in ("off", "disabled", "false", "0"):
        return None
    if v:
        return v
    home = os.path.expanduser("~")
    if not home or home == "~":
        return None
    return os.path.join(home, ".golem", "sdk-wal")


class PolicyDecision(NamedTuple):
    """One synchronous policy answer.

    ``reachable`` is separate from ``allowed`` because they mean different
    things to a caller deciding how much to trust the answer: allowed=True with
    reachable=False is "nobody said no", not "the policy said yes".
    """

    allowed: bool
    action: str
    reason: str
    reachable: bool
    rule_name: str = ""


class ResultDecision(NamedTuple):
    """What to do with a tool's result before the model reads it.

    ``withhold``: hand the model ``notice`` instead of the result. ``advise``:
    hand it the result with ``advice`` beside it. Unreachable reads as allow.
    """

    action: str
    advice: str
    notice: str
    reachable: bool

    @property
    def withheld(self) -> bool:
        return self.action == "withhold"


_RESULT_ALLOW = ResultDecision(action="allow", advice="", notice="", reachable=False)
MAX_RESULT_BYTES = 4 * 1024 * 1024


class WitnessClient:
    """Sends signed BOM entries to the Golem Witness for processing.

    All POSTs are fire-and-forget --- they run in daemon threads and never
    block the caller. Errors are logged, never raised. On persistent failure
    the entry is buffered to disk (see module docstring) and replayed later.
    """

    def __init__(self, witness_url: str = "http://localhost:39393"):
        self.witness_url = witness_url.rstrip("/")
        self._endpoint = f"{self.witness_url}/api/v1/sdk/tool-use"
        self._check_endpoint = f"{self.witness_url}/api/v1/policy-check"
        self._timeout = 5  # seconds
        # Shorter than the record timeout on purpose: this one blocks the agent
        # before every tool call, so a slow witness must cost the caller a
        # moment rather than a multi-second stall on each invocation.
        self._check_timeout = 2  # seconds
        self._result_endpoint = f"{self.witness_url}/api/v1/sdk/tool-result"
        self._usage_endpoint = f"{self.witness_url}/api/v1/sdk/usage"
        self._redaction_endpoint = f"{self.witness_url}/api/v1/redaction-policy"
        self._redaction_timeout = 1  # seconds

        wal_dir = _resolve_wal_dir()
        self._buffer = None
        if wal_dir:
            try:
                self._buffer = SDKBuffer(wal_dir)
            except OSError as e:
                logger.warning("SDK buffer disabled: cannot use %s (%s)", wal_dir, type(e).__name__)
        if self._buffer is not None:
            # Replay any backlog left by a previous run, off the caller's thread.
            threading.Thread(target=self._drain_buffer, daemon=True).start()

    def record(
        self,
        agent_id: str,
        mcp_server_id: str,
        tool_name: str,
        bom_hash: str,
        cbor_payload: bytes,
        cose_payload: bytes,
        public_key_der: bytes,
        session_id: str = "",
        host_id: str = "",
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Post a COSE-signed BOM entry to the Witness in a background thread."""
        body = encode_ledger_entry(
            cose_sign1=cose_payload,
            bom_hash=bom_hash,
            raw_payload=cbor_payload,
            payload_hash=bom_hash,
            session_id=session_id,
            host_id=host_id,
            agent_id=agent_id,
            mcp_server_id=mcp_server_id,
            tool_name=tool_name,
            event_type="sdk_tool_call",
            metadata={**(labels or {}), **(metadata or {})},
            sdk_public_key=public_key_der,
        )
        t = threading.Thread(target=self._post, args=(body, tool_name), daemon=True)
        t.start()

    def fetch_redaction_policy(self) -> Dict:
        """GET this machine's redaction policy. BLOCKS; raises on any failure."""
        req = urllib.request.Request(self._redaction_endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=self._redaction_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("redaction policy is not an object")
        return body

    def check_execution(
        self,
        tool_name: str,
        command: str = "",
        agent_id: str = "",
        session_id: str = "",
        host_id: str = "",
        tool_arguments: Optional[Dict[str, Any]] = None,
        mcp_server_id: str = "",
    ) -> "PolicyDecision":
        """Ask the witness whether this call may run. BLOCKS the caller.

        Deliberately synchronous, unlike every other method here. ``record``
        runs in a daemon thread because the ledger does not care when an entry
        arrives; this one decides whether the tool call happens at all, so an
        answer that lands after execution is not an answer.

        FAILS OPEN. A witness that is down, slow, or unreachable returns
        ``allowed`` with ``reachable=False``. That is the same fail direction
        the in-path MCP interceptor takes and the opposite of the MCP policy
        point's, and the asymmetry is deliberate: this sits in front of a
        customer's agent, so a collector outage that silently bricked every
        tool call would be a worse incident than the one it is guarding
        against. ``reachable`` is returned rather than hidden so a caller that
        wants to fail closed can choose to.
        """
        body = {
            "tool_name": tool_name,
            "command": command,
            "agent_id": agent_id,
            "session_id": session_id,
            "host_id": host_id,
        }
        # Rules match on the arguments and the server, and asset decisions on
        # the server: without them a check carried nothing to match on.
        if tool_arguments:
            body["tool_arguments"] = tool_arguments
        if mcp_server_id:
            body["mcp_server_id"] = mcp_server_id
            body["event_type"] = "sdk_tool_call"
        # default=str: an argument JSON cannot encode must not raise into the
        # caller's tool call from outside the fail-open path below.
        payload = json.dumps(body, default=str).encode("utf-8")
        req = urllib.request.Request(
            self._check_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._check_timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            # Log the type, never the message: a URL or a witness error string
            # can carry host and path detail the SDK has no business emitting.
            logger.debug("policy check unreachable (%s); allowing", type(e).__name__)
            return PolicyDecision(allowed=True, action="allow", reason="", reachable=False)

        action = str(body.get("action") or "allow").lower()
        return PolicyDecision(
            allowed=action not in ("block", "deny"),
            action=action,
            reason=str(body.get("reason") or ""),
            reachable=True,
            rule_name=str(body.get("rule_name") or ""),
        )

    def record_result(self, body: Dict[str, Any]) -> ResultDecision:
        """Record a tool's result and return what the injection check said. BLOCKS.

        Synchronous for the same reason as ``check_execution``: a verdict that
        arrives after the model has read the result is no use. Fails open.
        """
        payload = json.dumps(body, default=str).encode("utf-8")
        if len(payload) > MAX_RESULT_BYTES:
            body = {**body, "result": str(body.get("result"))[: MAX_RESULT_BYTES // 2] + "...[truncated]"}
            payload = json.dumps(body, default=str).encode("utf-8")
        try:
            answer = self._post_json(self._result_endpoint, payload, self._timeout)
        except Exception as e:
            logger.debug("tool result not recorded (%s)", type(e).__name__)
            return _RESULT_ALLOW
        action = str(answer.get("action") or "allow").lower()
        return ResultDecision(
            action=action,
            advice=str(answer.get("advice") or ""),
            notice=str(answer.get("notice") or ""),
            reachable=True,
        )

    def record_usage(self, body: Dict[str, Any]) -> None:
        """Post one model call's token counts in a background thread."""
        payload = json.dumps(body).encode("utf-8")

        def send() -> None:
            try:
                self._post_json(self._usage_endpoint, payload, self._timeout)
            except Exception as e:
                logger.debug("model usage not recorded (%s)", type(e).__name__)

        threading.Thread(target=send, daemon=True).start()

    @staticmethod
    def _post_json(endpoint: str, payload: bytes, timeout: float) -> Dict[str, Any]:
        req = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    def _post(self, body: bytes, tool_name: str = "") -> None:
        """Deliver one entry with a few live retries; buffer to disk on failure."""
        outcome = self._deliver_once(body)
        if outcome == "ok":
            logger.debug("BOM sent to witness (tool=%s)", tool_name)
            return
        if outcome == "permanent":
            logger.warning("witness permanently rejected entry (tool=%s)", tool_name)
            return

        # Transient failure: a couple more quick tries before buffering.
        for _ in range(_POST_MAX_TRIES - 1):
            outcome = self._deliver_once(body)
            if outcome == "ok":
                return
            if outcome == "permanent":
                logger.warning("witness permanently rejected entry (tool=%s)", tool_name)
                return

        if self._buffer is not None:
            self._buffer.append(body)
            logger.debug("entry buffered for later delivery (tool=%s)", tool_name)
        else:
            logger.warning("failed to POST to witness; buffering disabled — entry lost")

    def _deliver_once(self, body: bytes) -> str:
        """One POST attempt. Returns 'ok' | 'permanent' | 'retry'."""
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/x-protobuf"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status in (200, 201, 202, 409):
                    return "ok"
                if 400 <= resp.status < 500:
                    return "permanent"
                return "retry"
        except urllib.error.HTTPError as e:
            if e.code == 409:
                return "ok"  # idempotent — witness already has this entry
            if 400 <= e.code < 500:
                return "permanent"
            return "retry"
        except urllib.error.URLError:
            return "retry"  # network / witness down
        except Exception as e:
            logger.warning("unexpected error posting to witness: %s", type(e).__name__)
            return "retry"

    def _drain_buffer(self) -> None:
        if self._buffer is None:
            return
        self._buffer.drain(lambda b: self._deliver_once(b) in ("ok", "permanent"))
