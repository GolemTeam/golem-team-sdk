"""The synchronous pre-execution policy check (Phase-0 WP-0e).

What is being tested is a FAIL DIRECTION and a placement, not a happy path. Both
are easy to get wrong in a way that looks fine: a check that fails closed bricks a
customer's agent on a collector restart, and a refusal raised inside the adapter's
catch-all enforces nothing while appearing to.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from golem_sdk.policy import GolemPolicyBlocked
from golem_sdk.witness_client import WitnessClient


def _serve(response: dict, status: int = 200):
    """Run a one-shot witness stand-in and return its base URL plus a stopper."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(n) or b"{}")
            payload = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", srv, seen


def test_block_is_reported_as_not_allowed():
    url, srv, seen = _serve({"action": "block", "reason": "policy block"})
    try:
        d = WitnessClient(witness_url=url).check_execution(
            tool_name="Bash", command="curl evil.example", agent_id="a1")
    finally:
        srv.shutdown()

    assert d.allowed is False
    assert d.action == "block"
    assert d.reachable is True
    # It must reach the endpoint that actually evaluates policy. Posting to the
    # recording endpoint would return 2xx and read as an allow forever.
    assert seen["path"] == "/api/v1/policy-check"
    assert seen["body"]["command"] == "curl evil.example"


def test_allow_is_allowed():
    url, srv, _ = _serve({"action": "allow", "reason": "policy allow"})
    try:
        d = WitnessClient(witness_url=url).check_execution(tool_name="Read")
    finally:
        srv.shutdown()
    assert d.allowed is True
    assert d.reachable is True


def test_a_gate_degraded_to_block_still_blocks():
    """The sync surface cannot hold a human gate, so the witness returns block.

    Asserted because the SDK must not try to interpret ``original_action`` and
    let a gated call through on the grounds that it was "only" a gate.
    """
    url, srv, _ = _serve({
        "action": "block", "original_action": "gate",
        "degraded_from_gate": True, "reason": "policy block",
    })
    try:
        d = WitnessClient(witness_url=url).check_execution(tool_name="Bash")
    finally:
        srv.shutdown()
    assert d.allowed is False


def test_an_unreachable_witness_fails_OPEN():
    """THE fail direction. A collector outage must not brick the agent.

    Port 1 is reserved and nothing listens there, so this is a real connection
    failure rather than a mocked one. The opposite behaviour — fail closed —
    would turn every witness restart into a fleet-wide outage, which is worse
    than the attack this check exists to stop.
    """
    d = WitnessClient(witness_url="http://127.0.0.1:1").check_execution(tool_name="Bash")
    assert d.allowed is True
    # And it must SAY it could not ask, so a caller that wants to fail closed can.
    assert d.reachable is False


def test_a_malformed_response_fails_open():
    url, srv, _ = _serve({"unexpected": "shape"})
    try:
        d = WitnessClient(witness_url=url).check_execution(tool_name="Bash")
    finally:
        srv.shutdown()
    assert d.allowed is True


def test_a_5xx_fails_open():
    url, srv, _ = _serve({"error": "boom"}, status=500)
    try:
        d = WitnessClient(witness_url=url).check_execution(tool_name="Bash")
    finally:
        srv.shutdown()
    assert d.allowed is True
    assert d.reachable is False


def test_langchain_raises_on_block_and_the_catch_all_does_not_eat_it():
    """The refusal must escape on_tool_start's ``except Exception``.

    That handler swallows everything so a Golem fault never breaks a customer's
    chain. A policy block is not a fault and goes the other way, so the check is
    placed before the try — this test is what pins that placement, since moving
    it inside would silently disable enforcement while every other test passed.
    """
    pytest.importorskip("langchain_core")
    from golem_sdk.integrations.langchain import GolemCallbackHandler

    class FakeSDK:
        def check_execution(self, **kw):
            from golem_sdk.witness_client import PolicyDecision
            return PolicyDecision(allowed=False, action="block",
                                  reason="policy block", reachable=True)

        def intercept_execution(self, **kw):
            raise AssertionError("the tool was recorded despite being blocked")

    h = GolemCallbackHandler(sdk=FakeSDK(), agent_id="a1", enforce=True)
    with pytest.raises(GolemPolicyBlocked) as e:
        h.on_tool_start({"name": "Bash"}, "curl evil.example", run_id=None)
    assert e.value.tool_name == "Bash"
    assert "policy block" in e.value.reason


def test_langchain_does_not_enforce_by_default():
    """Off unless asked. Turning a monitored agent into a gated one is not a
    default, and an SDK upgrade must not start blocking a customer's tools."""
    pytest.importorskip("langchain_core")
    from golem_sdk.integrations.langchain import GolemCallbackHandler

    checked = []

    class FakeSDK:
        def check_execution(self, **kw):
            checked.append(kw)
            raise AssertionError("checked policy without being asked to enforce")

        def intercept_execution(self, **kw):
            pass

    h = GolemCallbackHandler(sdk=FakeSDK(), agent_id="a1")
    h.on_tool_start({"name": "Bash"}, "ls", run_id=None)
    assert checked == []


def test_semantic_kernel_declines_the_pipeline_on_block():
    """The SK surface refuses by NOT awaiting next, not by raising.

    A filter is a pipeline stage, so declining to continue is how a refusal is
    expressed here; an exception would reach the caller as a kernel fault rather
    than a policy decision. Asserted by observing that next was never awaited.
    """
    from golem_sdk.integrations.semantic_kernel import GolemFunctionInvocationFilter

    continued = []

    async def next_stage(ctx):
        continued.append(ctx)

    class FakeSDK:
        def check_execution(self, **kw):
            from golem_sdk.witness_client import PolicyDecision
            return PolicyDecision(allowed=False, action="block",
                                  reason="policy block", reachable=True)

        def intercept_execution(self, **kw):
            raise AssertionError("recorded a call that policy refused")

    class Ctx:
        class function:
            name = "dangerous_fn"
            plugin_name = "p"

    import asyncio
    f = GolemFunctionInvocationFilter(sdk=FakeSDK(), agent_id="a1", enforce=True)
    asyncio.run(f(Ctx(), next_stage))
    assert continued == [], "the pipeline continued despite a policy block"


def test_semantic_kernel_continues_when_allowed():
    from golem_sdk.integrations.semantic_kernel import GolemFunctionInvocationFilter

    continued = []

    async def next_stage(ctx):
        continued.append(ctx)

    class FakeSDK:
        def check_execution(self, **kw):
            from golem_sdk.witness_client import PolicyDecision
            return PolicyDecision(allowed=True, action="allow", reason="", reachable=True)

        def intercept_execution(self, **kw):
            pass

    class Ctx:
        class function:
            name = "safe_fn"
            plugin_name = "p"

    import asyncio
    f = GolemFunctionInvocationFilter(sdk=FakeSDK(), agent_id="a1", enforce=True)
    asyncio.run(f(Ctx(), next_stage))
    assert len(continued) == 1


def test_a_refusal_is_recorded_under_the_sdks_own_session():
    # The check sent an empty session and host, so a refusal landed in the
    # agent's per-host fallback session instead of the run it stopped.
    from golem_sdk import GolemSDK

    url, srv, seen = _serve({"action": "block", "rule_name": "r", "reason": "no"})
    try:
        sdk = GolemSDK(witness_url=url, session_id="run-42", host_id="ci-runner")
        sdk.check_execution(tool_name="Bash", command="rm -rf /", agent_id="a")
    finally:
        srv.shutdown()
    assert seen["body"]["session_id"] == "run-42"
    assert seen["body"]["host_id"] == "ci-runner"


def test_the_check_carries_the_arguments_and_server_rules_match_on():
    # The check sent only a tool name and a command, so argument rules and
    # inventory decisions had nothing to match an SDK call against.
    url, srv, seen = _serve({"action": "block", "rule_name": "no-deletes", "reason": "no"})
    try:
        d = WitnessClient(witness_url=url).check_execution(
            tool_name="drop_table", agent_id="a", session_id="s",
            tool_arguments={"table": "users"}, mcp_server_id="langchain")
    finally:
        srv.shutdown()
    assert seen["body"]["tool_arguments"] == {"table": "users"}
    assert seen["body"]["mcp_server_id"] == "langchain"
    assert seen["body"]["event_type"] == "sdk_tool_call"
    assert d.allowed is False and d.rule_name == "no-deletes"


def test_an_argument_json_cannot_encode_does_not_raise_into_the_call():
    url, srv, seen = _serve({"action": "allow"})
    try:
        d = WitnessClient(witness_url=url).check_execution(
            tool_name="t", tool_arguments={"when": object()}, mcp_server_id="x")
    finally:
        srv.shutdown()
    assert d.allowed is True and d.reachable is True
    assert isinstance(seen["body"]["tool_arguments"]["when"], str)
