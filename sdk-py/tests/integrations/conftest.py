"""A fake Golem agent for driving the real frameworks end to end.

It answers the three endpoints an SDK talks to, keeps every recorded entry and
policy check, and logs, in order, when the SDK recorded a call and when the
tool's own body ran, so a test can assert the record came first.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import cbor2
import pytest

from golem_sdk import GolemSDK
from tests.test_witness_client import parse_ledger_entry


class FakeGolem:
    def __init__(self) -> None:
        self.policy: Dict[str, Any] = {"action": "allow"}
        self.entries: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []
        self.result_policy: Dict[str, Any] = {"action": "allow"}
        self.results: List[Dict[str, Any]] = []
        self.usage: List[Dict[str, Any]] = []
        self.call_ids: List[str] = []
        self.events: List[tuple] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def block(self, reason: str) -> None:
        self.policy = {"action": "block", "reason": reason, "rule_name": "test-rule"}

    def withhold(self, notice: str) -> None:
        self.result_policy = {"action": "withhold", "notice": notice}

    def sdk(self, **kwargs: Any) -> GolemSDK:
        sdk = GolemSDK(witness_url=self.url, **kwargs)
        record = sdk.intercept_execution
        record_result = sdk.record_result

        def spy(**kw: Any) -> Dict[str, Any]:
            self.events.append(("recorded", kw.get("tool_name")))
            out = record(**kw)
            self.call_ids.append(out["tool_use_id"])
            return out

        def result_spy(*args: Any, **kw: Any) -> Any:
            self.events.append(("result", kw.get("tool_name", args[0] if args else None)))
            return record_result(*args, **kw)

        sdk.intercept_execution = spy
        sdk.record_result = result_spy
        return sdk

    def wait_usage(self, n: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        end = time.time() + timeout
        while time.time() < end and len(self.usage) < n:
            time.sleep(0.02)
        return list(self.usage)

    def ran(self, tool_name: str) -> None:
        self.events.append(("ran", tool_name))

    def wait_entries(self, n: int, timeout: float = 5.0) -> List[Dict[str, Any]]:
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if len(self.entries) >= n:
                    break
            time.sleep(0.02)
        # Long enough for an unexpected extra entry to land and fail the count.
        time.sleep(0.3)
        with self._lock:
            return list(self.entries)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler(self):
        golem = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, obj: Any, code: int = 200) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/api/v1/redaction-policy":
                    return self._json({"enabled": False})
                self._json({}, 404)

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/api/v1/sdk/tool-use":
                    entry = parse_ledger_entry(body)
                    entry["bom"] = cbor2.loads(entry["raw_payload"])
                    with golem._lock:
                        golem.entries.append(entry)
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif self.path == "/api/v1/policy-check":
                    with golem._lock:
                        golem.checks.append(json.loads(body))
                    self._json(golem.policy)
                elif self.path == "/api/v1/sdk/tool-result":
                    with golem._lock:
                        golem.results.append(json.loads(body))
                    self._json(golem.result_policy)
                elif self.path == "/api/v1/sdk/usage":
                    with golem._lock:
                        golem.usage.append(json.loads(body))
                    self._json({})
                else:
                    self._json({}, 404)

        return Handler


@pytest.fixture
def golem():
    g = FakeGolem()
    try:
        yield g
    finally:
        g.close()


def only_entry(golem: FakeGolem, tool_name: Optional[str] = None) -> Dict[str, Any]:
    entries = golem.wait_entries(1)
    assert len(entries) == 1, [e["tool_name"] for e in entries]
    if tool_name is not None:
        assert entries[0]["tool_name"] == tool_name
    return entries[0]
