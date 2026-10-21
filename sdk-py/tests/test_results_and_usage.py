import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from golem_sdk import GolemSDK


class _Agent:
    def __init__(self, answers):
        self.seen = []
        agent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                body = json.loads(raw) if self.headers.get("Content-Type") == "application/json" else raw
                agent.seen.append((self.path, body))
                payload = json.dumps(answers.get(self.path, {"status": "ok"})).encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}"

    def bodies(self, path, count=1, wait=3.0):
        deadline = time.time() + wait
        while time.time() < deadline:
            got = [b for p, b in self.seen if p == path]
            if len(got) >= count:
                return got
            time.sleep(0.02)
        return [b for p, b in self.seen if p == path]


def _sdk(agent, monkeypatch):
    monkeypatch.setenv("GOLEM_SDK_WAL_PATH", "off")
    return GolemSDK(witness_url=agent.url, session_id="run-1")


def test_a_result_pairs_with_its_call_and_a_withheld_one_says_what_to_hand_the_model(monkeypatch):
    agent = _Agent({"/api/v1/sdk/tool-result": {
        "status": "ok", "action": "withhold", "notice": "[Golem withheld this result]"}})
    try:
        sdk = _sdk(agent, monkeypatch)
        call = sdk.intercept_execution("bot", "langchain", "fetch_page", {"url": "https://blog.example"})
        d = sdk.record_result("fetch_page", {"text": "page"}, tool_use_id=call["tool_use_id"],
                              agent_id="bot", mcp_server_id="langchain", duration_ms=12)
        sent = agent.bodies("/api/v1/sdk/tool-result")[0]
    finally:
        agent.srv.shutdown()
    assert call["tool_use_id"].startswith("call-")
    assert d.withheld and d.reachable and d.notice == "[Golem withheld this result]"
    assert sent["tool_use_id"] == call["tool_use_id"]
    assert sent["session_id"] == "run-1"
    assert sent["result"] == {"text": "page"}
    assert sent["mcp_server_id"] == "langchain"


def test_the_call_carries_the_id_its_result_pairs_on(monkeypatch):
    agent = _Agent({})
    try:
        sdk = _sdk(agent, monkeypatch)
        sdk.intercept_execution("bot", "langchain", "search", {"q": "x"}, tool_use_id="run-1:7")
        raw = agent.bodies("/api/v1/sdk/tool-use")[0]
    finally:
        agent.srv.shutdown()
    assert b"tool_use_id" in raw and b"run-1:7" in raw


def test_a_result_the_agent_cannot_take_fails_open(monkeypatch):
    monkeypatch.setenv("GOLEM_SDK_WAL_PATH", "off")
    d = GolemSDK(witness_url="http://127.0.0.1:9").record_result("search", object())
    assert d.action == "allow" and not d.withheld and not d.reachable


def test_model_usage_is_sent_as_tokens(monkeypatch):
    agent = _Agent({})
    try:
        sdk = _sdk(agent, monkeypatch)
        sdk.record_usage("claude-sonnet-5-5", input_tokens=900, output_tokens=40, provider="anthropic")
        sdk.record_usage("", input_tokens=1)
        sent = agent.bodies("/api/v1/sdk/usage")
    finally:
        agent.srv.shutdown()
    assert len(sent) == 1
    assert sent[0]["model"] == "claude-sonnet-5-5"
    assert (sent[0]["input_tokens"], sent[0]["output_tokens"]) == (900, 40)
    assert sent[0]["session_id"] == "run-1"
