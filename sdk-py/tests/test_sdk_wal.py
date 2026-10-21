"""Tests for the SDK durable-delivery buffer + witness-client integration."""

import http.server
import threading

import pytest

from golem_sdk.wal import SDKBuffer
from golem_sdk.witness_client import WitnessClient


def test_buffer_append_and_drain_success(tmp_path):
    buf = SDKBuffer(str(tmp_path))
    for i in range(3):
        buf.append(f"body-{i}".encode())
    assert buf.pending() == 3

    got = []
    buf.drain(lambda b: (got.append(b), True)[1])  # deliver everything
    assert buf.pending() == 0
    assert got == [b"body-0", b"body-1", b"body-2"]  # oldest-first


def test_buffer_drain_stops_on_retryable(tmp_path):
    buf = SDKBuffer(str(tmp_path))
    buf.append(b"a")
    buf.append(b"b")

    attempts = []
    buf.drain(lambda b: (attempts.append(b), False)[1])  # backend down
    assert len(attempts) == 1  # stops at first failure
    assert buf.pending() == 2  # nothing dropped

    buf.drain(lambda b: True)  # recovered
    assert buf.pending() == 0


def test_buffer_cap_evicts_oldest(tmp_path):
    buf = SDKBuffer(str(tmp_path), max_records=3)
    for i in range(6):
        buf.append(f"e-{i}".encode())
    assert buf.pending() == 3
    survivors = []
    buf.drain(lambda b: (survivors.append(b), True)[1])
    assert survivors == [b"e-3", b"e-4", b"e-5"]


class _MockWitness(http.server.BaseHTTPRequestHandler):
    received = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        _MockWitness.received += 1
        self.send_response(201)
        self.end_headers()

    def log_message(self, *a):
        pass


def _serve():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _MockWitness)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_failed_post_is_buffered_then_replayed(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLEM_SDK_WAL_PATH", str(tmp_path))
    _MockWitness.received = 0

    # 1. Witness unreachable → _post buffers the entry.
    down = WitnessClient(witness_url="http://127.0.0.1:1")
    down._post(b"protobuf-entry", tool_name="read_file")
    assert down._buffer.pending() == 1

    # 2. A fresh client pointed at a live witness drains the backlog on demand.
    srv = _serve()
    try:
        up = WitnessClient(witness_url=f"http://127.0.0.1:{srv.server_address[1]}")
        up._drain_buffer()
        assert up._buffer.pending() == 0
        assert _MockWitness.received == 1
    finally:
        srv.shutdown()


def test_wal_disabled(monkeypatch):
    monkeypatch.setenv("GOLEM_SDK_WAL_PATH", "off")
    c = WitnessClient(witness_url="http://127.0.0.1:1")
    assert c._buffer is None
    # A failed post with no buffer must not raise.
    c._post(b"x", tool_name="t")


def test_unusable_wal_dir_disables_buffering_instead_of_raising(tmp_path, monkeypatch):
    # A read-only home, a container or a sandbox can refuse the buffer folder.
    # Recording must never break the program that uses the SDK.
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setenv("GOLEM_SDK_WAL_PATH", str(blocker / "sdk-wal"))
    c = WitnessClient(witness_url="http://127.0.0.1:1")
    assert c._buffer is None
    c._post(b"x", tool_name="t")
