"""Tests for GolemBridge — Ingest API HTTP client."""

import base64
import json
import threading
import time
from unittest import mock

import pytest

from golem_sdk.bridge import GolemBridge


def test_bridge_posts_correct_json():
    """Record() should POST JSON with base64-encoded cbor_payload and signature."""
    bridge = GolemBridge("http://localhost:8090")

    cbor_payload = b"\xa2dagent_iddtest"
    signature = b"\x30\x44\x02\x20"

    with mock.patch("golem_sdk.bridge.urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock.MagicMock()
        mock_resp.status = 201
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bridge._post(
            agent_id="test-agent",
            mcp_server_id="srv-test",
            tool_name="SearchTickets",
            bom_hash="a" * 64,
            cbor_payload=cbor_payload,
            signature=signature,
            storage_uri="file:///tmp/test.json",
        )

        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        req = call_args[0][0]

        body = json.loads(req.data.decode("utf-8"))
        assert body["agent_id"] == "test-agent"
        assert body["mcp_server_id"] == "srv-test"
        assert body["tool_name"] == "SearchTickets"
        assert body["bom_hash"] == "a" * 64
        assert body["cbor_payload"] == base64.b64encode(cbor_payload).decode("ascii")
        assert body["signature"] == base64.b64encode(signature).decode("ascii")
        assert body["storage_uri"] == "file:///tmp/test.json"


def test_bridge_endpoint_url():
    """Endpoint should be constructed correctly."""
    bridge = GolemBridge("http://myhost:9090/")
    assert bridge._endpoint == "http://myhost:9090/api/v1/ledger"

    bridge2 = GolemBridge("http://myhost:9090")
    assert bridge2._endpoint == "http://myhost:9090/api/v1/ledger"


def test_bridge_graceful_on_connection_error():
    """_post should not raise on connection errors."""
    bridge = GolemBridge("http://127.0.0.1:59999")
    # Should not raise
    bridge._post(
        agent_id="test",
        mcp_server_id="srv",
        tool_name="tool",
        bom_hash="a" * 64,
        cbor_payload=b"data",
        signature=b"sig",
        storage_uri="file:///tmp/test",
    )


def test_bridge_record_is_non_blocking():
    """record() should return immediately (non-blocking)."""
    bridge = GolemBridge("http://127.0.0.1:59999")

    start = time.time()
    bridge.record(
        agent_id="test",
        mcp_server_id="srv",
        tool_name="tool",
        bom_hash="a" * 64,
        cbor_payload=b"data",
        signature=b"sig",
        storage_uri="file:///tmp/test",
    )
    elapsed = time.time() - start

    # record() should return nearly instantly (< 0.1s) since it spawns a thread
    assert elapsed < 0.1


def test_bridge_fields_match_ingest_schema():
    """JSON fields should match the Ingest API's LedgerRequest schema."""
    bridge = GolemBridge()

    with mock.patch("golem_sdk.bridge.urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock.MagicMock()
        mock_resp.status = 201
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bridge._post(
            agent_id="a",
            mcp_server_id="s",
            tool_name="t",
            bom_hash="b" * 64,
            cbor_payload=b"c",
            signature=b"d",
            storage_uri="file:///x",
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))

        required_fields = {"agent_id", "mcp_server_id", "tool_name",
                          "bom_hash", "cbor_payload", "signature", "storage_uri"}
        assert set(body.keys()) == required_fields
