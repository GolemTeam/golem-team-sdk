"""Tests for the WitnessClient and witness mode in GolemSDK.

The witness mode transport is protobuf: the SDK POSTs a ``LedgerEntry``
protobuf message (Content-Type ``application/x-protobuf``) to
``/api/v1/sdk/tool-use``. These tests decode that protobuf body with a
minimal wire-format parser (there are no generated bindings in the SDK —
only the hand-rolled encoder in ``golem_sdk.gen.transport_pb``) and assert
on exactly the fields the Go witness handler requires.
"""

import base64  # noqa: F401 (kept for parity / ad-hoc debugging)
import hashlib
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cbor2
import pytest
from golem_sdk.sdk import GolemSDK
from golem_sdk.witness_client import WitnessClient  # noqa: F401


# --- Minimal protobuf wire-format decoder ---------------------------------
#
# Enough of the protobuf wire format to parse a LedgerEntry: varint (0) and
# length-delimited (2) fields. Returns {field_number: [values]}; length-
# delimited values are returned as raw bytes for the caller to interpret.

def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def decode_protobuf(buf):
    """Decode protobuf bytes into {field_number: [values]}."""
    fields = {}
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # varint
            val, pos = _read_varint(buf, pos)
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            val = bytes(buf[pos:pos + length])
            pos += length
        elif wire_type == 5:  # fixed32
            val = bytes(buf[pos:pos + 4])
            pos += 4
        elif wire_type == 1:  # fixed64
            val = bytes(buf[pos:pos + 8])
            pos += 8
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        fields.setdefault(field_num, []).append(val)
    return fields


def parse_ledger_entry(body):
    """Parse a LedgerEntry protobuf body into a friendly dict.

    Mirrors the field numbers in proto/golem/v1/transport.proto and the
    fields the witness's sdk_handler.go reads.
    """
    f = decode_protobuf(body)
    entry = {}

    # Field 1: envelope (embedded SignedEnvelope)
    if 1 in f:
        env = decode_protobuf(f[1][0])
        entry["cose_sign1"] = env[1][0] if 1 in env else b""
        entry["bom_hash"] = env[2][0].decode() if 2 in env else ""

    def s(num):
        return f[num][0].decode() if num in f else ""

    entry["raw_payload"] = f[4][0] if 4 in f else b""
    entry["session_id"] = s(7)
    entry["host_id"] = s(8)
    entry["agent_id"] = s(9)
    entry["mcp_server_id"] = s(10)
    entry["tool_name"] = s(11)
    entry["event_type"] = s(12)
    entry["sdk_public_key"] = f[17][0] if 17 in f else b""
    return entry


# --- Mock Witness Server ---------------------------------------------------

class MockWitnessHandler(BaseHTTPRequestHandler):
    """Records requests for test assertions."""

    requests = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        MockWitnessHandler.requests.append({
            "path": self.path,
            "content_type": self.headers.get("Content-Type", ""),
            "entry": parse_ledger_entry(body),
        })
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format, *args):
        pass  # Suppress log output during tests


@pytest.fixture
def mock_witness():
    """Start a mock witness server on a random port."""
    MockWitnessHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), MockWitnessHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", MockWitnessHandler.requests
    server.shutdown()


SAMPLE_ARGS = dict(
    agent_id="test-agent",
    mcp_server_id="test-server",
    tool_name="read_file",
    tool_arguments={"path": "/etc/hosts"},
)


def _wait_for_request(requests, timeout=2.0, count=1):
    """Poll until the daemon POST threads have recorded `count` requests."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(requests) >= count:
            return
        time.sleep(0.01)


class TestWitnessClient:
    def test_posts_to_correct_endpoint(self, mock_witness):
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)

        assert len(requests) == 1
        assert requests[0]["path"] == "/api/v1/sdk/tool-use"

    def test_sends_protobuf_content_type(self, mock_witness):
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)
        assert requests[0]["content_type"] == "application/x-protobuf"

    def test_sends_required_fields(self, mock_witness):
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)

        entry = requests[0]["entry"]
        # Fields the Golem agent's SDK handler requires.
        assert entry["cose_sign1"]
        assert entry["bom_hash"]
        assert entry["sdk_public_key"]
        assert entry["agent_id"] == "test-agent"
        assert entry["mcp_server_id"] == "test-server"
        assert entry["tool_name"] == "read_file"

    def test_bom_hash_is_valid_sha256(self, mock_witness):
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        result = sdk.intercept_execution(**SAMPLE_ARGS)

        assert len(result["bom_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["bom_hash"])

    def test_public_key_is_valid_der(self, mock_witness):
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)

        # sdk_public_key is DER SPKI — P-256 SPKI DER is 91 bytes and must
        # parse as an EC public key (this is exactly what the witness does).
        pub_key_der = requests[0]["entry"]["sdk_public_key"]
        assert len(pub_key_der) == 91

        from cryptography.hazmat.primitives.serialization import (
            load_der_public_key,
        )
        from cryptography.hazmat.primitives.asymmetric import ec
        pub = load_der_public_key(pub_key_der)
        assert isinstance(pub, ec.EllipticCurvePublicKey)

    def test_bom_hash_matches_inner_cbor(self, mock_witness):
        """The witness recomputes sha256(inner CBOR) and compares to bom_hash."""
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)

        entry = requests[0]["entry"]
        # Inner CBOR is the COSE_Sign1 payload (index 2 of the tagged array).
        cose = cbor2.loads(entry["cose_sign1"])
        inner = cose.value[2]
        assert hashlib.sha256(inner).hexdigest() == entry["bom_hash"]
        # And the raw_payload field carries the same CBOR bytes.
        assert entry["raw_payload"] == inner

    def test_cose_signature_is_verifiable(self, mock_witness):
        """The COSE_Sign1 signature verifies against the sent public key.

        This mirrors the witness's cosesign.Verify: rebuild the RFC 9052
        Sig_structure = ["Signature1", protected, external_aad, payload]
        and verify the raw r||s ECDSA signature over its SHA-256.
        """
        import ecdsa
        from ecdsa.util import sigdecode_string

        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)

        entry = requests[0]["entry"]
        cose = cbor2.loads(entry["cose_sign1"])
        protected, _unprotected, payload, signature = cose.value

        sig_structure = cbor2.dumps(
            ["Signature1", protected, b"", payload], canonical=True
        )

        vk = ecdsa.VerifyingKey.from_der(entry["sdk_public_key"])
        # Raises BadSignatureError if invalid.
        assert vk.verify(
            signature, sig_structure,
            hashfunc=hashlib.sha256, sigdecode=sigdecode_string,
        )

    def test_witness_mode_returns_no_storage_uri(self, mock_witness):
        """In witness mode, storage_uri is not returned (witness handles BYOB)."""
        url, _ = mock_witness
        sdk = GolemSDK(witness_url=url)
        result = sdk.intercept_execution(**SAMPLE_ARGS)

        assert "storage_uri" not in result

    def test_session_and_host_id_forwarded(self, mock_witness):
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url, session_id="sess-123", host_id="my-laptop")
        sdk.intercept_execution(**SAMPLE_ARGS)

        _wait_for_request(requests)

        entry = requests[0]["entry"]
        assert entry["session_id"] == "sess-123"
        assert entry["host_id"] == "my-laptop"

    def test_calls_share_one_session_by_default(self, mock_witness):
        # Without an id the Golem agent made a new session for every call, so a
        # run's calls never chained or read as one conversation.
        url, requests = mock_witness
        sdk = GolemSDK(witness_url=url)
        sdk.intercept_execution(**SAMPLE_ARGS)
        sdk.intercept_execution(**SAMPLE_ARGS)
        _wait_for_request(requests, count=2)

        sessions = {r["entry"]["session_id"] for r in requests}
        assert sessions == {sdk.session_id}
        assert sdk.session_id.startswith("sdk-")
        assert GolemSDK(witness_url=url).session_id != sdk.session_id

    def test_graceful_on_witness_unavailable(self):
        """SDK should not raise if witness is unreachable."""
        sdk = GolemSDK(witness_url="http://127.0.0.1:1")  # nothing listening
        # Should not raise — fire-and-forget
        result = sdk.intercept_execution(**SAMPLE_ARGS)
        assert len(result["bom_hash"]) == 64

    def test_legacy_mode_emits_deprecation_warning(self):
        """Using storage= or bridge= should emit a DeprecationWarning."""
        from golem_sdk.storage import LocalDiskStorage
        with pytest.warns(DeprecationWarning, match="deprecated"):
            GolemSDK(storage=LocalDiskStorage())
