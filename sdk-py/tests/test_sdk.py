import hashlib
import os
import json
import tempfile
import shutil
import pytest

import cbor2

from golem_sdk.sdk import GolemSDK
from golem_sdk.storage import LocalDiskStorage


@pytest.fixture
def tmp_storage_dir():
    d = tempfile.mkdtemp(prefix="golem-test-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sdk(tmp_storage_dir):
    """Legacy-mode SDK for backward compat tests."""
    storage = LocalDiskStorage(base_path=tmp_storage_dir)
    return GolemSDK(storage=storage)


@pytest.fixture
def witness_sdk():
    """Witness-mode SDK (no real witness, we test signing only)."""
    return GolemSDK(witness_url="http://localhost:59999")


SAMPLE_ARGS = dict(
    agent_id="agent-test-001",
    mcp_server_id="mcp-server-001",
    tool_name="read_file",
    tool_arguments={"path": "/etc/hosts"},
)


class TestGolemSDKLegacy:
    """Tests for legacy mode (backward compatibility with ECDSA signing)."""

    def test_instantiation(self, sdk):
        assert isinstance(sdk, GolemSDK)

    def test_intercept_returns_required_keys(self, sdk):
        result = sdk.intercept_execution(**SAMPLE_ARGS)
        assert "bom_hash" in result
        assert "storage_uri" in result
        assert "signature" in result
        assert "cbor_payload" in result

    def test_bom_hash_is_valid_sha256_hex(self, sdk):
        result = sdk.intercept_execution(**SAMPLE_ARGS)
        bom_hash = result["bom_hash"]
        assert len(bom_hash) == 64
        assert all(c in "0123456789abcdef" for c in bom_hash)

    def test_signature_is_nonempty_bytes(self, sdk):
        result = sdk.intercept_execution(**SAMPLE_ARGS)
        assert isinstance(result["signature"], bytes)
        assert len(result["signature"]) > 0

    def test_cbor_payload_is_valid_bytes(self, sdk):
        result = sdk.intercept_execution(**SAMPLE_ARGS)
        assert isinstance(result["cbor_payload"], bytes)
        assert len(result["cbor_payload"]) > 0

    def test_different_inputs_produce_different_hashes(self, sdk):
        result1 = sdk.intercept_execution(**SAMPLE_ARGS)
        result2 = sdk.intercept_execution(
            agent_id="agent-test-002",
            mcp_server_id="mcp-server-002",
            tool_name="write_file",
            tool_arguments={"path": "/tmp/out", "content": "hello"},
        )
        assert result1["bom_hash"] != result2["bom_hash"]

    def test_consistent_hashes_for_same_input(self, sdk):
        result1 = sdk.intercept_execution(**SAMPLE_ARGS)
        result2 = sdk.intercept_execution(**SAMPLE_ARGS)
        assert result1["bom_hash"] == result2["bom_hash"]

    def test_storage_writes_file(self, sdk, tmp_storage_dir):
        result = sdk.intercept_execution(**SAMPLE_ARGS)
        assert result["storage_uri"].startswith("file://")

        file_path = result["storage_uri"].replace("file://", "")
        assert os.path.exists(file_path)

        with open(file_path) as f:
            stored = json.load(f)
        assert stored["agent_id"] == "agent-test-001"
        assert stored["tool_name"] == "read_file"

    def test_optional_fields(self, sdk):
        result = sdk.intercept_execution(
            **SAMPLE_ARGS,
            system_prompt="You are a helpful assistant.",
            retrieved_chunks=[
                {"id": "chunk-1", "content": "Relevant context"},
            ],
        )
        assert len(result["bom_hash"]) == 64

    def test_stored_payload_contains_optional_fields(self, sdk, tmp_storage_dir):
        result = sdk.intercept_execution(
            **SAMPLE_ARGS,
            system_prompt="Test prompt",
            retrieved_chunks=[{"id": "c1", "content": "data"}],
        )
        file_path = result["storage_uri"].replace("file://", "")
        with open(file_path) as f:
            stored = json.load(f)
        assert stored["system_prompt"] == "Test prompt"
        assert len(stored["retrieved_chunks"]) == 1


class TestGolemSDKWitnessMode:
    """Tests for witness mode with COSE_Sign1 signing."""

    def test_intercept_returns_cose_payload(self, witness_sdk):
        result = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        assert "bom_hash" in result
        assert "cose_payload" in result
        assert "cbor_payload" in result
        # No storage_uri or signature in witness mode
        assert "storage_uri" not in result
        assert "signature" not in result

    def test_cose_payload_is_valid_cose_sign1(self, witness_sdk):
        """COSE bytes must decode as a valid COSE_Sign1 message."""
        result = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        cose_bytes = result["cose_payload"]
        assert isinstance(cose_bytes, bytes)
        assert len(cose_bytes) > 0

        # COSE_Sign1 = CBOR Tag(18) [protected, unprotected, payload, signature]
        msg = cbor2.loads(cose_bytes)
        assert msg.tag == 18
        assert len(msg.value) == 4
        assert msg.value[2] is not None  # payload

    def test_cose_inner_payload_matches_cbor_bom(self, witness_sdk):
        """Inner COSE payload must be the CBOR BOM bytes."""
        result = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        cose_bytes = result["cose_payload"]
        cbor_payload = result["cbor_payload"]

        msg = cbor2.loads(cose_bytes)
        assert msg.value[2] == cbor_payload

    def test_bom_hash_matches_sha256_of_cbor(self, witness_sdk):
        """bom_hash must be SHA-256 of the inner CBOR payload."""
        result = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        expected_hash = hashlib.sha256(result["cbor_payload"]).hexdigest()
        assert result["bom_hash"] == expected_hash

    def test_cbor_payload_decodes_to_bom(self, witness_sdk):
        """CBOR payload must decode to the BOM dict."""
        result = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        bom = cbor2.loads(result["cbor_payload"])
        assert bom["agent_id"] == "agent-test-001"
        assert bom["mcp_server_id"] == "mcp-server-001"
        assert bom["tool_name"] == "read_file"
        assert bom["tool_arguments"] == {"path": "/etc/hosts"}

    def test_cose_protected_headers(self, witness_sdk):
        """COSE protected headers must contain alg=ES256 and a kid."""
        result = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        msg = cbor2.loads(result["cose_payload"])

        # Protected headers are a CBOR-encoded map (bstr) at index 0.
        phdr = cbor2.loads(msg.value[0])
        # COSE header labels (RFC 9052 §3.1): 1=alg, 4=kid. ES256 = -7.
        assert phdr[1] == -7
        kid = phdr[4]
        assert isinstance(kid, bytes)
        # kid is hex SHA-256 (64 chars)
        assert len(kid.decode("utf-8")) == 64

    def test_key_id_matches_public_key_fingerprint(self, witness_sdk):
        """key_id should be SHA-256 of the DER public key."""
        pub_der = witness_sdk.public_key_der
        expected = hashlib.sha256(pub_der).hexdigest()
        assert witness_sdk.key_id == expected

    def test_different_inputs_produce_different_cose(self, witness_sdk):
        """Different BOM inputs must produce different COSE envelopes."""
        result1 = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        result2 = witness_sdk.intercept_execution(
            agent_id="agent-test-002",
            mcp_server_id="mcp-server-002",
            tool_name="write_file",
            tool_arguments={"path": "/tmp/out", "content": "hello"},
        )
        assert result1["bom_hash"] != result2["bom_hash"]
        assert result1["cose_payload"] != result2["cose_payload"]

    def test_consistent_hashes_for_same_input(self, witness_sdk):
        """Same BOM input must produce the same hash (COSE envelope differs due to nonce)."""
        result1 = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        result2 = witness_sdk.intercept_execution(**SAMPLE_ARGS)
        assert result1["bom_hash"] == result2["bom_hash"]

    def test_optional_fields_in_cose(self, witness_sdk):
        """Optional BOM fields should be included in the COSE payload."""
        result = witness_sdk.intercept_execution(
            **SAMPLE_ARGS,
            system_prompt="Test prompt",
            retrieved_chunks=[{"id": "c1", "content": "data"}],
        )
        bom = cbor2.loads(result["cbor_payload"])
        assert bom["system_prompt"] == "Test prompt"
        assert len(bom["retrieved_chunks"]) == 1


class TestSDKLabelsParameter:
    """Verify the new `labels=` constructor parameter that lets the
    SDK pass workspace / branch / env labels through to the witness."""

    def test_valid_labels_stored(self):
        sdk = GolemSDK(
            witness_url="http://localhost:39393",
            labels={"branch": "feat/foo", "env": "dev", "workspace": "alice"},
        )
        assert sdk._labels == {"branch": "feat/foo", "env": "dev", "workspace": "alice"}

    def test_unknown_key_dropped(self):
        # `cwd` is not in the allow-list — should be silently dropped.
        sdk = GolemSDK(
            witness_url="http://localhost:39393",
            labels={"branch": "main", "cwd": "/should/be/dropped"},
        )
        assert sdk._labels == {"branch": "main"}
        assert "cwd" not in sdk._labels

    def test_malformed_value_dropped(self):
        # Shell metachars must be rejected — the witness re-validates
        # but the SDK does it first for fast failure.
        sdk = GolemSDK(
            witness_url="http://localhost:39393",
            labels={"branch": "main", "purpose": "rm -rf /; echo pwned"},
        )
        assert sdk._labels == {"branch": "main"}

    def test_no_labels_means_empty_dict(self):
        sdk = GolemSDK(witness_url="http://localhost:39393")
        assert sdk._labels == {}

    def test_non_string_value_dropped(self):
        sdk = GolemSDK(
            witness_url="http://localhost:39393",
            labels={"branch": "main", "env": 42},  # type: ignore
        )
        assert sdk._labels == {"branch": "main"}
