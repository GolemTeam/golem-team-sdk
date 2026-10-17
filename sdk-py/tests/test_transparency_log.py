"""
Transparency Log Integrity Tests

Pure Python tests that simulate the append-only hash chain WITHOUT requiring
a running PostgreSQL instance. Verifies both positive integrity and negative
tamper detection.
"""
import hashlib
import time
import cbor2
import pytest


GENESIS_HASH = hashlib.sha256(b"").hexdigest()


def make_cbor_payload(agent_id: str, tool_name: str, seq: int) -> bytes:
    """Create a deterministic CBOR payload for testing."""
    return cbor2.dumps({
        "agent_id": agent_id,
        "mcp_server_id": f"srv-{seq}",
        "tool_name": tool_name,
        "tool_arguments": {"seq": seq},
    })


def compute_entry_hash(cbor_payload: bytes, previous_hash: str) -> str:
    """SHA-256(cbor_payload || previous_hash)"""
    data = cbor_payload + previous_hash.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_chain(n: int) -> list:
    """Build a valid chain of n entries."""
    chain = []
    prev_hash = GENESIS_HASH

    for i in range(n):
        payload = make_cbor_payload(f"agent-{i}", f"tool-{i}", i)
        entry_hash = compute_entry_hash(payload, prev_hash)
        chain.append({
            "sequence_id": i + 1,
            "entry_hash": entry_hash,
            "previous_hash": prev_hash,
            "payload_cbor": payload,
        })
        prev_hash = entry_hash

    return chain


def verify_chain(chain: list) -> list:
    """Walk the chain and return a list of error strings. Empty = intact."""
    errors = []
    for i, entry in enumerate(chain):
        # Verify hash integrity
        recomputed = compute_entry_hash(entry["payload_cbor"], entry["previous_hash"])
        if recomputed != entry["entry_hash"]:
            errors.append(f"TAMPER at seq={entry['sequence_id']}: hash mismatch")

        # Verify chain link
        if i == 0:
            if entry["previous_hash"] != GENESIS_HASH:
                errors.append(f"GENESIS at seq={entry['sequence_id']}: expected genesis hash")
        else:
            if entry["previous_hash"] != chain[i - 1]["entry_hash"]:
                errors.append(f"BREAK at seq={entry['sequence_id']}: previous_hash mismatch")

    return errors


class TestHashChainIntegrity:
    def test_single_entry_hash_integrity(self):
        """A single entry's hash should match recomputation."""
        payload = make_cbor_payload("agent-1", "read_file", 1)
        prev_hash = GENESIS_HASH
        entry_hash = compute_entry_hash(payload, prev_hash)

        # Recompute independently
        recomputed = hashlib.sha256(payload + prev_hash.encode("utf-8")).hexdigest()
        assert entry_hash == recomputed

    def test_chain_links_correctly(self):
        """A chain of 5 entries should have valid forward links."""
        chain = build_chain(5)
        errors = verify_chain(chain)
        assert errors == [], f"Expected no errors, got: {errors}"

        # Verify each entry's previous_hash matches the prior entry's entry_hash
        for i in range(1, len(chain)):
            assert chain[i]["previous_hash"] == chain[i - 1]["entry_hash"]

    def test_chain_genesis(self):
        """The first entry should have previous_hash = SHA-256('')."""
        chain = build_chain(1)
        assert chain[0]["previous_hash"] == GENESIS_HASH

    def test_tamper_detection_payload_modified(self):
        """Modifying an entry's CBOR payload should break hash verification."""
        chain = build_chain(5)

        # Tamper: modify entry 3's payload
        chain[2]["payload_cbor"] = make_cbor_payload("EVIL-AGENT", "evil_tool", 999)

        errors = verify_chain(chain)
        assert len(errors) >= 1
        assert any("TAMPER" in e and "seq=3" in e for e in errors)

    def test_tamper_detection_entry_deleted(self):
        """Deleting an entry from the middle should break the chain."""
        chain = build_chain(5)

        # Remove entry 3 (index 2)
        del chain[2]

        errors = verify_chain(chain)
        assert len(errors) >= 1
        # Entry 4 (now at index 2) should have a broken chain link
        assert any("BREAK" in e for e in errors)

    def test_tamper_detection_entry_reordered(self):
        """Swapping two entries should break the chain."""
        chain = build_chain(5)

        # Swap entries 2 and 3
        chain[1], chain[2] = chain[2], chain[1]

        errors = verify_chain(chain)
        assert len(errors) >= 1

    def test_tamper_detection_hash_rewritten(self):
        """
        Even if an attacker rewrites both the payload AND the entry_hash,
        the NEXT entry's previous_hash will no longer match.
        """
        chain = build_chain(5)

        # Attacker modifies entry 3 and covers their tracks
        new_payload = make_cbor_payload("EVIL-AGENT", "evil_tool", 999)
        new_hash = compute_entry_hash(new_payload, chain[2]["previous_hash"])
        chain[2]["payload_cbor"] = new_payload
        chain[2]["entry_hash"] = new_hash

        errors = verify_chain(chain)
        # Entry 3 itself passes, but entry 4's previous_hash still points to the old hash
        assert len(errors) >= 1
        assert any("BREAK" in e and "seq=4" in e for e in errors)

    def test_large_chain_verification(self):
        """Verify a 100-entry chain completes in under 1 second."""
        chain = build_chain(100)

        start = time.time()
        errors = verify_chain(chain)
        elapsed = time.time() - start

        assert errors == []
        assert elapsed < 1.0, f"Verification took {elapsed:.2f}s, expected < 1.0s"

    def test_deterministic_hashing(self):
        """Same inputs should always produce the same hash."""
        payload = make_cbor_payload("agent-x", "tool-y", 42)
        prev = GENESIS_HASH

        h1 = compute_entry_hash(payload, prev)
        h2 = compute_entry_hash(payload, prev)
        assert h1 == h2

    def test_different_payloads_different_hashes(self):
        """Different payloads should produce different hashes."""
        prev = GENESIS_HASH
        p1 = make_cbor_payload("agent-1", "tool-a", 1)
        p2 = make_cbor_payload("agent-2", "tool-b", 2)

        h1 = compute_entry_hash(p1, prev)
        h2 = compute_entry_hash(p2, prev)
        assert h1 != h2
