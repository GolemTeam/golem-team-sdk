"""Redaction before signing: the Go engine's vectors, the policy, and the SDK path."""

import hashlib
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cbor2
import pytest

from golem_sdk.redaction import (
    CustomSpec,
    Operator,
    PolicyCache,
    RedactionPolicy,
    detect,
    redact,
    secret_types,
    structured_types,
)
from golem_sdk.sdk import GolemSDK
from golem_sdk.storage import LocalDiskStorage
from tests.test_witness_client import decode_protobuf, parse_ledger_entry

VECTORS = os.path.join(
    os.path.dirname(__file__), "..", "..", "pkg", "redaction", "testdata", "vectors.json"
)
NO_VECTORS = "shared redaction test vectors not found"


def _vectors():
    if not os.path.exists(VECTORS):
        return []
    with open(VECTORS, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not os.path.exists(VECTORS), reason=NO_VECTORS)
@pytest.mark.parametrize("v", _vectors(), ids=lambda v: v["name"])
def test_golden_vector(v):
    op = Operator.from_dict(v["operator"])
    custom = [CustomSpec.from_dict(c) for c in v.get("custom") or []]
    got, counts = redact(v["text"], v["types"], op, custom)
    assert got == v["want"]
    assert [e.type for e in detect(v["text"], v["types"], custom)] == v["found"]
    assert sum(counts.values()) == len(v["found"])


def test_vectors_file_is_not_empty():
    if not os.path.exists(VECTORS):
        pytest.skip(NO_VECTORS)
    assert len(_vectors()) >= 38


def test_type_lists_match_the_go_engine():
    assert structured_types() == [
        "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER",
        "IP_ADDRESS", "IBAN_CODE", "AWS_ACCESS_KEY", "CRYPTO",
    ]
    assert "GOLEM_KEY" in secret_types() and "EMAIL_ADDRESS" not in secret_types()


class TestPolicyParsing:
    def test_full_policy(self):
        p = RedactionPolicy.from_dict({
            "enabled": True,
            "entity_types": ["EMAIL_ADDRESS", "CUSTOM_BADGE", "PERSON"],
            "operator": {"type": "hash", "hash_len": 8, "salt": "s"},
            "custom": [{"EntityType": "CUSTOM_BADGE", "Pattern": r"BDG-\d{6}", "Group": 0, "Score": 0.9, "Shadow": False}],
        })
        assert p.enabled
        assert p.entity_types == ("EMAIL_ADDRESS", "CUSTOM_BADGE", "PERSON")
        assert p.operator == Operator(type="hash", hash_len=8, salt="s")
        assert p.custom == (CustomSpec("CUSTOM_BADGE", r"BDG-\d{6}", 0, 0.9, False),)
        text, counts = p.redact_text("Jane a@b.co BDG-123456")
        assert "a@b.co" not in text and "BDG-123456" not in text
        assert text.startswith("Jane ")
        assert counts == {"EMAIL_ADDRESS": 1, "CUSTOM_BADGE": 1}

    def test_empty_types_mean_every_structured_type(self):
        p = RedactionPolicy.from_dict({"enabled": True, "entity_types": [], "operator": {"type": "replace"}, "custom": []})
        assert p.redact_text("a@b.co 10.0.0.1 ghp_" + "a" * 36)[0] == "<EMAIL_ADDRESS> <IP_ADDRESS> ghp_" + "a" * 36

    def test_missing_and_malformed_fields_default(self):
        p = RedactionPolicy.from_dict({"enabled": "yes", "entity_types": "EMAIL", "operator": None, "custom": {"x": 1}})
        assert not p.enabled and p.entity_types == () and p.operator == Operator() and p.custom == ()

    def test_not_an_object(self):
        with pytest.raises(ValueError):
            RedactionPolicy.from_dict(["enabled"])

    def test_bad_custom_patterns_are_skipped_without_logging_them(self, caplog):
        caplog.set_level(logging.WARNING, logger="golem_sdk.redaction")
        p = RedactionPolicy.from_dict({
            "enabled": True,
            "entity_types": ["CUSTOM_BROKEN", "CUSTOM_OK", "CUSTOM_SHADOW", "GOLEM_KEY"],
            "custom": [
                {"EntityType": "CUSTOM_BROKEN", "Pattern": "(secretpattern", "Score": 0.9},
                {"EntityType": "CUSTOM_OK", "Pattern": "OK-[0-9]+", "Score": 0.9},
                {"EntityType": "CUSTOM_SHADOW", "Pattern": "SH-[0-9]+", "Score": 0.9, "Shadow": True},
                {"EntityType": "GOLEM_KEY", "Pattern": "never", "Score": 0.9},
            ],
        })
        text, counts = p.redact_text("OK-1 SH-2 glk_" + "j" * 30)
        assert counts == {"CUSTOM_OK": 1, "GOLEM_KEY": 1}
        assert "SH-2" in text
        assert "CUSTOM_BROKEN" in caplog.text
        assert "secretpattern" not in caplog.text

    def test_low_score_custom_falls_under_the_threshold(self):
        assert redact("OK-1", ["CUSTOM_LOW"], custom=[CustomSpec("CUSTOM_LOW", "OK-[0-9]", 0, 0.4)]) == ("OK-1", {})

    @pytest.mark.parametrize("pattern,text,want", [
        (r"ID[[:digit:]]{3}", "ID123", "<CUSTOM_X>"),
        (r"[^[:space:]]+@corp\z", "x jo@corp", "x <CUSTOM_X>"),
        (r"\Qa.b\E-[0-9]", "a.b-1 axb-1", "<CUSTOM_X> axb-1"),
        (r"\x{41}{3}", "AAA", "<CUSTOM_X>"),
        (r"\d+", "٣٣٣ 123", "٣٣٣ <CUSTOM_X>"),
    ])
    def test_re2_spellings(self, pattern, text, want):
        assert redact(text, ["CUSTOM_X"], custom=[CustomSpec("CUSTOM_X", pattern, 0, 0.9)])[0] == want


class TestPolicyCache:
    def _clock(self):
        t = [100.0]
        return t, lambda: t[0]

    def test_caches_for_the_ttl(self):
        calls = []
        t, clock = self._clock()
        cache = PolicyCache(lambda: calls.append(1) or {"enabled": True}, ttl=30, clock=clock)
        assert cache.current().enabled
        t[0] += 29
        cache.current()
        assert len(calls) == 1
        t[0] += 2
        cache.current()
        assert len(calls) == 2

    def test_failure_keeps_the_last_policy(self):
        responses = [{"enabled": True, "entity_types": ["EMAIL_ADDRESS"]}]
        t, clock = self._clock()

        def fetch():
            if responses:
                return responses.pop()
            raise ConnectionRefusedError()

        cache = PolicyCache(fetch, ttl=30, clock=clock)
        first = cache.current()
        t[0] += 31
        assert cache.current() is first

    def test_never_fetched_is_none_and_retries_sooner(self):
        calls = []
        t, clock = self._clock()

        def fetch():
            calls.append(1)
            raise TimeoutError()

        cache = PolicyCache(fetch, ttl=30, retry=5, clock=clock)
        assert cache.current() is None
        t[0] += 4
        assert cache.current() is None and len(calls) == 1
        t[0] += 2
        cache.current()
        assert len(calls) == 2


EMAIL = "jane.roe@example.org"
TOKEN = "ghp_" + "Z" * 36


class _Witness(BaseHTTPRequestHandler):
    policy = None
    posts = []

    def do_GET(self):
        if self.path != "/api/v1/redaction-policy" or _Witness.policy is None:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(_Witness.policy).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Witness.posts.append(body)
        self.send_response(201)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def witness():
    _Witness.policy = None
    _Witness.posts = []
    server = HTTPServer(("127.0.0.1", 0), _Witness)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _posted():
    deadline = time.time() + 2
    while time.time() < deadline and not _Witness.posts:
        time.sleep(0.01)
    assert _Witness.posts, "nothing was posted to the witness"
    body = _Witness.posts[0]
    entry = parse_ledger_entry(body)
    metadata = {}
    for raw in decode_protobuf(body).get(16, []):
        kv = decode_protobuf(raw)
        metadata[kv[1][0].decode()] = kv[2][0].decode()
    return entry, metadata


def _intercept(sdk, args):
    return sdk.intercept_execution(
        agent_id="test-agent",
        mcp_server_id="test-server",
        tool_name="send_mail",
        tool_arguments=args,
        system_prompt=f"You act for {EMAIL}.",
        retrieved_chunks=[{"text": f"token {TOKEN}"}],
    )


SECRETS_AND_EMAIL = {"enabled": True, "entity_types": ["EMAIL_ADDRESS", "GITHUB_TOKEN"], "operator": {"type": "replace"}, "custom": []}


def test_redacts_before_signing(witness):
    _Witness.policy = SECRETS_AND_EMAIL
    sdk = GolemSDK(witness_url=witness, labels={"workspace": EMAIL})
    args = {"to": EMAIL, "nested": [{"auth": TOKEN}], EMAIL: 1, "n": 3}
    result = _intercept(sdk, args)

    entry, metadata = _posted()
    cose = cbor2.loads(entry["cose_sign1"])
    signed = cose.value[2]
    for blob in (result["cbor_payload"], entry["raw_payload"], signed, entry["cose_sign1"]):
        assert EMAIL.encode() not in blob
        assert TOKEN.encode() not in blob
    assert hashlib.sha256(signed).hexdigest() == entry["bom_hash"] == result["bom_hash"]

    bom = cbor2.loads(signed)
    assert bom["tool_arguments"] == {
        "to": "<EMAIL_ADDRESS>", "nested": [{"auth": "<GITHUB_TOKEN>"}], "<EMAIL_ADDRESS>": 1, "n": 3,
    }
    assert bom["system_prompt"] == "You act for <EMAIL_ADDRESS>."
    assert bom["retrieved_chunks"] == [{"text": "token <GITHUB_TOKEN>"}]
    assert bom["tool_name"] == "send_mail"

    assert json.loads(metadata["redaction"]) == {"EMAIL_ADDRESS": 4, "GITHUB_TOKEN": 2}
    assert metadata["workspace"] == "<EMAIL_ADDRESS>"
    assert args["to"] == EMAIL, "the caller's arguments were modified"


def test_policy_off_sends_as_written(witness):
    _Witness.policy = {**SECRETS_AND_EMAIL, "enabled": False}
    sdk = GolemSDK(witness_url=witness)
    result = _intercept(sdk, {"to": EMAIL})
    _, metadata = _posted()
    assert EMAIL.encode() in result["cbor_payload"]
    assert "redaction" not in metadata


def test_nothing_found_records_no_counts(witness):
    _Witness.policy = SECRETS_AND_EMAIL
    sdk = GolemSDK(witness_url=witness)
    sdk.intercept_execution(agent_id="a", mcp_server_id="s", tool_name="t", tool_arguments={"q": "hello"})
    _, metadata = _posted()
    assert "redaction" not in metadata


def test_unreachable_policy_is_marked_and_never_raises(witness):
    sdk = GolemSDK(witness_url=witness)
    result = _intercept(sdk, {"to": EMAIL})
    _, metadata = _posted()
    assert metadata["redaction"] == "policy_unavailable"
    assert EMAIL.encode() in result["cbor_payload"]


def test_witness_down_never_raises():
    sdk = GolemSDK(witness_url="http://127.0.0.1:9")
    assert _intercept(sdk, {"to": EMAIL})["bom_hash"]


def test_a_failing_redaction_is_marked(witness, monkeypatch):
    _Witness.policy = SECRETS_AND_EMAIL
    sdk = GolemSDK(witness_url=witness)

    def boom(self, value, counts):
        raise RuntimeError("bug")

    monkeypatch.setattr(RedactionPolicy, "redact_value", boom)
    _intercept(sdk, {"to": EMAIL})
    _, metadata = _posted()
    assert metadata["redaction"] == "failed"


def test_legacy_mode_does_not_fetch_a_policy(tmp_path):
    with pytest.warns(DeprecationWarning):
        sdk = GolemSDK(storage=LocalDiskStorage(base_path=str(tmp_path)))
    assert EMAIL.encode() in _intercept(sdk, {"to": EMAIL})["cbor_payload"]
