import os
import json
import tempfile
import shutil
import pytest
from golem_sdk.storage import LocalDiskStorage

# Valid SHA-256 hex strings for testing
HASH_1 = "a" * 64  # aaaa...a (64 chars)
HASH_2 = "b" * 64  # bbbb...b (64 chars)
HASH_3 = "0123456789abcdef" * 4  # 64 chars


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="golem-storage-test-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestLocalDiskStorage:
    def test_creates_base_directory(self, tmp_dir):
        target = os.path.join(tmp_dir, "nested", "golem")
        storage = LocalDiskStorage(base_path=target)
        assert os.path.isdir(target)

    def test_store_returns_file_uri(self, tmp_dir):
        storage = LocalDiskStorage(base_path=tmp_dir)
        uri = storage.store_raw_payload(HASH_1, {"key": "value"})
        assert uri.startswith("file://")
        assert HASH_1 in uri

    def test_store_writes_valid_json(self, tmp_dir):
        storage = LocalDiskStorage(base_path=tmp_dir)
        payload = {
            "agent_id": "agent-1",
            "tool_name": "read_file",
            "tool_arguments": {"path": "/tmp/data"},
        }
        storage.store_raw_payload(HASH_3, payload)

        file_path = os.path.join(tmp_dir, f"{HASH_3}.json")
        assert os.path.exists(file_path)

        with open(file_path) as f:
            contents = json.load(f)
        assert contents["agent_id"] == "agent-1"
        assert contents["tool_name"] == "read_file"

    def test_multiple_payloads_no_collision(self, tmp_dir):
        storage = LocalDiskStorage(base_path=tmp_dir)
        storage.store_raw_payload(HASH_1, {"id": 1})
        storage.store_raw_payload(HASH_2, {"id": 2})

        with open(os.path.join(tmp_dir, f"{HASH_1}.json")) as f:
            assert json.load(f)["id"] == 1
        with open(os.path.join(tmp_dir, f"{HASH_2}.json")) as f:
            assert json.load(f)["id"] == 2

    def test_overwrites_existing_file(self, tmp_dir):
        storage = LocalDiskStorage(base_path=tmp_dir)
        storage.store_raw_payload(HASH_1, {"version": 1})
        storage.store_raw_payload(HASH_1, {"version": 2})

        with open(os.path.join(tmp_dir, f"{HASH_1}.json")) as f:
            assert json.load(f)["version"] == 2

    def test_rejects_invalid_hash_id(self, tmp_dir):
        storage = LocalDiskStorage(base_path=tmp_dir)
        with pytest.raises(ValueError, match="Invalid hash_id"):
            storage.store_raw_payload("../../../etc/passwd", {"key": "value"})

    def test_rejects_short_hash_id(self, tmp_dir):
        storage = LocalDiskStorage(base_path=tmp_dir)
        with pytest.raises(ValueError, match="Invalid hash_id"):
            storage.store_raw_payload("abc123", {"key": "value"})
