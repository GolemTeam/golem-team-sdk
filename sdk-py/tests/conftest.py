import os

# Default the SDK durable-delivery buffer OFF for tests so transport tests don't
# touch ~/.golem/sdk-wal. Tests that exercise the buffer set GOLEM_SDK_WAL_PATH
# to a temp dir themselves (via monkeypatch) before constructing a client.
os.environ.setdefault("GOLEM_SDK_WAL_PATH", "off")
