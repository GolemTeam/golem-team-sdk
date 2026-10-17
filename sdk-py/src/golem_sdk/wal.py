"""On-disk durable-delivery buffer for the Golem SDK.

The witness runs a WAL for its own witness->cloud hop; this is the SDK-side
equivalent for the SDK->witness hop. Because an SDK process is typically
short-lived (fire an event, exit), the buffer is on disk so entries survive
until a later run can replay them — an in-memory queue would die with the
process.

Each record is the raw protobuf LedgerEntry body. Replay is idempotent: the
witness (and Ingest behind it) dedups by entry_hash, so re-POSTing a record
whose original delivery actually landed is a no-op.
"""

import logging
import os
import tempfile
import threading
from typing import Callable

logger = logging.getLogger("golem_sdk.wal")

_WAL_EXT = ".wal"


class SDKBuffer:
    """A bounded, filesystem-backed queue of undelivered LedgerEntry bodies.

    Thread-safe. Enforces ``max_records`` by dropping the OLDEST record with a
    warning — bounded disk use, never silent unbounded growth.
    """

    def __init__(self, directory: str, max_records: int = 10000):
        self._dir = directory
        self._max = max_records if max_records > 0 else 10000
        self._lock = threading.Lock()
        self._seq = 0
        os.makedirs(self._dir, mode=0o700, exist_ok=True)

    def _files(self) -> list:
        """Record files, oldest-first (names sort chronologically)."""
        try:
            names = [n for n in os.listdir(self._dir) if n.endswith(_WAL_EXT)]
        except OSError:
            return []
        names.sort()
        return names

    def pending(self) -> int:
        with self._lock:
            return len(self._files())

    def append(self, body: bytes) -> None:
        """Durably buffer one LedgerEntry body."""
        with self._lock:
            self._prune_locked(self._max - 1)
            self._seq += 1
            # Sortable, unique name: monotonic nanoseconds + per-instance seq.
            name = f"{_now_ns():020d}-{self._seq:06d}{_WAL_EXT}"
            final = os.path.join(self._dir, name)
            try:
                fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
                try:
                    os.write(fd, body)
                finally:
                    os.close(fd)
                os.chmod(tmp, 0o600)
                os.replace(tmp, final)
            except OSError as e:
                logger.warning("wal: append failed: %s", type(e).__name__)

    def _prune_locked(self, cap: int) -> None:
        if cap < 0:
            cap = 0
        names = self._files()
        while len(names) > cap:
            oldest = names.pop(0)
            logger.warning("wal: capacity exceeded, dropping oldest buffered entry")
            try:
                os.remove(os.path.join(self._dir, oldest))
            except OSError:
                pass

    def drain(self, post: Callable[[bytes], bool]) -> None:
        """Replay buffered records oldest-first.

        ``post(body)`` returns True when the record is delivered OR permanently
        rejected (delete it), and False for a retryable failure (keep it and
        stop this cycle — the backend is likely still down).
        """
        with self._lock:
            for name in self._files():
                path = os.path.join(self._dir, name)
                try:
                    with open(path, "rb") as f:
                        body = f.read()
                except OSError:
                    continue
                try:
                    done = post(body)
                except Exception as e:  # a failing post must never crash the SDK
                    logger.warning("wal: replay post raised: %s", type(e).__name__)
                    return
                if done:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                else:
                    return  # backend down — retry remaining records next drain


def _now_ns() -> int:
    import time
    return time.monotonic_ns()
