"""GolemBridge: Fire-and-forget HTTP client for the Golem Ingest API."""

import base64
import json
import logging
import threading
import urllib.request
import urllib.error

logger = logging.getLogger("golem_sdk.bridge")


class GolemBridge:
    """Sends signed BOM entries to the Golem Ingest API.

    All POSTs are fire-and-forget — they run in daemon threads and never
    block the caller. Errors are logged, never raised.
    """

    def __init__(self, ingest_url: str = "http://localhost:8090"):
        self.ingest_url = ingest_url.rstrip("/")
        self._endpoint = f"{self.ingest_url}/api/v1/ledger"
        self._timeout = 5  # seconds

    def record(
        self,
        agent_id: str,
        mcp_server_id: str,
        tool_name: str,
        bom_hash: str,
        cbor_payload: bytes,
        signature: bytes,
        storage_uri: str,
    ) -> None:
        """Post a BOM entry to the ingest API in a background thread."""
        t = threading.Thread(
            target=self._post,
            args=(agent_id, mcp_server_id, tool_name, bom_hash,
                  cbor_payload, signature, storage_uri),
            daemon=True,
        )
        t.start()

    def _post(
        self,
        agent_id: str,
        mcp_server_id: str,
        tool_name: str,
        bom_hash: str,
        cbor_payload: bytes,
        signature: bytes,
        storage_uri: str,
    ) -> None:
        """Synchronous POST — called from daemon thread."""
        try:
            body = json.dumps({
                "agent_id": agent_id,
                "mcp_server_id": mcp_server_id,
                "tool_name": tool_name,
                "bom_hash": bom_hash,
                "cbor_payload": base64.b64encode(cbor_payload).decode("ascii"),
                "signature": base64.b64encode(signature).decode("ascii"),
                "storage_uri": storage_uri,
            }).encode("utf-8")

            req = urllib.request.Request(
                self._endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 201:
                    logger.debug(
                        "BOM recorded in ingest API: %s (tool=%s)",
                        bom_hash[:16], tool_name,
                    )
                else:
                    logger.warning(
                        "Ingest API returned %d for tool=%s",
                        resp.status, tool_name,
                    )
        except urllib.error.URLError as e:
            logger.warning("Failed to POST to ingest API: %s", e)
        except Exception as e:
            logger.warning("Unexpected error posting to ingest API: %s", e)
