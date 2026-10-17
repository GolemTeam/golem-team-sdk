import hashlib
import logging
import re
import warnings

import cbor2
from typing import Optional, List, Dict, Any, Tuple

from golem_sdk.cose_signer import COSESigner
from golem_sdk.redaction import PolicyCache, counts_metadata
from golem_sdk.storage import StorageBackend, LocalDiskStorage

logger = logging.getLogger("golem_sdk")

# Identifiers also travel unsigned beside the BOM, so rewriting them here would
# hide nothing and only break the witness's attribution.
_BOM_IDENTIFIERS = frozenset({"agent_id", "mcp_server_id", "tool_name"})

# Validation constants
MAX_STRING_LENGTH = 1024
MAX_TOOL_ARGS_KEYS = 100
MAX_CHUNKS = 500
_SAFE_ID_PATTERN = re.compile(r"^[\w\-.:/]{1,255}$")


def _validate_string_field(name: str, value: str, max_length: int = MAX_STRING_LENGTH) -> None:
    """Validate a required string field is non-empty and within bounds."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length of {max_length}")


def _validate_id_field(name: str, value: str) -> None:
    """Validate an identifier field (agent_id, server_id, tool_name)."""
    _validate_string_field(name, value, max_length=255)
    if not _SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"{name} contains invalid characters. "
            "Only alphanumeric, hyphens, underscores, dots, colons, and slashes are allowed."
        )


def _validate_sdk_labels(raw: Dict[str, str]) -> Dict[str, str]:
    """Filter caller-supplied labels to the allow-list + format.

    Returns a copy with unknown keys and malformed values dropped. The
    witness re-validates on receipt and silently drops anything that
    leaks through, so this is defense-in-depth (faster failure +
    clearer error to the caller, and avoids putting garbage on the
    wire).
    """
    out: Dict[str, str] = {}
    for k, v in raw.items():
        if k not in _SDK_LABEL_KEYS:
            logger.debug("dropping unknown SDK label key: %s", k)
            continue
        if not isinstance(v, str) or not _SDK_LABEL_VALUE_RE.match(v):
            logger.debug("dropping malformed SDK label %s", k)
            continue
        out[k] = v
    return out


# Label keys the witness will accept and propagate into session metadata.
# The witness re-validates and silently drops anything outside this list,
# so adding a key here is necessary but not sufficient — keep the two
# sides in sync (the Golem agent's SDK handler).
_SDK_LABEL_KEYS = frozenset({
    "branch",     # git branch
    "worktree",   # worktree / workspace directory name
    "workspace",  # explicit friendly name for the run
    "env",        # "dev", "staging", "prod"
    "project",    # repo / project identifier
    "purpose",    # free-form short label
})
_SDK_LABEL_VALUE_RE = re.compile(r"^[\w\-./@: ]{1,255}$")


class GolemSDK:
    def __init__(
        self,
        witness_url: Optional[str] = None,
        storage: Optional[StorageBackend] = None,
        bridge=None,
        session_id: str = "",
        host_id: str = "",
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize GolemSDK.

        **Witness mode** (recommended): SDK signs with COSE_Sign1 and POSTs
        protobuf to the Golem Witness on localhost. The Witness forwards to
        the Relay, which handles BYOB storage, embedding, and Ingest forwarding.

            sdk = GolemSDK(witness_url="http://localhost:39393")

        **Legacy mode** (deprecated): SDK does its own BYOB storage and optionally
        bridges directly to the Ingest API. No embeddings. BYOB storage has moved
        to the Relay --- use witness mode instead.

            sdk = GolemSDK(storage=LocalDiskStorage(), bridge=GolemBridge())

        Args:
            witness_url: Golem Witness URL (enables witness mode). Default: None.
                         Set to "http://localhost:39393" for standard deployments.
            storage: Deprecated. BYOB storage backend for legacy mode.
                     BYOB is now handled by the Relay. Use witness_url instead.
            bridge: Deprecated. GolemBridge for legacy mode direct-to-ingest forwarding.
            session_id: Optional session identifier (passed to witness/ingest)
            host_id: Optional host identifier (passed to witness/ingest)
            labels: Optional friendly-name labels to attach to every event from
                this SDK instance. Useful when the witness can't infer them
                from the environment (sidecar/docker deployments where the
                witness doesn't see the caller's filesystem). Allowed keys:
                branch, worktree, workspace, env, project, purpose. Values
                must match `^[\\w\\-./@: ]{1,255}$`. Unknown keys / invalid
                values are dropped silently. Example:

                    sdk = GolemSDK(
                        witness_url="http://localhost:39393",
                        labels={"branch": "feat/foo", "env": "dev"},
                    )
        """
        self._session_id = session_id
        self._host_id = host_id
        self._labels = _validate_sdk_labels(labels or {})

        # COSE_Sign1 signer with ephemeral P-256 key
        self._cose_signer = COSESigner()

        # Retain legacy ECDSA key for backward-compatible legacy mode
        self._identity_private_key = None
        self._redaction: Optional[PolicyCache] = None

        if witness_url is not None:
            # Witness mode: thin SDK -> Witness sidecar
            self._mode = "witness"
            from golem_sdk.witness_client import WitnessClient
            self._witness = WitnessClient(witness_url=witness_url)
            self._redaction = PolicyCache(self._witness.fetch_redaction_policy)
            self.storage = None
            self._bridge = None
        elif storage is not None or bridge is not None:
            # Legacy mode: fat SDK with local storage + direct bridge
            self._mode = "legacy"
            self.storage = storage or LocalDiskStorage()
            self._bridge = bridge
            self._witness = None
            self._init_legacy_key()
            warnings.warn(
                "Legacy mode (storage/bridge) is deprecated. "
                "Use GolemSDK(witness_url='http://localhost:39393') instead. "
                "The Witness handles storage, embedding, and forwarding.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            # Default: legacy mode with local storage (backward compatible)
            self._mode = "legacy"
            self.storage = LocalDiskStorage()
            self._bridge = None
            self._witness = None
            self._init_legacy_key()

    def _init_legacy_key(self):
        """Initialize legacy ECDSA key for backward-compatible legacy mode."""
        import ecdsa
        self._identity_private_key = ecdsa.SigningKey.generate(curve=ecdsa.NIST256p)

    @property
    def public_key_der(self) -> bytes:
        """Export the SDK's ephemeral public key in DER (SPKI) format.

        Used by the Witness to verify signatures.
        """
        return self._cose_signer.public_key_der

    @property
    def key_id(self) -> str:
        """Hex-encoded SHA-256 fingerprint of the signing public key."""
        return self._cose_signer.key_id

    def check_execution(
        self,
        tool_name: str,
        command: str = "",
        agent_id: str = "",
        session_id: str = "",
        host_id: str = "",
    ):
        """Ask the witness whether this call may run. BLOCKS.

        Returns a ``PolicyDecision``. In non-witness mode there is nothing to
        ask, so the answer is allow with ``reachable=False`` — the same shape a
        witness outage returns, because to a caller they are the same
        situation: nobody said no.
        """
        from golem_sdk.witness_client import PolicyDecision

        if self._witness is None:
            return PolicyDecision(allowed=True, action="allow", reason="", reachable=False)
        return self._witness.check_execution(
            tool_name=tool_name,
            command=command,
            agent_id=agent_id,
            session_id=session_id,
            host_id=host_id,
        )

    def _redact(
        self, bom: Dict[str, Any], labels: Dict[str, str]
    ) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
        """Apply this machine's redaction policy to everything that will be signed.

        Never raises: with no policy, or if redaction itself fails, the event goes
        out as written and its metadata says so.
        """
        policy = self._redaction.current() if self._redaction is not None else None
        if policy is None:
            return bom, labels, {"redaction": "policy_unavailable"}
        if not policy.enabled:
            return bom, labels, {}
        counts: Dict[str, int] = {}
        try:
            red_bom = {
                k: v if k in _BOM_IDENTIFIERS else policy.redact_value(v, counts)
                for k, v in bom.items()
            }
            red_labels = {k: policy.redact_value(v, counts) for k, v in labels.items()}
        except Exception as e:
            logger.warning("redaction failed (%s); event sent unredacted", type(e).__name__)
            return bom, labels, {"redaction": "failed"}
        return red_bom, red_labels, ({"redaction": counts_metadata(counts)} if counts else {})

    def intercept_execution(
        self,
        agent_id: str,
        mcp_server_id: str,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        system_prompt: Optional[str] = None,
        retrieved_chunks: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Intercepts an agent's execution intent, hashes the state, signs it,
        and routes it to the appropriate destination.

        In witness mode: COSE_Sign1 signs and POSTs protobuf to the Witness.
        In legacy mode: signs, stores to BYOB, optionally bridges to Ingest.
        """
        # Input validation
        _validate_id_field("agent_id", agent_id)
        _validate_id_field("mcp_server_id", mcp_server_id)
        _validate_id_field("tool_name", tool_name)

        if not isinstance(tool_arguments, dict):
            raise TypeError("tool_arguments must be a dictionary")
        if len(tool_arguments) > MAX_TOOL_ARGS_KEYS:
            raise ValueError(f"tool_arguments exceeds maximum of {MAX_TOOL_ARGS_KEYS} keys")

        if system_prompt is not None:
            _validate_string_field("system_prompt", system_prompt, max_length=100_000)

        if retrieved_chunks is not None:
            if not isinstance(retrieved_chunks, list):
                raise TypeError("retrieved_chunks must be a list")
            if len(retrieved_chunks) > MAX_CHUNKS:
                raise ValueError(f"retrieved_chunks exceeds maximum of {MAX_CHUNKS} entries")

        # Build BOM payload
        bom = {
            "agent_id": agent_id,
            "mcp_server_id": mcp_server_id,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
        }

        if system_prompt:
            bom["system_prompt"] = system_prompt
        if retrieved_chunks:
            bom["retrieved_chunks"] = retrieved_chunks

        labels = self._labels
        redaction_meta: Dict[str, str] = {}
        if self._mode == "witness":
            bom, labels, redaction_meta = self._redact(bom, labels)

        # 1. Serialize deterministic payload (CBOR canonical)
        cbor_payload = cbor2.dumps(bom, canonical=True)

        # 2. Cryptographic Hash (SHA-256)
        bom_hash = hashlib.sha256(cbor_payload).hexdigest()

        # 3. Route based on mode
        if self._mode == "witness":
            # COSE_Sign1 signing
            cose_bytes = self._cose_signer.sign(cbor_payload)

            # POST protobuf to Witness
            self._witness.record(
                agent_id=agent_id,
                mcp_server_id=mcp_server_id,
                tool_name=tool_name,
                bom_hash=bom_hash,
                cbor_payload=cbor_payload,
                cose_payload=cose_bytes,
                public_key_der=self.public_key_der,
                session_id=self._session_id,
                host_id=self._host_id,
                labels=labels,
                metadata=redaction_meta,
            )

            return {
                "bom_hash": bom_hash,
                "cose_payload": cose_bytes,
                "cbor_payload": cbor_payload,
            }
        else:
            # Legacy mode: ECDSA sign + store locally + optional bridge to ingest
            signature = self._identity_private_key.sign_digest(bytes.fromhex(bom_hash))
            storage_uri = self.storage.store_raw_payload(bom_hash, bom)

            result = {
                "bom_hash": bom_hash,
                "storage_uri": storage_uri,
                "signature": signature,
                "cbor_payload": cbor_payload,
            }

            if self._bridge is not None:
                self._bridge.record(
                    agent_id=agent_id,
                    mcp_server_id=mcp_server_id,
                    tool_name=tool_name,
                    bom_hash=bom_hash,
                    cbor_payload=cbor_payload,
                    signature=signature,
                    storage_uri=storage_uri,
                )

            return result
