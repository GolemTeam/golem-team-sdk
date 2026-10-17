"""
COSE_Sign1 signer for the Golem SDK.

Wraps CBOR-canonical BOM bytes in a COSE_Sign1 envelope (RFC 9052) using
ECDSA P-256 with SHA-256 (ES256). Protected headers carry:

    alg: ES256 (-7)
    kid: SHA-256 hex fingerprint of the DER public key

This matches the Go cosesign package used by the Witness (and the
TypeScript SDK's cose.ts), so the Witness can verify SDK-produced COSE
envelopes with the same verification path.

Implemented directly with ``cryptography`` + ``cbor2`` — no COSE
framework dependency. (The previous pycose-based implementation could not
decode its own output under cbor2 >= 6, which returns immutable frozendict
maps that pycose 1.1.0 rejects.)
"""

import hashlib
from typing import Tuple

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

# COSE header labels (RFC 9052 §3.1)
_COSE_HEADER_ALG = 1
_COSE_HEADER_KID = 4

# COSE algorithm: ES256 = ECDSA w/ SHA-256 on P-256
_COSE_ALG_ES256 = -7

# COSE_Sign1 CBOR tag
_COSE_SIGN1_TAG = 18

# P-256 coordinate size in bytes
_P256_COORD_SIZE = 32


class COSESigner:
    """ECDSA P-256 COSE_Sign1 signer.

    Generates an ephemeral P-256 key and produces COSE_Sign1 envelopes
    compatible with the Go Witness ``cosesign`` package.
    """

    def __init__(self, private_key: ec.EllipticCurvePrivateKey = None):
        """Create a signer, optionally with an existing private key.

        Args:
            private_key: An existing EC P-256 private key. If None, a new
                         ephemeral key is generated.
        """
        if private_key is not None:
            self._private_key = private_key
        else:
            self._private_key = ec.generate_private_key(ec.SECP256R1())

        # Compute key ID = SHA-256(DER public key) in hex, matching Go's cosesign
        pub_der = self._private_key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo,
        )
        self._key_id = hashlib.sha256(pub_der).hexdigest()

    @property
    def key_id(self) -> str:
        """Hex-encoded SHA-256 fingerprint of the DER public key."""
        return self._key_id

    @property
    def public_key_der(self) -> bytes:
        """Export the public key in DER (SPKI) format."""
        return self._private_key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo,
        )

    def _protected_headers(self) -> bytes:
        """CBOR-encoded protected headers: {1: -7, 4: kid_bytes}."""
        return cbor2.dumps(
            {
                _COSE_HEADER_ALG: _COSE_ALG_ES256,
                _COSE_HEADER_KID: self._key_id.encode("utf-8"),
            },
            canonical=True,
        )

    def sign(self, cbor_payload: bytes) -> bytes:
        """Wrap CBOR-canonical BOM bytes in a COSE_Sign1 envelope.

        Follows RFC 9052 §4.4:
          1. Build Sig_structure = ["Signature1", protected, external_aad, payload]
          2. Sign SHA-256(Sig_structure) with ECDSA P-256
          3. Convert the signature from ASN.1 DER to raw r||s (64 bytes)
          4. Wrap as CBOR Tag(18) [protected, unprotected, payload, signature]

        Args:
            cbor_payload: CBOR-canonical BOM bytes (the signing/hashing input).

        Returns:
            Complete COSE_Sign1 CBOR bytes.
        """
        protected = self._protected_headers()

        # Sig_structure per RFC 9052 §4.4. external_aad is empty.
        sig_structure = cbor2.dumps(
            ["Signature1", protected, b"", cbor_payload],
            canonical=True,
        )

        # ECDSA over SHA-256 produces an ASN.1 DER signature.
        der_signature = self._private_key.sign(
            sig_structure, ec.ECDSA(hashes.SHA256())
        )
        raw_signature = self._der_to_raw(der_signature)

        # COSE_Sign1 = Tag(18) [protected, unprotected, payload, signature]
        cose_array = [protected, {}, cbor_payload, raw_signature]
        return cbor2.dumps(cbor2.CBORTag(_COSE_SIGN1_TAG, cose_array))

    @staticmethod
    def _der_to_raw(der_sig: bytes) -> bytes:
        """Convert an ASN.1 DER ECDSA signature to raw r||s (IEEE P1363).

        go-cose expects raw r||s (64 bytes) for ES256, not ASN.1 DER.
        """
        r, s = asym_utils.decode_dss_signature(der_sig)
        return r.to_bytes(_P256_COORD_SIZE, "big") + s.to_bytes(
            _P256_COORD_SIZE, "big"
        )

    def sign_and_hash(self, cbor_payload: bytes) -> Tuple[bytes, str]:
        """Sign CBOR payload and compute its SHA-256 hash.

        Returns:
            (cose_bytes, bom_hash) where bom_hash is the hex SHA-256 of
            cbor_payload.
        """
        bom_hash = hashlib.sha256(cbor_payload).hexdigest()
        cose_bytes = self.sign(cbor_payload)
        return cose_bytes, bom_hash
