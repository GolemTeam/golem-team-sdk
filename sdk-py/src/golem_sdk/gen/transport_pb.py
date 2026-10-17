"""
Hand-rolled protobuf encoder for golem.v1 transport messages.

Matches the wire format defined in proto/golem/v1/envelope.proto and
proto/golem/v1/transport.proto. We encode directly rather than using
generated stubs to keep the SDK dependency-light (no grpcio-tools or
protoc needed at build time).

Wire types:
  0 = varint (int32, int64, bool)
  2 = length-delimited (string, bytes, embedded message, map entry)
  5 = 32-bit fixed (float)
"""

import struct
from typing import Dict, List, Optional


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned varint."""
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _encode_signed_varint(value: int) -> bytes:
    """Encode a signed varint (two's complement for negative)."""
    if value < 0:
        # Protobuf uses 10-byte two's complement for negative varints
        value = value + (1 << 64)
    return _encode_varint(value)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    """Encode a field tag."""
    return _encode_varint((field_number << 3) | wire_type)


def _encode_bytes_field(field_number: int, data: bytes) -> bytes:
    """Encode a bytes/string field (wire type 2)."""
    if not data:
        return b""
    tag = _encode_tag(field_number, 2)
    length = _encode_varint(len(data))
    return tag + length + data


def _encode_string_field(field_number: int, value: str) -> bytes:
    """Encode a string field (wire type 2)."""
    if not value:
        return b""
    return _encode_bytes_field(field_number, value.encode("utf-8"))


def _encode_int64_field(field_number: int, value: int) -> bytes:
    """Encode an int64 field (wire type 0)."""
    if value == 0:
        return b""
    tag = _encode_tag(field_number, 0)
    return tag + _encode_signed_varint(value)


def _encode_int32_field(field_number: int, value: int) -> bytes:
    """Encode an int32 field (wire type 0)."""
    if value == 0:
        return b""
    tag = _encode_tag(field_number, 0)
    return tag + _encode_signed_varint(value)


def _encode_float_field(field_number: int, value: float) -> bytes:
    """Encode a float field (wire type 5, fixed32)."""
    tag = _encode_tag(field_number, 5)
    return tag + struct.pack("<f", value)


def _encode_map_entry(field_number: int, key: str, value: str) -> bytes:
    """Encode a map<string,string> entry as an embedded message.

    Map entries are: key = field 1 (string), value = field 2 (string),
    wrapped as a length-delimited embedded message.
    """
    inner = _encode_string_field(1, key) + _encode_string_field(2, value)
    return _encode_bytes_field(field_number, inner)


def encode_signed_envelope(
    cose_sign1: bytes,
    bom_hash: str,
    entry_hash: str = "",
) -> bytes:
    """Encode a SignedEnvelope protobuf message.

    Fields (from envelope.proto):
      1: cose_sign1 (bytes)
      2: bom_hash (string)
      3: entry_hash (string)
    """
    buf = b""
    buf += _encode_bytes_field(1, cose_sign1)
    buf += _encode_string_field(2, bom_hash)
    buf += _encode_string_field(3, entry_hash)
    return buf


def encode_ledger_entry(
    cose_sign1: bytes,
    bom_hash: str,
    storage_uri: str = "",
    payload_hash: str = "",
    raw_payload: bytes = b"",
    sequence_number: int = 0,
    previous_hash: str = "",
    session_id: str = "",
    host_id: str = "",
    agent_id: str = "",
    mcp_server_id: str = "",
    tool_name: str = "",
    event_type: str = "",
    agent_pid: int = 0,
    embedding: Optional[List[float]] = None,
    summary_text: str = "",
    metadata: Optional[Dict[str, str]] = None,
    entry_hash: str = "",
    sdk_public_key: bytes = b"",
) -> bytes:
    """Encode a LedgerEntry protobuf message.

    Fields (from transport.proto):
      1: envelope (SignedEnvelope, embedded)
      2: storage_uri (string)
      3: payload_hash (string)
      4: raw_payload (bytes)
      5: sequence_number (int64)
      6: previous_hash (string)
      7: session_id (string)
      8: host_id (string)
      9: agent_id (string)
     10: mcp_server_id (string)
     11: tool_name (string)
     12: event_type (string)
     13: agent_pid (int32)
     14: embedding (repeated float)
     15: summary_text (string)
     16: metadata (map<string,string>)
     17: sdk_public_key (bytes)
    """
    buf = b""

    # Field 1: envelope (embedded SignedEnvelope)
    envelope_bytes = encode_signed_envelope(
        cose_sign1=cose_sign1,
        bom_hash=bom_hash,
        entry_hash=entry_hash,
    )
    buf += _encode_bytes_field(1, envelope_bytes)

    buf += _encode_string_field(2, storage_uri)
    buf += _encode_string_field(3, payload_hash)
    buf += _encode_bytes_field(4, raw_payload)

    # sequence_number: int64, field 5 — must encode even if 0 for ticks (-1)
    if sequence_number != 0:
        buf += _encode_int64_field(5, sequence_number)

    buf += _encode_string_field(6, previous_hash)
    buf += _encode_string_field(7, session_id)
    buf += _encode_string_field(8, host_id)
    buf += _encode_string_field(9, agent_id)
    buf += _encode_string_field(10, mcp_server_id)
    buf += _encode_string_field(11, tool_name)
    buf += _encode_string_field(12, event_type)

    if agent_pid != 0:
        buf += _encode_int32_field(13, agent_pid)

    # Field 14: repeated float (each value is its own tag+fixed32)
    if embedding:
        for val in embedding:
            buf += _encode_float_field(14, val)

    buf += _encode_string_field(15, summary_text)

    # Field 16: map<string,string> entries
    if metadata:
        for k, v in sorted(metadata.items()):
            buf += _encode_map_entry(16, k, v)

    # Field 17: sdk_public_key (DER SPKI) — the witness requires this to
    # verify the COSE_Sign1 signature and rejects entries without it.
    if sdk_public_key:
        buf += _encode_bytes_field(17, sdk_public_key)

    return buf
