/**
 * Hand-rolled protobuf encoder for golem.v1 transport messages.
 *
 * Matches the wire format defined in proto/golem/v1/envelope.proto and
 * proto/golem/v1/transport.proto — and mirrors the Python SDK's
 * golem_sdk/gen/transport_pb.py field-for-field. We encode directly rather
 * than pulling in protobufjs/ts-proto so the SDK stays dependency-light.
 *
 * The Golem Witness `/api/v1/sdk/tool-use` endpoint accepts ONLY protobuf
 * `LedgerEntry` messages (Content-Type: application/x-protobuf); it
 * `proto.Unmarshal`s the body and rejects anything else.
 *
 * Wire types:
 *   0 = varint (int32, int64, bool)
 *   2 = length-delimited (string, bytes, embedded message, map entry)
 *   5 = 32-bit fixed (float)
 */

/** Encode an unsigned varint. */
function encodeVarint(value: number): Buffer {
  const parts: number[] = [];
  // Use BigInt-free path: values here are small (lengths, tags, field nums).
  let v = value;
  while (v > 0x7f) {
    parts.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  parts.push(v & 0x7f);
  return Buffer.from(parts);
}

/** Encode a field tag (field number + wire type). */
function encodeTag(fieldNumber: number, wireType: number): Buffer {
  return encodeVarint((fieldNumber << 3) | wireType);
}

/**
 * Encode a bytes/string field (wire type 2). Empty payloads are omitted
 * entirely (proto3 default semantics), matching the Python encoder.
 */
function encodeBytesField(fieldNumber: number, data: Buffer): Buffer {
  if (!data || data.length === 0) {
    return Buffer.alloc(0);
  }
  return Buffer.concat([encodeTag(fieldNumber, 2), encodeVarint(data.length), data]);
}

/** Encode a string field (wire type 2). Empty strings are omitted. */
function encodeStringField(fieldNumber: number, value: string): Buffer {
  if (!value) {
    return Buffer.alloc(0);
  }
  return encodeBytesField(fieldNumber, Buffer.from(value, 'utf-8'));
}

/**
 * Encode a map<string,string> entry as an embedded message.
 * Map entries are: key = field 1 (string), value = field 2 (string),
 * wrapped as a length-delimited embedded message.
 */
function encodeMapEntry(fieldNumber: number, key: string, value: string): Buffer {
  const inner = Buffer.concat([encodeStringField(1, key), encodeStringField(2, value)]);
  return encodeBytesField(fieldNumber, inner);
}

/**
 * Encode a SignedEnvelope protobuf message.
 * Fields (from envelope.proto):
 *   1: cose_sign1 (bytes)
 *   2: bom_hash (string)
 *   3: entry_hash (string)
 */
export function encodeSignedEnvelope(
  coseSign1: Buffer,
  bomHash: string,
  entryHash = '',
): Buffer {
  return Buffer.concat([
    encodeBytesField(1, coseSign1),
    encodeStringField(2, bomHash),
    encodeStringField(3, entryHash),
  ]);
}

export interface LedgerEntryFields {
  coseSign1: Buffer;
  bomHash: string;
  rawPayload?: Buffer;
  payloadHash?: string;
  sessionId?: string;
  hostId?: string;
  agentId?: string;
  mcpServerId?: string;
  toolName?: string;
  eventType?: string;
  metadata?: Record<string, string>;
  entryHash?: string;
  /** DER (SPKI) public key — the witness requires this to verify the COSE signature. */
  sdkPublicKey?: Buffer;
}

/**
 * Encode a LedgerEntry protobuf message.
 * Fields (from transport.proto):
 *   1: envelope (SignedEnvelope, embedded)
 *   2: storage_uri (string)
 *   3: payload_hash (string)
 *   4: raw_payload (bytes)
 *   5: sequence_number (int64)
 *   6: previous_hash (string)
 *   7: session_id (string)
 *   8: host_id (string)
 *   9: agent_id (string)
 *  10: mcp_server_id (string)
 *  11: tool_name (string)
 *  12: event_type (string)
 *  13: agent_pid (int32)
 *  14: embedding (repeated float)
 *  15: summary_text (string)
 *  16: metadata (map<string,string>)
 *  17: sdk_public_key (bytes)
 */
export function encodeLedgerEntry(f: LedgerEntryFields): Buffer {
  const chunks: Buffer[] = [];

  // Field 1: envelope (embedded SignedEnvelope)
  const envelope = encodeSignedEnvelope(f.coseSign1, f.bomHash, f.entryHash || '');
  chunks.push(encodeBytesField(1, envelope));

  chunks.push(encodeStringField(3, f.payloadHash || ''));
  chunks.push(encodeBytesField(4, f.rawPayload || Buffer.alloc(0)));
  chunks.push(encodeStringField(7, f.sessionId || ''));
  chunks.push(encodeStringField(8, f.hostId || ''));
  chunks.push(encodeStringField(9, f.agentId || ''));
  chunks.push(encodeStringField(10, f.mcpServerId || ''));
  chunks.push(encodeStringField(11, f.toolName || ''));
  chunks.push(encodeStringField(12, f.eventType || ''));

  // Field 16: map<string,string> entries (sorted for deterministic output)
  if (f.metadata) {
    for (const key of Object.keys(f.metadata).sort()) {
      chunks.push(encodeMapEntry(16, key, f.metadata[key]));
    }
  }

  // Field 17: sdk_public_key (DER SPKI) — required by the witness.
  chunks.push(encodeBytesField(17, f.sdkPublicKey || Buffer.alloc(0)));

  return Buffer.concat(chunks);
}
