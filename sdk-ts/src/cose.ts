/**
 * COSE_Sign1 (RFC 9052) builder for Golem SDK.
 *
 * Produces COSE_Sign1 envelopes compatible with go-cose (veraison/go-cose).
 * Uses ECDSA P-256 with SHA-256 (ES256, algorithm ID -7).
 *
 * COSE_Sign1 structure: CBOR Tag(18) [protected, unprotected, payload, signature]
 * Sig_structure:       ["Signature1", protected, external_aad, payload]
 *
 * Protected headers carry:
 *   alg: ES256 (-7)
 *   kid: hex-encoded SHA-256 of the DER public key (as bytes)
 *
 * Uses only Node.js built-in crypto module — no external signing dependencies.
 */

import * as crypto from 'crypto';
import * as cbor from 'cbor';

// COSE header labels (RFC 9052 §3.1)
const COSE_HEADER_ALG = 1;
const COSE_HEADER_KID = 4;

// COSE algorithm: ES256 = ECDSA w/ SHA-256 on P-256
const COSE_ALG_ES256 = -7;

// COSE_Sign1 CBOR tag
const COSE_SIGN1_TAG = 18;

// P-256 coordinate size in bytes
const P256_COORD_SIZE = 32;

/**
 * Compute the key ID: hex-encoded SHA-256 of the DER (SPKI) public key.
 * This matches the Go cosesign package's KeyID() method.
 */
export function computeKeyId(publicKeyDer: Buffer): string {
  return crypto.createHash('sha256').update(publicKeyDer).digest('hex');
}

/**
 * Build the COSE_Sign1 protected headers as CBOR bytes.
 * Contains: { 1: -7, 4: kid_bytes }
 *
 * Uses `cbor.encodeAsync` rather than the synchronous `cbor.encode`: the
 * latter's NoFilter-backed synchronous read-back is broken under Node.js >= 26
 * (it emits only the CBOR type/length prefix and drops the payload). The async
 * path flushes the encoder stream correctly and is byte-identical across Node
 * versions, so the produced wire format is unchanged.
 */
async function buildProtectedHeaders(keyId: string): Promise<Buffer> {
  const headers = new Map<number, any>();
  headers.set(COSE_HEADER_ALG, COSE_ALG_ES256);
  headers.set(COSE_HEADER_KID, Buffer.from(keyId, 'utf-8'));
  return cbor.encodeAsync(headers);
}

/**
 * Build the Sig_structure for COSE_Sign1 (RFC 9052 §4.4).
 * Sig_structure = ["Signature1", protectedHeaders, externalAad, payload]
 */
async function buildSigStructure(protectedHeaders: Buffer, payload: Buffer): Promise<Buffer> {
  return cbor.encodeAsync([
    'Signature1',
    protectedHeaders,
    Buffer.alloc(0), // external_aad (empty)
    payload,
  ]);
}

/**
 * Convert an ASN.1 DER-encoded ECDSA signature to raw r||s format (IEEE P1363).
 * DER format: SEQUENCE { INTEGER r, INTEGER s }
 * Raw format: r (32 bytes, zero-padded) || s (32 bytes, zero-padded)
 *
 * go-cose expects raw r||s format for ES256, not ASN.1 DER.
 */
function derSignatureToRaw(derSig: Buffer): Buffer {
  // DER: 0x30 <len> 0x02 <rLen> <r> 0x02 <sLen> <s>
  let offset = 0;

  if (derSig[offset++] !== 0x30) {
    throw new Error('Invalid DER signature: missing SEQUENCE tag');
  }

  // Skip SEQUENCE length (may be 1 or 2 bytes)
  const seqLen = derSig[offset];
  if (seqLen & 0x80) {
    // Long form length — for ECDSA P-256 this shouldn't happen but handle it
    const lenBytes = seqLen & 0x7f;
    offset += 1 + lenBytes;
  } else {
    offset += 1;
  }

  // Parse r
  if (derSig[offset++] !== 0x02) {
    throw new Error('Invalid DER signature: missing INTEGER tag for r');
  }
  const rLen = derSig[offset++];
  let rBytes = derSig.subarray(offset, offset + rLen);
  offset += rLen;

  // Parse s
  if (derSig[offset++] !== 0x02) {
    throw new Error('Invalid DER signature: missing INTEGER tag for s');
  }
  const sLen = derSig[offset++];
  let sBytes = derSig.subarray(offset, offset + sLen);

  // Strip leading zero byte if present (ASN.1 uses it for positive sign)
  if (rBytes.length > P256_COORD_SIZE && rBytes[0] === 0x00) {
    rBytes = rBytes.subarray(1);
  }
  if (sBytes.length > P256_COORD_SIZE && sBytes[0] === 0x00) {
    sBytes = sBytes.subarray(1);
  }

  // Zero-pad to coordinate size
  const raw = Buffer.alloc(P256_COORD_SIZE * 2);
  rBytes.copy(raw, P256_COORD_SIZE - rBytes.length);
  sBytes.copy(raw, P256_COORD_SIZE * 2 - sBytes.length);

  return raw;
}

/**
 * Sign a CBOR payload using COSE_Sign1 (ES256 / P-256).
 *
 * Returns the complete COSE_Sign1 CBOR bytes (tagged with Tag 18),
 * compatible with go-cose Verify().
 *
 * The signing follows RFC 9052 §4.4:
 * 1. Build Sig_structure = ["Signature1", protected, external_aad, payload]
 * 2. Sign SHA-256(CBOR(Sig_structure)) with ECDSA P-256
 * 3. Convert signature from ASN.1 DER to raw r||s (64 bytes)
 * 4. Wrap as CBOR Tag(18) [protected, unprotected, payload, signature]
 *
 * @param cborPayload - The CBOR-encoded BOM payload
 * @param privateKey  - Node.js EC private key (P-256)
 * @param publicKeyDer - DER-encoded SPKI public key (for key ID computation)
 * @returns COSE_Sign1 CBOR bytes
 */
export async function coseSign1(
  cborPayload: Buffer,
  privateKey: crypto.KeyObject,
  publicKeyDer: Buffer,
): Promise<Buffer> {
  const keyId = computeKeyId(publicKeyDer);
  const protectedHeaders = await buildProtectedHeaders(keyId);

  // Build the Sig_structure per RFC 9052 §4.4
  const sigStructure = await buildSigStructure(protectedHeaders, cborPayload);

  // Sign the Sig_structure using Node.js crypto (produces ASN.1 DER signature)
  // crypto.sign('sha256', data, key) computes SHA-256(data) then signs with ECDSA.
  const derSignature = crypto.sign('sha256', sigStructure, privateKey);

  // Convert DER to raw r||s (64 bytes) — required by COSE/go-cose
  const rawSignature = derSignatureToRaw(derSignature);

  // Build COSE_Sign1 = Tag(18) [protected, unprotected, payload, signature]
  const coseArray = [
    protectedHeaders,  // protected headers (CBOR-encoded map, as bstr)
    {},                // unprotected headers (empty map)
    cborPayload,       // payload
    rawSignature,      // signature (raw r||s, 64 bytes)
  ];

  return cbor.encodeAsync(new cbor.Tagged(COSE_SIGN1_TAG, coseArray));
}
