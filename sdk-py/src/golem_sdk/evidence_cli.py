#!/usr/bin/env python3
"""
Golem Evidence Package CLI

Verifies transparency log integrity and exports cryptographic evidence packages
for audit and non-repudiation.
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

logger = logging.getLogger("golem_sdk.evidence_cli")

# Validation constants
_VALID_HASH = re.compile(r"^[a-f0-9]{64}$")
MAX_SEQUENCE_RANGE = 100_000  # Maximum entries per query to prevent DoS


def _validate_entry_hash(value: str) -> str:
    """Validate that an entry hash is a valid SHA-256 hex string."""
    if not _VALID_HASH.match(value):
        raise argparse.ArgumentTypeError(
            "Entry hash must be a 64-character lowercase hex string"
        )
    return value


def _validate_sequence_id(value: str) -> int:
    """Validate that a sequence ID is a positive integer."""
    try:
        seq = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Sequence ID must be an integer")
    if seq < 1:
        raise argparse.ArgumentTypeError("Sequence ID must be >= 1")
    if seq > 2_147_483_647:
        raise argparse.ArgumentTypeError("Sequence ID exceeds maximum value")
    return seq


def _validate_output_path(value: str) -> str:
    """Validate and resolve output file path. Prevents path traversal."""
    resolved = os.path.realpath(value)
    # Ensure the parent directory exists and is writable
    parent = os.path.dirname(resolved)
    if not os.path.isdir(parent):
        raise argparse.ArgumentTypeError(
            f"Output directory does not exist: {parent}"
        )
    return resolved


def get_db_connection():
    """Create a PostgreSQL connection using environment variables."""
    try:
        import psycopg2
    except ImportError:
        print(
            "Error: psycopg2 is required. Install with: pip install psycopg2-binary",
            file=sys.stderr,
        )
        sys.exit(1)

    db_password = os.environ.get("GOLEM_DB_PASSWORD")
    if not db_password:
        print(
            "Warning: GOLEM_DB_PASSWORD not set. Set it via environment variable.",
            file=sys.stderr,
        )
        # Fall back to default only for local development
        db_password = "golem"

    try:
        return psycopg2.connect(
            host=os.environ.get("GOLEM_DB_HOST", "localhost"),
            port=os.environ.get("GOLEM_DB_PORT", "5432"),
            dbname=os.environ.get("GOLEM_DB_NAME", "golem_ledger"),
            user=os.environ.get("GOLEM_DB_USER", "golem"),
            password=db_password,
        )
    except Exception:
        print("Error: Failed to connect to the database.", file=sys.stderr)
        sys.exit(1)


def compute_entry_hash(cbor_payload: bytes, previous_hash: str) -> str:
    """Recompute the entry hash from CBOR payload and previous hash."""
    data = cbor_payload + previous_hash.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def extract_cbor_from_cose(cose_bytes: bytes) -> bytes:
    """Extract the inner CBOR payload from a COSE_Sign1 envelope.

    Decodes the COSE_Sign1 structure and returns the payload without
    verifying the signature (read-path trust model).

    COSE_Sign1 (RFC 9052) is CBOR Tag(18) wrapping the array
    ``[protected, unprotected, payload, signature]`` — the payload is
    element index 2. Decoded with cbor2 directly (a hard dependency); the
    read path needs only the payload, not signature verification.
    """
    try:
        import cbor2
        tagged = cbor2.loads(cose_bytes)
        tag = getattr(tagged, "tag", None)
        if tag != 18:
            raise ValueError(f"not a COSE_Sign1 envelope (tag={tag})")
        arr = tagged.value
        if not isinstance(arr, (list, tuple)) or len(arr) != 4:
            raise ValueError("malformed COSE_Sign1 array")
        payload = arr[2]
        if payload is None:
            raise ValueError("COSE payload is nil")
        return bytes(payload)
    except ImportError:
        logger.warning("cbor2 not installed; cannot decode COSE envelope")
        raise
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to extract CBOR from COSE: {type(e).__name__}") from e


def _get_cbor_payload(row_payload_cbor, row_cose_payload):
    """Resolve the CBOR payload from either legacy payload_cbor or COSE envelope.

    Returns the raw CBOR bytes suitable for hash verification.
    """
    # Prefer COSE envelope (new schema)
    if row_cose_payload:
        cose_bytes = bytes(row_cose_payload)
        return extract_cbor_from_cose(cose_bytes)
    # Fall back to legacy payload_cbor column
    if row_payload_cbor:
        return bytes(row_payload_cbor)
    raise ValueError("Neither raw_cose_payload nor payload_cbor found for entry")


def _has_column(cur, table: str, column: str) -> bool:
    """Check if a column exists in a table (PostgreSQL)."""
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s LIMIT 1",
        (table, column),
    )
    return cur.fetchone() is not None


def _build_select_query(cur, base_cols: str, table: str, where: str = "") -> str:
    """Build a SELECT query that includes raw_cose_payload if available."""
    has_cose = _has_column(cur, table, "raw_cose_payload")
    cose_col = ", raw_cose_payload" if has_cose else ", NULL AS raw_cose_payload"
    query = f"SELECT {base_cols}{cose_col} FROM {table}"
    if where:
        query += f" {where}"
    return query


def cmd_export(args):
    """Export an evidence package for a given entry hash."""
    conn = get_db_connection()
    cur = conn.cursor()

    query = _build_select_query(
        cur,
        "sequence_id, entry_hash, previous_hash, agent_id, mcp_server_id, "
        "tool_name, payload_cbor, workload_identity, signature, "
        "recorded_at, storage_uri",
        "golem_transparency_log",
        "WHERE entry_hash = %s",
    )
    cur.execute(query, (args.entry_hash,))
    row = cur.fetchone()
    if not row:
        print("Error: No entry found with the specified hash.", file=sys.stderr)
        sys.exit(1)

    (
        seq_id, entry_hash, prev_hash, agent_id, mcp_server_id,
        tool_name, payload_cbor, workload_id, signature,
        recorded_at, storage_uri, raw_cose_payload,
    ) = row

    # Verify hash integrity — extract CBOR from COSE if available
    cbor_data = _get_cbor_payload(payload_cbor, raw_cose_payload)
    recomputed = compute_entry_hash(cbor_data, prev_hash)
    hash_valid = recomputed == entry_hash

    # Verify chain link
    chain_valid = True
    if seq_id > 1:
        cur.execute(
            "SELECT entry_hash FROM golem_transparency_log WHERE sequence_id = %s",
            (seq_id - 1,),
        )
        prev_row = cur.fetchone()
        if prev_row:
            chain_valid = prev_row[0] == prev_hash
        else:
            chain_valid = False

    evidence = {
        "evidence_package": {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "ledger_record": {
            "sequence_id": seq_id,
            "entry_hash": entry_hash,
            "previous_hash": prev_hash,
            "agent_id": agent_id,
            "mcp_server_id": mcp_server_id,
            "tool_name": tool_name,
            "workload_identity": workload_id,
            "signature_hex": signature.hex() if signature else None,
            "recorded_at": recorded_at.isoformat() if recorded_at else None,
            "storage_uri": storage_uri,
        },
        "verification": {
            "hash_integrity": "PASS" if hash_valid else "FAIL",
            "recomputed_hash": recomputed,
            "chain_link": "PASS" if chain_valid else "FAIL",
        },
    }

    print(json.dumps(evidence, indent=2))
    conn.close()


def cmd_verify(args):
    """Verify a single entry's integrity."""
    conn = get_db_connection()
    cur = conn.cursor()

    query = _build_select_query(
        cur,
        "sequence_id, entry_hash, previous_hash, payload_cbor, signature",
        "golem_transparency_log",
        "WHERE entry_hash = %s",
    )
    cur.execute(query, (args.entry_hash,))
    row = cur.fetchone()
    if not row:
        print("Error: No entry found with the specified hash.", file=sys.stderr)
        sys.exit(1)

    seq_id, entry_hash, prev_hash, payload_cbor, signature, raw_cose_payload = row

    # 1. Recompute hash — extract CBOR from COSE if available
    cbor_data = _get_cbor_payload(payload_cbor, raw_cose_payload)
    recomputed = compute_entry_hash(cbor_data, prev_hash)
    hash_ok = recomputed == entry_hash

    # 2. Verify chain link
    chain_ok = True
    if seq_id > 1:
        cur.execute(
            "SELECT entry_hash FROM golem_transparency_log WHERE sequence_id = %s",
            (seq_id - 1,),
        )
        prev_row = cur.fetchone()
        chain_ok = prev_row is not None and prev_row[0] == prev_hash

    print(f"Entry: {entry_hash}")
    print(f"  Sequence:   {seq_id}")
    print(f"  Hash Check: {'PASS' if hash_ok else 'FAIL'}")
    print(f"  Chain Link: {'PASS' if chain_ok else 'FAIL'}")

    if not hash_ok or not chain_ok:
        sys.exit(1)

    conn.close()


def cmd_verify_chain(args):
    """Verify an entire range of the hash chain."""
    if args.to_seq - args.from_seq > MAX_SEQUENCE_RANGE:
        print(
            f"Error: Range exceeds maximum of {MAX_SEQUENCE_RANGE} entries. "
            "Use a smaller range.",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = get_db_connection()
    cur = conn.cursor()

    query = _build_select_query(
        cur,
        "sequence_id, entry_hash, previous_hash, payload_cbor",
        "golem_transparency_log",
        "WHERE sequence_id BETWEEN %s AND %s ORDER BY sequence_id ASC",
    )
    cur.execute(query, (args.from_seq, args.to_seq))
    rows = cur.fetchall()

    if not rows:
        print("No entries found in the specified range.", file=sys.stderr)
        sys.exit(1)

    errors = []
    prev_hash = None

    for row in rows:
        seq_id, entry_hash, stored_prev_hash, payload_cbor, raw_cose_payload = row
        # Verify hash integrity — extract CBOR from COSE if available
        cbor_data = _get_cbor_payload(payload_cbor, raw_cose_payload)
        recomputed = compute_entry_hash(cbor_data, stored_prev_hash)
        if recomputed != entry_hash:
            errors.append(
                f"  TAMPER at seq={seq_id}: hash mismatch "
                f"(stored={entry_hash[:16]}... computed={recomputed[:16]}...)"
            )

        # Verify chain continuity
        if prev_hash is not None and stored_prev_hash != prev_hash:
            errors.append(
                f"  BREAK at seq={seq_id}: previous_hash mismatch "
                f"(expected={prev_hash[:16]}... got={stored_prev_hash[:16]}...)"
            )

        prev_hash = entry_hash

    print(f"Chain Verification: seq {args.from_seq} -> {args.to_seq} ({len(rows)} entries)")
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("RESULT: PASS - All entries verified, chain intact.")

    conn.close()


def cmd_audit_report(args):
    """Generate a JSON audit report for a range of entries."""
    if args.to_seq - args.from_seq > MAX_SEQUENCE_RANGE:
        print(
            f"Error: Range exceeds maximum of {MAX_SEQUENCE_RANGE} entries. "
            "Use a smaller range.",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = get_db_connection()
    cur = conn.cursor()

    query = _build_select_query(
        cur,
        "sequence_id, entry_hash, previous_hash, agent_id, "
        "mcp_server_id, tool_name, payload_cbor, workload_identity, "
        "signature, recorded_at, storage_uri",
        "golem_transparency_log",
        "WHERE sequence_id BETWEEN %s AND %s ORDER BY sequence_id ASC",
    )
    cur.execute(query, (args.from_seq, args.to_seq))
    rows = cur.fetchall()

    entries = []
    prev_hash = None

    for row in rows:
        (
            seq_id, entry_hash, stored_prev_hash, agent_id, mcp_server_id,
            tool_name, payload_cbor, workload_id, signature,
            recorded_at, storage_uri, raw_cose_payload,
        ) = row

        cbor_data = _get_cbor_payload(payload_cbor, raw_cose_payload)
        recomputed = compute_entry_hash(cbor_data, stored_prev_hash)
        hash_ok = recomputed == entry_hash
        chain_ok = prev_hash is None or stored_prev_hash == prev_hash

        entries.append(
            {
                "sequence_id": seq_id,
                "entry_hash": entry_hash,
                "agent_id": agent_id,
                "mcp_server_id": mcp_server_id,
                "tool_name": tool_name,
                "recorded_at": recorded_at.isoformat() if recorded_at else None,
                "storage_uri": storage_uri,
                "verification": {
                    "hash_integrity": "PASS" if hash_ok else "FAIL",
                    "chain_link": "PASS" if chain_ok else "FAIL",
                },
            }
        )
        prev_hash = entry_hash

    all_pass = all(
        e["verification"]["hash_integrity"] == "PASS"
        and e["verification"]["chain_link"] == "PASS"
        for e in entries
    )

    report = {
        "audit_report": {
            "version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "range": {"from": args.from_seq, "to": args.to_seq},
            "total_entries": len(entries),
            "overall_verdict": "PASS" if all_pass else "FAIL",
        },
        "entries": entries,
    }

    output = json.dumps(report, indent=2)
    if args.output:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(output)
        except Exception:
            os.close(fd)
            raise
        print(f"Audit report written to {args.output}")
    else:
        print(output)

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        prog="golem-evidence",
        description="Golem Evidence Package CLI - Verify transparency log integrity and export evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # export
    export_parser = subparsers.add_parser(
        "export", help="Export an evidence package for a given entry"
    )
    export_parser.add_argument(
        "--entry-hash",
        required=True,
        type=_validate_entry_hash,
        help="The SHA-256 hash of the entry (64-char hex)",
    )

    # verify
    verify_parser = subparsers.add_parser(
        "verify", help="Verify a single entry's integrity"
    )
    verify_parser.add_argument(
        "--entry-hash",
        required=True,
        type=_validate_entry_hash,
        help="The SHA-256 hash of the entry (64-char hex)",
    )

    # verify-chain
    chain_parser = subparsers.add_parser(
        "verify-chain", help="Verify a range of the hash chain"
    )
    chain_parser.add_argument(
        "--from", dest="from_seq", type=_validate_sequence_id, required=True,
        help="Starting sequence ID (positive integer)",
    )
    chain_parser.add_argument(
        "--to", dest="to_seq", type=_validate_sequence_id, required=True,
        help="Ending sequence ID (positive integer)",
    )

    # audit-report
    audit_parser = subparsers.add_parser(
        "audit-report", help="Generate an audit report"
    )
    audit_parser.add_argument(
        "--from", dest="from_seq", type=_validate_sequence_id, required=True,
        help="Starting sequence ID (positive integer)",
    )
    audit_parser.add_argument(
        "--to", dest="to_seq", type=_validate_sequence_id, required=True,
        help="Ending sequence ID (positive integer)",
    )
    audit_parser.add_argument(
        "--output",
        type=_validate_output_path,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    if args.command == "export":
        cmd_export(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "verify-chain":
        cmd_verify_chain(args)
    elif args.command == "audit-report":
        cmd_audit_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
