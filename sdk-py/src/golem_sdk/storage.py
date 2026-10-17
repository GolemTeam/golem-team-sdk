"""
DEPRECATED: BYOB storage has moved to the Golem Relay.

The Relay is now the single BYOB writer. SDKs and the Witness pass raw content
through to the Relay, which handles storage to S3/GCS/Azure/local disk.

These storage backends are retained for backward compatibility but are no longer
the recommended path. Use GolemSDK(witness_url="http://localhost:39393") instead,
which routes through the Witness -> Relay pipeline.

To migrate:
  - Remove any StorageBackend instantiation from your code
  - Use GolemSDK(witness_url="http://localhost:39393") (witness mode)
  - The Relay handles BYOB storage automatically

These classes will be removed in a future major version.
"""

import os
import json
import logging
import re
import warnings
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Strict validation: hash IDs must be lowercase hex SHA-256 (64 chars)
_VALID_HASH_ID = re.compile(r"^[a-f0-9]{64}$")

# Maximum payload size (10 MB) to prevent resource exhaustion
MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024


def _validate_hash_id(hash_id: str) -> None:
    """Validate that hash_id is a valid SHA-256 hex string. Prevents path traversal."""
    if not isinstance(hash_id, str) or not _VALID_HASH_ID.match(hash_id):
        raise ValueError(
            f"Invalid hash_id: must be a 64-character lowercase hex string, got {repr(hash_id[:80])}"
        )


def _safe_serialize(payload: dict) -> str:
    """Serialize payload to JSON with size limit enforcement."""
    body = json.dumps(payload, indent=2)
    if len(body.encode("utf-8")) > MAX_PAYLOAD_SIZE_BYTES:
        raise ValueError(
            f"Payload exceeds maximum size of {MAX_PAYLOAD_SIZE_BYTES} bytes"
        )
    return body


class StorageBackend(ABC):
    """
    Deprecated: BYOB storage has moved to the Golem Relay.

    The Relay is now the single BYOB writer. This ABC is retained for backward
    compatibility only. Use GolemSDK(witness_url=...) instead.
    """

    @abstractmethod
    def store_raw_payload(self, hash_id: str, payload: dict) -> str:
        pass


class LocalDiskStorage(StorageBackend):
    """Deprecated: Use the Relay for BYOB storage. Retained as a fallback for local dev."""

    def __init__(self, base_path: str = "/tmp/golem-byob"):
        self.base_path = os.path.realpath(base_path)
        os.makedirs(self.base_path, mode=0o700, exist_ok=True)

    def store_raw_payload(self, hash_id: str, payload: dict) -> str:
        _validate_hash_id(hash_id)
        body = _safe_serialize(payload)

        file_path = os.path.join(self.base_path, f"{hash_id}.json")
        # Double-check resolved path is within base_path (defense in depth)
        resolved = os.path.realpath(file_path)
        if not resolved.startswith(self.base_path + os.sep) and resolved != self.base_path:
            raise ValueError("Path traversal detected: resolved path escapes base directory")

        # Write with restrictive permissions (owner read/write only)
        fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(body)
        except Exception:
            os.close(fd)
            raise
        return f"file://{resolved}"


class S3Storage(StorageBackend):
    """Deprecated: BYOB storage has moved to the Golem Relay. Configure S3 on the Relay instead."""

    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        warnings.warn(
            "S3Storage is deprecated. BYOB storage has moved to the Golem Relay. "
            "Use GolemSDK(witness_url='http://localhost:39393') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not bucket or not isinstance(bucket, str):
            raise ValueError("S3 bucket name is required")
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3", region_name=self.region)
            except ImportError:
                raise ImportError(
                    "[Golem S3] boto3 is not installed. "
                    "Install it with: pip install golem-team-sdk[aws]"
                )
        return self._client

    def store_raw_payload(self, hash_id: str, payload: dict) -> str:
        _validate_hash_id(hash_id)
        body = _safe_serialize(payload)
        key = f"{self.prefix}{hash_id}.json"
        uri = f"s3://{self.bucket}/{key}"

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            logger.info("[Golem S3] Stored raw payload at %s", uri)
            return uri
        except Exception as e:
            raise RuntimeError(f"[Golem S3] Failed to upload: {type(e).__name__}") from e


class GCSStorage(StorageBackend):
    """Deprecated: BYOB storage has moved to the Golem Relay. Configure GCS on the Relay instead."""

    def __init__(self, bucket: str, prefix: str = "", storage_client=None):
        warnings.warn(
            "GCSStorage is deprecated. BYOB storage has moved to the Golem Relay. "
            "Use GolemSDK(witness_url='http://localhost:39393') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not bucket or not isinstance(bucket, str):
            raise ValueError("GCS bucket name is required")
        self.bucket_name = bucket
        self.prefix = prefix
        self._storage_client = storage_client

    @property
    def client(self):
        if self._storage_client is None:
            try:
                from google.cloud import storage
                self._storage_client = storage.Client()
            except ImportError:
                raise ImportError(
                    "[Golem GCS] google-cloud-storage is not installed. "
                    "Install it with: pip install golem-team-sdk[gcp]"
                )
        return self._storage_client

    def store_raw_payload(self, hash_id: str, payload: dict) -> str:
        _validate_hash_id(hash_id)
        body = _safe_serialize(payload)
        blob_name = f"{self.prefix}{hash_id}.json"
        uri = f"gs://{self.bucket_name}/{blob_name}"

        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_string(body, content_type="application/json")
            logger.info("[Golem GCS] Stored raw payload at %s", uri)
            return uri
        except Exception as e:
            raise RuntimeError(f"[Golem GCS] Failed to upload: {type(e).__name__}") from e


class AzureBlobStorage(StorageBackend):
    """Deprecated: BYOB storage has moved to the Golem Relay. Configure Azure Blob on the Relay instead."""

    def __init__(self, connection_string: str, container_name: str, prefix: str = ""):
        warnings.warn(
            "AzureBlobStorage is deprecated. BYOB storage has moved to the Golem Relay. "
            "Use GolemSDK(witness_url='http://localhost:39393') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not connection_string or not isinstance(connection_string, str):
            raise ValueError("Azure connection string is required")
        if not container_name or not isinstance(container_name, str):
            raise ValueError("Azure container name is required")
        self.container_name = container_name
        self.prefix = prefix
        self._container_client = None

        try:
            from azure.storage.blob import BlobServiceClient
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self._container_client = blob_service_client.get_container_client(container_name)
        except ImportError:
            raise ImportError(
                "[Golem Azure] azure-storage-blob is not installed. "
                "Install it with: pip install golem-team-sdk[azure]"
            )

    def store_raw_payload(self, hash_id: str, payload: dict) -> str:
        _validate_hash_id(hash_id)
        body = _safe_serialize(payload)
        blob_name = f"{self.prefix}{hash_id}.json"

        try:
            # Ensure container exists — catch only the expected conflict error
            try:
                self._container_client.create_container()
            except Exception as container_err:
                # Azure raises ResourceExistsError when container already exists
                err_name = type(container_err).__name__
                if "Exists" not in err_name and "Conflict" not in str(container_err):
                    logger.warning("[Golem Azure] Container creation warning: %s", err_name)

            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                body.encode("utf-8"),
                overwrite=True,
                content_settings={"content_type": "application/json"},
            )
            url = blob_client.url
            logger.info("[Golem Azure] Stored raw payload successfully")
            return url
        except Exception as e:
            raise RuntimeError(
                f"[Golem Azure] Failed to upload blob: {type(e).__name__}"
            ) from e
