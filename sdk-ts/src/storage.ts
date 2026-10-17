/**
 * DEPRECATED: BYOB storage has moved to the Golem Relay.
 *
 * The Relay is now the single BYOB writer. SDKs and the Witness pass raw
 * content through to the Relay, which handles storage to S3/GCS/Azure/local disk.
 *
 * These storage backends are retained for backward compatibility but are no longer
 * the recommended path. Use GolemSDK({ witnessUrl: "http://localhost:39393" }) instead,
 * which routes through the Witness -> Relay pipeline.
 *
 * To migrate:
 *   - Remove any StorageBackend instantiation from your code
 *   - Use GolemSDK({ witnessUrl: "http://localhost:39393" }) (witness mode)
 *   - The Relay handles BYOB storage automatically
 *
 * These classes will be removed in a future major version.
 */

import * as fs from 'fs';
import * as path from 'path';

/**
 * @deprecated BYOB storage has moved to the Golem Relay. Use witness mode instead.
 */
export interface StorageBackend {
  /**
   * Stashes the raw, sensitive payload in the customer's BYOB environment.
   * returning a URI locator that can be placed in the Transparency Log.
   * @deprecated Use GolemSDK({ witnessUrl: "http://localhost:39393" }) instead.
   */
  storeRawPayload(hashId: string, payload: any): Promise<string>;
}

/** Strict validation: hash IDs must be lowercase hex SHA-256 (64 chars) */
const VALID_HASH_ID = /^[a-f0-9]{64}$/;

/** Maximum payload size (10 MB) to prevent resource exhaustion */
const MAX_PAYLOAD_SIZE_BYTES = 10 * 1024 * 1024;

function validateHashId(hashId: string): void {
  if (typeof hashId !== 'string' || !VALID_HASH_ID.test(hashId)) {
    throw new Error('Invalid hashId: must be a 64-character lowercase hex string');
  }
}

function safeSerialize(payload: any): string {
  const body = JSON.stringify(payload, null, 2);
  if (Buffer.byteLength(body, 'utf-8') > MAX_PAYLOAD_SIZE_BYTES) {
    throw new Error(`Payload exceeds maximum size of ${MAX_PAYLOAD_SIZE_BYTES} bytes`);
  }
  return body;
}

/** @deprecated BYOB storage has moved to the Golem Relay. Retained as a fallback for local dev. */
export class LocalDiskStorage implements StorageBackend {
  private readonly resolvedBasePath: string;

  constructor(private basePath: string = '/tmp/golem-byob') {
    this.resolvedBasePath = path.resolve(this.basePath);
    if (!fs.existsSync(this.resolvedBasePath)) {
      fs.mkdirSync(this.resolvedBasePath, { recursive: true, mode: 0o700 });
    }
  }

  async storeRawPayload(hashId: string, payload: any): Promise<string> {
    validateHashId(hashId);
    const body = safeSerialize(payload);

    const filePath = path.join(this.resolvedBasePath, `${hashId}.json`);
    // Defense in depth: verify resolved path is within base directory
    const resolved = path.resolve(filePath);
    if (!resolved.startsWith(this.resolvedBasePath + path.sep) && resolved !== this.resolvedBasePath) {
      throw new Error('Path traversal detected: resolved path escapes base directory');
    }

    // Write with restrictive permissions (owner read/write only)
    fs.writeFileSync(filePath, body, { mode: 0o600 });
    return `file://${resolved}`;
  }
}

/** @deprecated BYOB storage has moved to the Golem Relay. Configure S3 on the Relay instead. */
export class S3Storage implements StorageBackend {
  private client: any;
  private bucket: string;
  private prefix: string;

  constructor(bucket: string, prefix: string = '', region: string = 'us-east-1', s3Client?: any) {
    if (!bucket || typeof bucket !== 'string') {
      throw new Error('S3 bucket name is required');
    }
    this.bucket = bucket;
    this.prefix = prefix;

    if (s3Client) {
      this.client = s3Client;
    } else {
      try {
        const { S3Client } = require('@aws-sdk/client-s3');
        this.client = new S3Client({ region });
      } catch (e) {
        console.warn('[Golem S3] @aws-sdk/client-s3 not installed. Install it with: npm install @aws-sdk/client-s3');
        this.client = null;
      }
    }
  }

  async storeRawPayload(hashId: string, payload: any): Promise<string> {
    validateHashId(hashId);
    const body = safeSerialize(payload);
    const key = `${this.prefix}${hashId}.json`;
    const uri = `s3://${this.bucket}/${key}`;

    if (!this.client) {
      console.warn(`[Golem S3] Client not available. Would upload to ${uri}`);
      return uri;
    }

    try {
      const { PutObjectCommand } = require('@aws-sdk/client-s3');
      await this.client.send(new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: body,
        ContentType: 'application/json',
      }));
      console.log(`[Golem S3] Stored raw payload successfully`);
      return uri;
    } catch (error: any) {
      throw new Error(`[Golem S3] Failed to upload: ${error.constructor?.name || 'UnknownError'}`);
    }
  }
}

/** @deprecated BYOB storage has moved to the Golem Relay. Configure GCS on the Relay instead. */
export class GCSStorage implements StorageBackend {
  private storage: any;
  private bucket: string;
  private prefix: string;

  constructor(bucket: string, prefix: string = '', storageClient?: any) {
    if (!bucket || typeof bucket !== 'string') {
      throw new Error('GCS bucket name is required');
    }
    this.bucket = bucket;
    this.prefix = prefix;

    if (storageClient) {
      this.storage = storageClient;
    } else {
      try {
        const { Storage } = require('@google-cloud/storage');
        this.storage = new Storage();
      } catch (e) {
        console.warn('[Golem GCS] @google-cloud/storage not installed. Install it with: npm install @google-cloud/storage');
        this.storage = null;
      }
    }
  }

  async storeRawPayload(hashId: string, payload: any): Promise<string> {
    validateHashId(hashId);
    const body = safeSerialize(payload);
    const fileName = `${this.prefix}${hashId}.json`;
    const uri = `gs://${this.bucket}/${fileName}`;

    if (!this.storage) {
      console.warn(`[Golem GCS] Client not available. Would upload to ${uri}`);
      return uri;
    }

    try {
      const bucket = this.storage.bucket(this.bucket);
      const file = bucket.file(fileName);
      await file.save(body, {
        contentType: 'application/json',
        resumable: false,
      });
      console.log(`[Golem GCS] Stored raw payload successfully`);
      return uri;
    } catch (error: any) {
      throw new Error(`[Golem GCS] Failed to upload: ${error.constructor?.name || 'UnknownError'}`);
    }
  }
}

/** @deprecated BYOB storage has moved to the Golem Relay. Configure Azure Blob on the Relay instead. */
export class AzureBlobStorage implements StorageBackend {
  private containerClient: any;
  private prefix: string;

  constructor(connectionString: string, containerName: string, prefix: string = '') {
    if (!connectionString || typeof connectionString !== 'string') {
      throw new Error('Azure connection string is required');
    }
    if (!containerName || typeof containerName !== 'string') {
      throw new Error('Azure container name is required');
    }
    this.prefix = prefix;

    const { BlobServiceClient } = require('@azure/storage-blob');
    const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
    this.containerClient = blobServiceClient.getContainerClient(containerName);
  }

  async storeRawPayload(hashId: string, payload: any): Promise<string> {
    validateHashId(hashId);
    const body = safeSerialize(payload);
    const blobName = `${this.prefix}${hashId}.json`;

    try {
      // Ensure container exists (no-op if already created)
      await this.containerClient.createIfNotExists();

      const blockBlobClient = this.containerClient.getBlockBlobClient(blobName);
      await blockBlobClient.upload(body, Buffer.byteLength(body), {
        blobHTTPHeaders: { blobContentType: 'application/json' },
      });

      console.log(`[Golem Azure] Stored raw payload successfully`);
      return blockBlobClient.url;
    } catch (error: any) {
      throw new Error(`[Golem Azure] Failed to upload: ${error.constructor?.name || 'UnknownError'}`);
    }
  }
}
