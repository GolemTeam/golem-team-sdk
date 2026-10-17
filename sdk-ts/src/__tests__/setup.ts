// Default the SDK durable-delivery buffer OFF for tests so transport tests
// don't touch ~/.golem/sdk-wal. Tests that exercise the buffer set
// GOLEM_SDK_WAL_PATH to a temp dir themselves before constructing a client.
process.env.GOLEM_SDK_WAL_PATH = process.env.GOLEM_SDK_WAL_PATH || 'off';
