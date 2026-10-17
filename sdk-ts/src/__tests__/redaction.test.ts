import * as fs from 'fs';
import * as path from 'path';
import { redact, parseOperator, parseRedactionPolicy, RedactionPolicyCache } from '../redaction';

const vectorsPath = path.resolve(__dirname, '../../../pkg/redaction/testdata/vectors.json');

const describeVectors = fs.existsSync(vectorsPath) ? describe : describe.skip;

describeVectors('redaction matches the Golem agent', () => {
  const vectors = fs.existsSync(vectorsPath) ? JSON.parse(fs.readFileSync(vectorsPath, 'utf8')) : [];

  it('has the shared vectors', () => {
    expect(vectors.length).toBeGreaterThan(30);
  });

  for (const v of vectors) {
    it(v.name, () => {
      const r = redact(v.text, v.types, parseOperator(v.operator), v.custom ?? []);
      expect(r.text).toBe(v.want);
      expect(r.found).toEqual(v.found);
    });
  }
});

describe('the policy', () => {
  it('parses what the agent serves', () => {
    const p = parseRedactionPolicy({
      enabled: true,
      entity_types: ['EMAIL_ADDRESS'],
      operator: { type: 'hash', hash_len: 12 },
      custom: [{ EntityType: 'CUSTOM_BADGE', Pattern: 'BDG-\\d{6}', Group: 0, Score: 0.9, Shadow: false }],
    });
    expect(p.enabled).toBe(true);
    expect(p.operator.type).toBe('hash');
    expect(p.custom[0].EntityType).toBe('CUSTOM_BADGE');
    expect(() => parseRedactionPolicy([])).toThrow();
    expect(() => parseRedactionPolicy({})).toThrow();
  });

  it('is kept between fetches and through a failure', async () => {
    let t = 0;
    let calls = 0;
    let fail = false;
    const cache = new RedactionPolicyCache(
      async () => {
        calls++;
        if (fail) throw new Error('down');
        return { enabled: true, entity_types: [], operator: { type: 'replace' }, custom: [] };
      },
      30_000,
      5_000,
      () => t,
    );
    expect((await cache.current())?.policy.enabled).toBe(true);
    t = 10_000;
    await cache.current();
    expect(calls).toBe(1);
    fail = true;
    t = 40_000;
    expect((await cache.current())?.policy.enabled).toBe(true);
    expect(calls).toBe(2);
  });

  it('is null until a fetch succeeds, retried after a pause', async () => {
    let t = 0;
    let calls = 0;
    const cache = new RedactionPolicyCache(
      async () => {
        calls++;
        throw new Error('down');
      },
      30_000,
      5_000,
      () => t,
    );
    expect(await cache.current()).toBeNull();
    t = 1_000;
    expect(await cache.current()).toBeNull();
    expect(calls).toBe(1);
    t = 6_000;
    await cache.current();
    expect(calls).toBe(2);
  });
});
