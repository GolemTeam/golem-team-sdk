/**
 * Port of pkg/redaction (Go), pinned to it by pkg/redaction/testdata/vectors.json.
 * The SDK redacts before it signs because the Golem agent cannot rewrite an
 * event whose hash the SDK already committed.
 */

import * as crypto from 'crypto';

export interface Operator {
  type: string;
  new_value?: string;
  masking_char?: string;
  chars_to_mask?: number;
  from_end?: boolean;
  hash_len?: number;
  salt?: string;
}

export interface CustomSpec {
  EntityType: string;
  Pattern: string;
  Group?: number;
  Score: number;
  Shadow?: boolean;
}

export interface RedactionPolicy {
  enabled: boolean;
  entity_types: string[];
  operator: Operator;
  custom: CustomSpec[];
}

export interface RedactResult {
  text: string;
  counts: Record<string, number>;
  found: string[];
}

interface Entity {
  type: string;
  start: number;
  end: number;
  score: number;
}

interface Recognizer {
  entity: string;
  re: RegExp;
  group: number;
  validate: (match: string) => number;
}

type RegExpIndices = RegExpExecArray & { indices?: Array<[number, number] | undefined> };

const DEFAULT_THRESHOLD = 0.5;

// Go's \s is [\t\n\f\r ]; JavaScript's also takes \v and Unicode spaces.
const GO_SPACE = String.raw`\t\n\f\r `;

function recognizer(
  entity: string,
  source: string,
  flags: string,
  validate: (m: string) => number,
  group = 0,
): Recognizer {
  return { entity, re: new RegExp(source, 'g' + (group > 0 ? 'd' : '') + flags), group, validate };
}

const alwaysValid = (score: number) => (): number => score;

function rejectPlaceholder(score: number): (m: string) => number {
  const bad = ['example', 'changeme', 'your_', 'your-', 'placeholder', 'redacted', 'xxxx', '<', 'insert_', 'dummy', 'sample', 'test_token', 'fake'];
  return (m: string): number => {
    const l = m.toLowerCase();
    if (bad.some((b) => l.includes(b))) return 0;
    if (l.length > 0 && l.split('').every((c) => c === l[0])) return 0;
    return score;
  };
}

function digitsOnly(s: string): string {
  return s.replace(/[^0-9]/g, '');
}

function luhnValid(s: string): boolean {
  let sum = 0;
  let alt = false;
  let digits = 0;
  for (let i = s.length - 1; i >= 0; i--) {
    const c = s.charCodeAt(i);
    if (c < 48 || c > 57) continue;
    digits++;
    let d = c - 48;
    if (alt) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    alt = !alt;
  }
  return digits > 0 && sum % 10 === 0;
}

function validateSSN(m: string): number {
  const d = digitsOnly(m);
  if (d.length !== 9) return 0;
  const area = d.slice(0, 3);
  if (area === '000' || area === '666' || area[0] === '9') return 0;
  if (d.slice(3, 5) === '00' || d.slice(5) === '0000') return 0;
  return 0.85;
}

function validateCard(m: string): number {
  const n = digitsOnly(m).length;
  if (n < 13 || n > 19) return 0;
  return luhnValid(m) ? 0.95 : 0;
}

function validatePhone(m: string): number {
  const n = digitsOnly(m).length;
  return n < 10 || n > 11 ? 0 : 0.6;
}

function validateIPv4(m: string): number {
  const octets = m.split('.');
  if (octets.length !== 4) return 0;
  return octets.every((o) => Number(o) <= 255) ? 0.6 : 0;
}

function validateIBAN(m: string): number {
  if (m.length < 15 || m.length > 34) return 0;
  const rearranged = m.slice(4) + m.slice(0, 4);
  let rem = 0;
  for (const c of rearranged) {
    if (c >= '0' && c <= '9') {
      rem = (rem * 10 + (c.charCodeAt(0) - 48)) % 97;
    } else if (c >= 'A' && c <= 'Z') {
      rem = (rem * 100 + (c.charCodeAt(0) - 65 + 10)) % 97;
    } else {
      return 0;
    }
  }
  return rem === 1 ? 0.8 : 0;
}

const structuredRecognizers: Recognizer[] = [
  recognizer('EMAIL_ADDRESS', String.raw`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`, '', alwaysValid(1.0)),
  recognizer('US_SSN', String.raw`\b\d{3}[- ]\d{2}[- ]\d{4}\b`, '', validateSSN),
  recognizer('CREDIT_CARD', String.raw`\b\d(?:[ -]?\d){12,18}\b`, '', validateCard),
  recognizer('PHONE_NUMBER', String.raw`\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b`, '', validatePhone),
  recognizer('IP_ADDRESS', String.raw`\b(?:\d{1,3}\.){3}\d{1,3}\b`, '', validateIPv4),
  recognizer('IBAN_CODE', String.raw`\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b`, '', validateIBAN),
  recognizer('AWS_ACCESS_KEY', String.raw`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`, '', alwaysValid(0.9)),
  recognizer('CRYPTO', String.raw`\b0x[a-fA-F0-9]{40}\b`, '', alwaysValid(0.7)),
];

const secretRecognizers: Recognizer[] = [
  recognizer('GITHUB_TOKEN', String.raw`\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})`, '', alwaysValid(0.95)),
  recognizer('GITLAB_TOKEN', String.raw`\bglpat-[A-Za-z0-9_\-]{20,}`, '', alwaysValid(0.95)),
  recognizer('SLACK_TOKEN', String.raw`\b(?:xox[abprs]-[A-Za-z0-9\-]{10,}|xapp-\d-[A-Za-z0-9\-]{10,})`, '', alwaysValid(0.95)),
  recognizer('ANTHROPIC_API_KEY', String.raw`\bsk-ant-[A-Za-z0-9_\-]{20,}`, '', alwaysValid(0.95)),
  recognizer('OPENAI_API_KEY', String.raw`\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b`, '', alwaysValid(0.95)),
  recognizer('GOOGLE_API_KEY', String.raw`\bAIza[A-Za-z0-9_\-]{35}`, '', alwaysValid(0.95)),
  recognizer('STRIPE_KEY', String.raw`\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}`, '', alwaysValid(0.95)),
  recognizer('NPM_TOKEN', String.raw`\bnpm_[A-Za-z0-9]{36}`, '', alwaysValid(0.95)),
  recognizer('HUGGINGFACE_TOKEN', String.raw`\bhf_[A-Za-z0-9]{30,}`, '', alwaysValid(0.9)),
  recognizer('PRIVATE_KEY', String.raw`-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----`, 's', alwaysValid(1.0)),
  recognizer('JWT', String.raw`\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}`, '', alwaysValid(0.9)),
  recognizer('BEARER_TOKEN', String.raw`bearer[${GO_SPACE}]+([A-Za-z0-9._~+/\-]{16,}=*)`, 'i', rejectPlaceholder(0.85), 1),
  recognizer(
    'DB_PASSWORD',
    String.raw`\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?)://[^:@/${GO_SPACE}]+:([^@/${GO_SPACE}]+)@`,
    'i',
    rejectPlaceholder(0.9),
    1,
  ),
  recognizer('GOLEM_KEY', String.raw`\b(?:gla|glk)_[A-Za-z0-9_\-]{22,}`, '', alwaysValid(1.0)),
  recognizer(
    'GENERIC_SECRET',
    String.raw`(?:api[_\-]?key|secret|token|password|passwd|access[_\-]?key)["']?[${GO_SPACE}]*[:=][${GO_SPACE}]*["']?([A-Za-z0-9/+=_.\-]{16,})`,
    'i',
    rejectPlaceholder(0.6),
    1,
  ),
];

const builtinByName = new Map<string, Recognizer>(
  [...structuredRecognizers, ...secretRecognizers].map((r) => [r.entity, r]),
);

export const STRUCTURED_TYPES: readonly string[] = structuredRecognizers.map((r) => r.entity);
export const SECRET_TYPES: readonly string[] = secretRecognizers.map((r) => r.entity);

// Mirrors Go's FindAll loop, which drops an empty match that abuts the previous match.
function find(r: Recognizer, text: string): Entity[] {
  const out: Entity[] = [];
  let pos = 0;
  let prevEnd = -1;
  while (pos <= text.length) {
    r.re.lastIndex = pos;
    const m = r.re.exec(text) as RegExpIndices | null;
    if (!m) break;
    const mStart = m.index;
    const mEnd = m.index + m[0].length;
    let accept = true;
    if (mEnd === pos) {
      if (mStart === prevEnd) accept = false;
      pos += pos < text.length && text.codePointAt(pos)! > 0xffff ? 2 : 1;
    } else {
      pos = mEnd;
    }
    prevEnd = mEnd;
    if (!accept) continue;

    let start = mStart;
    let end = mEnd;
    if (r.group > 0) {
      const span = m.indices?.[r.group];
      if (!span) continue;
      [start, end] = span;
    }
    const score = r.validate(text.slice(start, end));
    if (score <= 0) continue;
    out.push({ type: r.entity, start, end, score });
  }
  return out;
}

function resolveOverlaps(entities: Entity[]): Entity[] {
  if (entities.length <= 1) return entities;
  const sorted = [...entities].sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score;
    const la = a.end - a.start;
    const lb = b.end - b.start;
    if (la !== lb) return lb - la;
    return a.start - b.start;
  });
  const kept: Entity[] = [];
  for (const e of sorted) {
    if (!kept.some((k) => e.start < k.end && k.start < e.end)) kept.push(e);
  }
  return kept.sort((a, b) => a.start - b.start);
}

function maskValue(value: string, op: Operator): string {
  const mc = op.masking_char ? Array.from(op.masking_char)[0] : '*';
  const chars = Array.from(value);
  const n = chars.length;
  const toMask = op.chars_to_mask ?? 0;
  if (toMask <= 0 || toMask >= n) return mc.repeat(n);
  if (op.from_end) return chars.slice(0, n - toMask).join('') + mc.repeat(toMask);
  return mc.repeat(toMask) + chars.slice(toMask).join('');
}

function hashValue(value: string, op: Operator): string {
  const h = crypto.createHash('sha256').update((op.salt ?? '') + value, 'utf8').digest('hex');
  const len = op.hash_len ?? 0;
  return len > 0 && len < h.length ? h.slice(0, len) : h;
}

function applyOperator(op: Operator, value: string, entityType: string): string {
  switch (op.type) {
    case 'redact':
      return '';
    case 'mask':
      return maskValue(value, op);
    case 'hash':
      return hashValue(value, op);
    case 'replace':
    case '':
      return op.new_value ? op.new_value : `<${entityType}>`;
    default:
      return `<${entityType}>`;
  }
}

const CUSTOM_NAME_RE = /^CUSTOM_[A-Z0-9_]{1,48}$/;

const POSIX_CLASSES: Record<string, string> = {
  alnum: '0-9A-Za-z',
  alpha: 'A-Za-z',
  ascii: String.raw`\x00-\x7F`,
  blank: String.raw`\t `,
  cntrl: String.raw`\x00-\x1F\x7F`,
  digit: '0-9',
  graph: '!-~',
  lower: 'a-z',
  print: ' -~',
  punct: String.raw`!-\/:-@\[-\x60{-~`,
  space: String.raw`\t\n\v\f\r `,
  upper: 'A-Z',
  word: '0-9A-Za-z_',
  xdigit: '0-9A-Fa-f',
};

const JS_SYNTAX_CHARS = '^$\\.*+?()[]{}|/';

function escapeLiteral(c: string): string {
  return /[A-Za-z0-9_\s]/.test(c) || c.charCodeAt(0) > 0x7f ? c : '\\x' + c.charCodeAt(0).toString(16).padStart(2, '0');
}

/**
 * Translates an RE2 pattern into a JavaScript RegExp source and flags, or
 * returns null when it cannot be translated faithfully.
 *
 * A repeated group that itself repeats something is refused: RE2 runs it in
 * linear time, a backtracking engine can take exponential time on it, and the
 * SDK runs in the caller's process.
 */
export function translateRE2(pattern: string): { source: string; flags: string } | null {
  let flags = '';
  let i = 0;
  for (let lead = /^\(\?([a-zA-Z]+)\)/.exec(pattern); lead; lead = /^\(\?([a-zA-Z]+)\)/.exec(pattern.slice(i))) {
    for (const f of lead[1]) {
      if (f !== 'i' && f !== 's' && f !== 'm') return null;
      if (!flags.includes(f)) flags += f;
    }
    i += lead[0].length;
  }

  let unicode = false;
  let out = '';
  const groups: boolean[] = [];
  let prevGroupRepeats = false;
  let inClass = false;

  const markRepeat = (): void => {
    if (groups.length > 0) groups[groups.length - 1] = true;
  };

  const escape = (at: number, cls: boolean): [string, number] | null => {
    const c = pattern[at + 1];
    if (c === undefined) return null;
    if (c === 'A' && !cls) return [String.raw`(?<![\s\S])`, 2];
    if (c === 'z' && !cls) return [String.raw`(?![\s\S])`, 2];
    if (c === 'C' || c === 'Z') return null;
    if (c === 's') return [cls ? GO_SPACE : `[${GO_SPACE}]`, 2];
    if (c === 'S' && !cls) return [`[^${GO_SPACE}]`, 2];
    if (c === 'p' || c === 'P') {
      unicode = true;
      let negate = c === 'P';
      let name: string;
      let used: number;
      if (pattern[at + 2] === '{') {
        const close = pattern.indexOf('}', at + 3);
        if (close < 0) return null;
        name = pattern.slice(at + 3, close);
        used = close - at + 1;
      } else {
        name = pattern[at + 2] ?? '';
        used = 3;
      }
      if (name.startsWith('^')) {
        negate = !negate;
        name = name.slice(1);
      }
      if (!/^[A-Za-z_]+$/.test(name)) return null;
      const prop = /^[A-Z][a-z]?$/.test(name) || name === 'Any' ? name : `Script=${name}`;
      return [`\\${negate ? 'P' : 'p'}{${prop}}`, used];
    }
    if (c === 'x' && pattern[at + 2] === '{') {
      const close = pattern.indexOf('}', at + 3);
      if (close < 0) return null;
      const code = parseInt(pattern.slice(at + 3, close), 16);
      if (!Number.isFinite(code) || code > 0x10ffff) return null;
      if (code > 0xffff) unicode = true;
      return [code > 0xffff ? `\\u{${code.toString(16)}}` : '\\u' + code.toString(16).padStart(4, '0'), close - at + 1];
    }
    if (c === 'Q') {
      const close = pattern.indexOf('\\E', at + 2);
      const lit = pattern.slice(at + 2, close < 0 ? pattern.length : close);
      const body = Array.from(lit).map(escapeLiteral).join('');
      return [cls || lit.length <= 1 ? body : `(?:${body})`, (close < 0 ? pattern.length : close + 2) - at];
    }
    if (/[A-Za-z0-9]/.test(c)) return [`\\${c}`, 2];
    return [escapeLiteral(c), 2];
  };

  while (i < pattern.length) {
    const c = pattern[i];

    if (inClass) {
      if (c === ']') {
        out += ']';
        inClass = false;
        i++;
      } else if (c === '[' && pattern[i + 1] === ':') {
        const close = pattern.indexOf(':]', i + 2);
        if (close < 0) return null;
        const expansion = POSIX_CLASSES[pattern.slice(i + 2, close)];
        if (expansion === undefined) return null;
        out += expansion;
        i = close + 2;
      } else if (c === '\\') {
        const e = escape(i, true);
        if (!e) return null;
        out += e[0];
        i += e[1];
      } else if (c === '[') {
        out += '\\[';
        i++;
      } else {
        out += c;
        i++;
      }
      continue;
    }

    let quantifier: { text: string; repeats: boolean } | null = null;
    if (c === '*' || c === '+') {
      quantifier = { text: c, repeats: true };
    } else if (c === '?') {
      quantifier = { text: c, repeats: false };
    } else if (c === '{') {
      const q = /^\{(\d+)(,(\d*))?\}/.exec(pattern.slice(i));
      if (q) {
        const upper = q[2] === undefined ? Number(q[1]) : q[3] === '' ? Infinity : Number(q[3]);
        quantifier = { text: q[0], repeats: upper > 1 };
      }
    }
    if (quantifier) {
      if (quantifier.repeats && prevGroupRepeats) return null;
      if (quantifier.repeats) markRepeat();
      out += quantifier.text;
      i += quantifier.text.length;
      if (pattern[i] === '?') {
        out += '?';
        i++;
      }
      prevGroupRepeats = false;
      continue;
    }

    prevGroupRepeats = false;
    if (c === '(') {
      groups.push(false);
      if (pattern.startsWith('(?P<', i)) {
        out += '(?<';
        i += 4;
      } else if (pattern[i + 1] === '?') {
        out += '(?';
        i += 2;
      } else {
        out += '(';
        i++;
      }
    } else if (c === ')') {
      const repeats = groups.pop();
      if (repeats === undefined) return null;
      if (repeats) markRepeat();
      prevGroupRepeats = repeats;
      out += ')';
      i++;
    } else if (c === '[') {
      inClass = true;
      out += '[';
      i++;
      if (pattern[i] === '^') {
        out += '^';
        i++;
      }
      // RE2 reads a leading ] as a literal; JavaScript reads [] as an empty class.
      if (pattern[i] === ']') {
        out += '\\]';
        i++;
      }
    } else if (c === '\\') {
      const e = escape(i, false);
      if (!e) return null;
      out += e[0];
      i += e[1];
    } else if (c === '.') {
      out += flags.includes('s') ? '.' : '[^\\n]';
      i++;
    } else if (c === '{' || c === '}') {
      out += '\\' + c;
      i++;
    } else {
      out += c;
      i++;
    }
  }
  if (inClass || groups.length > 0) return null;

  if (unicode) {
    // The u flag rejects identity escapes such as \" that RE2 accepts.
    out = out.replace(/\\([^A-Za-z0-9])/g, (whole, ch: string) =>
      JS_SYNTAX_CHARS.includes(ch) || ch === '-' ? whole : escapeLiteral(ch),
    );
    flags += 'u';
  }
  return { source: out, flags };
}

/**
 * Compiles customer recognizers. A row that fails is skipped and logged by
 * entity type only, so one bad row never disables the others. Shadow rows are
 * skipped because a shadow recognizer is measured, never allowed to transform.
 */
export function compileCustom(specs: CustomSpec[]): Recognizer[] {
  const out: Recognizer[] = [];
  const seen = new Set<string>();
  for (const s of specs) {
    if (s.Shadow) continue;
    const name = typeof s.EntityType === 'string' && CUSTOM_NAME_RE.test(s.EntityType) ? s.EntityType : null;
    const reason = customSpecProblem(s, name, seen);
    if (reason || !name) {
      console.warn(`Golem SDK: custom redaction pattern ${name ?? '(invalid name)'} skipped: ${reason}`);
      continue;
    }
    const translated = translateRE2(s.Pattern);
    if (!translated) {
      console.warn(`Golem SDK: custom redaction pattern ${name} skipped: not supported outside RE2`);
      continue;
    }
    const group = s.Group ?? 0;
    let re: RegExp;
    let groupCount: number;
    try {
      re = new RegExp(translated.source, 'g' + (group > 0 ? 'd' : '') + translated.flags);
      groupCount = new RegExp(translated.source + '|', translated.flags).exec('')!.length - 1;
    } catch (err) {
      console.warn(`Golem SDK: custom redaction pattern ${name} skipped: ${(err as Error).constructor.name}`);
      continue;
    }
    if (group > groupCount) {
      console.warn(`Golem SDK: custom redaction pattern ${name} skipped: capture group out of range`);
      continue;
    }
    seen.add(name);
    out.push({ entity: name, re, group, validate: alwaysValid(s.Score) });
  }
  return out;
}

function customSpecProblem(s: CustomSpec, name: string | null, seen: Set<string>): string | null {
  if (!name) return 'name must match CUSTOM_[A-Z0-9_]';
  if ([...builtinByName.keys()].some((k) => k.toLowerCase() === name.toLowerCase())) return 'built-in name';
  if (seen.has(name)) return 'duplicate entity type';
  if (typeof s.Pattern !== 'string') return 'missing pattern';
  const bytes = Buffer.byteLength(s.Pattern, 'utf8');
  if (bytes < 3 || bytes > 1024) return 'pattern length outside 3..1024';
  const group = s.Group ?? 0;
  if (!Number.isInteger(group) || group < 0 || group > 9) return 'capture group outside 0..9';
  if (typeof s.Score !== 'number' || !(s.Score > 0 && s.Score <= 1)) return 'score outside (0,1]';
  return null;
}

export class Redactor {
  private readonly recognizers: Recognizer[];
  private readonly operator: Operator;

  constructor(types: string[] | null | undefined, operator: Operator, custom: CustomSpec[] = []) {
    this.operator = operator;
    const customByName = new Map(compileCustom(custom).map((r) => [r.entity, r]));
    if (!types || types.length === 0) {
      this.recognizers = structuredRecognizers;
      return;
    }
    // Any other name (PERSON, STREET_ADDRESS, ...) is a model label, and the SDK carries no model.
    this.recognizers = types
      .map((t) => builtinByName.get(t) ?? customByName.get(t))
      .filter((r): r is Recognizer => r !== undefined);
  }

  /**
   * `context` is scanned but never returned: prefixing a value with its key
   * lets `password: <value>` be found in a value that alone says nothing.
   */
  redact(text: string, context = ''): RedactResult {
    const off = context.length;
    let found: Entity[] = [];
    for (const r of this.recognizers) found.push(...find(r, context + text));
    found = resolveOverlaps(found.filter((e) => e.score >= DEFAULT_THRESHOLD));
    const entities =
      off === 0
        ? found
        : found.filter((e) => e.end > off).map((e) => ({ ...e, start: Math.max(e.start, off) - off, end: e.end - off }));

    const counts: Record<string, number> = {};
    for (const e of entities) counts[e.type] = (counts[e.type] ?? 0) + 1;

    let out = text;
    for (let k = entities.length - 1; k >= 0; k--) {
      const e = entities[k];
      out = out.slice(0, e.start) + applyOperator(this.operator, out.slice(e.start, e.end), e.type) + out.slice(e.end);
    }
    return { text: out, counts, found: entities.map((e) => e.type) };
  }
}

export function redact(
  text: string,
  types: string[] | null | undefined,
  operator: Operator,
  custom: CustomSpec[] = [],
): RedactResult {
  return new Redactor(types, operator, custom).redact(text);
}

function asString(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined;
}

function asInt(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isInteger(v) ? v : undefined;
}

export function parseOperator(raw: unknown): Operator {
  const o = raw !== null && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
  const op: Operator = { type: asString(o.type) ?? '' };
  const nv = asString(o.new_value);
  if (nv !== undefined) op.new_value = nv;
  const mc = asString(o.masking_char);
  if (mc !== undefined) op.masking_char = mc;
  const ctm = asInt(o.chars_to_mask);
  if (ctm !== undefined) op.chars_to_mask = ctm;
  if (typeof o.from_end === 'boolean') op.from_end = o.from_end;
  const hl = asInt(o.hash_len);
  if (hl !== undefined) op.hash_len = hl;
  const salt = asString(o.salt);
  if (salt !== undefined) op.salt = salt;
  return op;
}

/** Throws on a body that is not a policy, so "no policy" stays distinct from "redaction is off". */
export function parseRedactionPolicy(raw: unknown): RedactionPolicy {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new TypeError('redaction policy is not an object');
  }
  const o = raw as Record<string, unknown>;
  if (typeof o.enabled !== 'boolean') {
    throw new TypeError('redaction policy has no enabled flag');
  }
  const types = o.entity_types ?? [];
  if (!Array.isArray(types) || !types.every((t) => typeof t === 'string')) {
    throw new TypeError('redaction policy entity_types is not a list of names');
  }
  const custom = o.custom ?? [];
  if (!Array.isArray(custom)) {
    throw new TypeError('redaction policy custom is not a list');
  }
  return {
    enabled: o.enabled,
    entity_types: types as string[],
    operator: parseOperator(o.operator),
    custom: custom
      .filter((c): c is Record<string, unknown> => c !== null && typeof c === 'object')
      .map((c) => ({
        EntityType: asString(c.EntityType) ?? '',
        Pattern: asString(c.Pattern) ?? '',
        Group: asInt(c.Group) ?? 0,
        Score: typeof c.Score === 'number' ? c.Score : 0,
        Shadow: c.Shadow === true,
      })),
  };
}

/**
 * This machine's policy, kept for 30 s. After a failed fetch the last policy
 * stays in use; with none ever fetched, null is returned and a fetch is retried
 * after 5 s. Concurrent callers share one fetch.
 */
export class RedactionPolicyCache {
  private policy: RedactionPolicy | null = null;
  private redactor: Redactor | null = null;
  private fetchedAt = Number.NEGATIVE_INFINITY;
  private failedAt = Number.NEGATIVE_INFINITY;
  private failing = false;
  private inflight: Promise<void> | null = null;

  constructor(
    private readonly fetch: () => Promise<unknown>,
    private readonly ttlMs = 30_000,
    private readonly retryMs = 5_000,
    private readonly now: () => number = Date.now,
  ) {}

  async current(): Promise<{ policy: RedactionPolicy; redactor: Redactor } | null> {
    const t = this.now();
    const stale = t - this.fetchedAt >= this.ttlMs;
    const mayRetry = t - this.failedAt >= this.retryMs;
    if ((this.policy === null || stale) && mayRetry) {
      this.inflight ??= this.refresh().finally(() => {
        this.inflight = null;
      });
      await this.inflight;
    }
    return this.policy && this.redactor ? { policy: this.policy, redactor: this.redactor } : null;
  }

  private async refresh(): Promise<void> {
    try {
      const policy = parseRedactionPolicy(await this.fetch());
      if (JSON.stringify(policy) !== JSON.stringify(this.policy)) {
        this.redactor = new Redactor(policy.entity_types, policy.operator, policy.custom);
      }
      this.policy = policy;
      this.fetchedAt = this.now();
      this.failing = false;
    } catch (e) {
      if (!this.failing) {
        console.warn(`GolemSDK: redaction policy unavailable (${(e as Error)?.name ?? 'Error'})`);
      }
      this.failing = true;
      this.failedAt = this.now();
      if (this.policy !== null) this.fetchedAt = this.now();
    }
  }
}

/** Redacts every string in a value, each scanned with its key as context. */
export function redactValue(redactor: Redactor, value: unknown, counts: Record<string, number>, key = ''): unknown {
  if (typeof value === 'string') {
    const r = redactor.redact(value, key ? `"${key}":"` : '');
    for (const [t, n] of Object.entries(r.counts)) counts[t] = (counts[t] ?? 0) + n;
    return r.text;
  }
  if (Array.isArray(value)) return value.map((v) => redactValue(redactor, v, counts, key));
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) out[k] = redactValue(redactor, v, counts, k);
    return out;
  }
  return value;
}

/** Counts as the Golem agent records them: compact JSON, keys sorted. */
export function countsMetadata(counts: Record<string, number>): string {
  const sorted: Record<string, number> = {};
  for (const k of Object.keys(counts).sort()) sorted[k] = counts[k];
  return JSON.stringify(sorted);
}
