"""Redaction before signing.

The Golem agent cannot redact an SDK event: the SDK commits the payload hash
and signs it before the agent ever sees it, so rewriting the payload would break
the signature. The SDK therefore removes what this machine's redaction policy
names itself, before it encodes and signs.

This is a port of ``pkg/redaction`` (Go). Both are held to the same golden
vectors in ``pkg/redaction/testdata/vectors.json``. Only the pattern-based types
are found here; types that need a model (names, street addresses) are skipped.
"""

import bisect
import hashlib
import json
import logging
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger("golem_sdk.redaction")

THRESHOLD = 0.5


class Entity(NamedTuple):
    """One detected span. Offsets are str indices; never carries the value."""

    type: str
    start: int
    end: int
    score: float
    byte_len: int


@dataclass(frozen=True)
class Operator:
    type: str = ""
    new_value: str = ""
    masking_char: str = ""
    chars_to_mask: int = 0
    from_end: bool = False
    hash_len: int = 0
    salt: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "Operator":
        if not isinstance(d, dict):
            return cls()
        return cls(
            type=_str(d.get("type")),
            new_value=_str(d.get("new_value")),
            masking_char=_str(d.get("masking_char")),
            chars_to_mask=_int(d.get("chars_to_mask")),
            from_end=d.get("from_end") is True,
            hash_len=_int(d.get("hash_len")),
            salt=_str(d.get("salt")),
        )

    def apply(self, value: str, entity_type: str) -> str:
        if self.type == "redact":
            return ""
        if self.type == "mask":
            return _mask(value, self)
        if self.type == "hash":
            return _hash(value, self)
        if self.type in ("replace", "") and self.new_value:
            return self.new_value
        return "<" + entity_type + ">"


@dataclass(frozen=True)
class CustomSpec:
    entity_type: str
    pattern: str
    group: int = 0
    score: float = 0.0
    shadow: bool = False

    @classmethod
    def from_dict(cls, d: Any) -> Optional["CustomSpec"]:
        # The Go struct has no json tags, so the wire names are its field names.
        if not isinstance(d, dict):
            return None
        score = d.get("Score")
        return cls(
            entity_type=_str(d.get("EntityType")),
            pattern=_str(d.get("Pattern")),
            group=_int(d.get("Group")),
            score=float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
            shadow=d.get("Shadow") is True,
        )


def _str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _int(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


def _mask(value: str, op: Operator) -> str:
    mc = op.masking_char[0] if op.masking_char else "*"
    n = len(value)
    k = op.chars_to_mask
    if k <= 0 or k >= n:
        return mc * n
    if op.from_end:
        return value[: n - k] + mc * k
    return mc * k + value[k:]


def _hash(value: str, op: Operator) -> str:
    h = hashlib.sha256((op.salt + value).encode("utf-8", "surrogatepass")).hexdigest()
    if 0 < op.hash_len < len(h):
        h = h[: op.hash_len]
    return h


def _utf8_len(s: str) -> int:
    return len(s.encode("utf-8", "surrogatepass"))


class _Recognizer:
    __slots__ = ("entity", "regex", "group", "validate")

    def __init__(self, entity: str, regex: "re.Pattern[str]", validate: Callable[[str], float], group: int = 0):
        self.entity = entity
        self.regex = regex
        self.group = group
        self.validate = validate

    def find(self, text: str) -> List[Entity]:
        out = []
        for m in self.regex.finditer(text):
            start, end = m.span(self.group)
            if start < 0:
                continue
            value = text[start:end]
            score = self.validate(value)
            if score <= 0:
                continue
            out.append(Entity(self.entity, start, end, score, _utf8_len(value)))
        return out


def _always(score: float) -> Callable[[str], float]:
    return lambda _m: score


def _digits(s: str) -> str:
    return "".join(c for c in s if "0" <= c <= "9")


def _validate_ssn(m: str) -> float:
    d = _digits(m)
    if len(d) != 9:
        return 0
    area, group, serial = d[0:3], d[3:5], d[5:9]
    if area in ("000", "666") or area[0] == "9":
        return 0
    if group == "00" or serial == "0000":
        return 0
    return 0.85


def _luhn_valid(s: str) -> bool:
    total = 0
    alt = False
    digits = 0
    for c in reversed(s):
        if not "0" <= c <= "9":
            continue
        digits += 1
        d = ord(c) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return digits > 0 and total % 10 == 0


def _validate_card(m: str) -> float:
    n = len(_digits(m))
    if n < 13 or n > 19 or not _luhn_valid(m):
        return 0
    return 0.95


def _validate_phone(m: str) -> float:
    n = len(_digits(m))
    return 0.6 if 10 <= n <= 11 else 0


def _validate_ipv4(m: str) -> float:
    parts = m.split(".")
    if len(parts) != 4 or any(int(p) > 255 for p in parts):
        return 0
    return 0.6


def _validate_iban(m: str) -> float:
    if len(m) < 15 or len(m) > 34:
        return 0
    rem = 0
    for c in m[4:] + m[:4]:
        if "0" <= c <= "9":
            rem = (rem * 10 + ord(c) - 48) % 97
        elif "A" <= c <= "Z":
            rem = (rem * 100 + ord(c) - 55) % 97
        else:
            return 0
    return 0.8 if rem == 1 else 0


_PLACEHOLDERS = ("example", "changeme", "your_", "your-", "placeholder", "redacted", "xxxx", "<",
                 "insert_", "dummy", "sample", "test_token", "fake")


def _reject_placeholder(score: float) -> Callable[[str], float]:
    def validate(m: str) -> float:
        lo = m.lower()
        if any(bad in lo for bad in _PLACEHOLDERS):
            return 0
        # Go compares a count of the first BYTE against the byte length, which
        # only ever holds for a run of one ASCII character.
        if lo and ord(lo[0]) < 0x80 and lo.count(lo[0]) == len(lo):
            return 0
        return score

    return validate


# Go's \d, \w, \b and \s are ASCII-only; re.ASCII gives the first three. \s is
# spelled out because Python's ASCII \s also matches \v, which Go's does not.
# The case-insensitive recognizers are compiled without re.ASCII because Go's
# (?i) folds Unicode (k matches the Kelvin sign), so their \b is written out.
_A = re.ASCII
_I = re.IGNORECASE
_SP = r"[\t\n\f\r ]"
_WORD_START = r"(?-i:(?<![0-9A-Za-z_]))"

_STRUCTURED: List[_Recognizer] = [
    _Recognizer("EMAIL_ADDRESS", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", _A), _always(1.0)),
    _Recognizer("US_SSN", re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", _A), _validate_ssn),
    _Recognizer("CREDIT_CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b", _A), _validate_card),
    _Recognizer("PHONE_NUMBER", re.compile(r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b", _A), _validate_phone),
    _Recognizer("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", _A), _validate_ipv4),
    _Recognizer("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", _A), _validate_iban),
    _Recognizer("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", _A), _always(0.9)),
    _Recognizer("CRYPTO", re.compile(r"\b0x[a-fA-F0-9]{40}\b", _A), _always(0.7)),
]

_SECRETS: List[_Recognizer] = [
    _Recognizer("GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})", _A), _always(0.95)),
    _Recognizer("GITLAB_TOKEN", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}", _A), _always(0.95)),
    _Recognizer("SLACK_TOKEN", re.compile(r"\b(?:xox[abprs]-[A-Za-z0-9\-]{10,}|xapp-\d-[A-Za-z0-9\-]{10,})", _A), _always(0.95)),
    _Recognizer("ANTHROPIC_API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}", _A), _always(0.95)),
    _Recognizer("OPENAI_API_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b", _A), _always(0.95)),
    _Recognizer("GOOGLE_API_KEY", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}", _A), _always(0.95)),
    _Recognizer("STRIPE_KEY", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}", _A), _always(0.95)),
    _Recognizer("NPM_TOKEN", re.compile(r"\bnpm_[A-Za-z0-9]{36}", _A), _always(0.95)),
    _Recognizer("HUGGINGFACE_TOKEN", re.compile(r"\bhf_[A-Za-z0-9]{30,}", _A), _always(0.9)),
    _Recognizer("PRIVATE_KEY", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", _A | re.DOTALL), _always(1.0)),
    _Recognizer("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}", _A), _always(0.9)),
    _Recognizer("BEARER_TOKEN", re.compile(r"bearer" + _SP + r"+([A-Za-z0-9._~+/\-]{16,}=*)", _I),
                _reject_placeholder(0.85), group=1),
    _Recognizer("DB_PASSWORD", re.compile(
        _WORD_START + r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?)://[^:@/\t\n\f\r ]+:([^@/\t\n\f\r ]+)@", _I),
        _reject_placeholder(0.9), group=1),
    _Recognizer("GOLEM_KEY", re.compile(r"\b(?:gla|glk)_[A-Za-z0-9_\-]{22,}", _A), _always(1.0)),
    _Recognizer("GENERIC_SECRET", re.compile(
        r"(?:api[_\-]?key|secret|token|password|passwd|access[_\-]?key)[\"']?" + _SP + r"*[:=]" + _SP
        + r"*[\"']?([A-Za-z0-9/+=_.\-]{16,})", _I),
        _reject_placeholder(0.6), group=1),
]

_BUILTIN: Dict[str, _Recognizer] = {r.entity: r for r in _STRUCTURED + _SECRETS}


def structured_types() -> List[str]:
    return [r.entity for r in _STRUCTURED]


def secret_types() -> List[str]:
    return [r.entity for r in _SECRETS]


_CUSTOM_NAME = re.compile(r"^CUSTOM_[A-Z0-9_]{1,48}$")

_POSIX = {
    "alnum": "0-9A-Za-z", "alpha": "A-Za-z", "ascii": "\\x00-\\x7F", "blank": "\\t ",
    "cntrl": "\\x00-\\x1F\\x7F", "digit": "0-9", "graph": "!-~", "lower": "a-z",
    "print": " -~", "punct": "!-/:-@\\[-`{-~", "space": "\\t\\n\\v\\f\\r ",
    "upper": "A-Z", "word": "0-9A-Za-z_", "xdigit": "0-9A-Fa-f",
}


def _translate_re2(p: str) -> str:
    """Rewrite the RE2 spellings Python reads differently or not at all.

    A construct Python would silently read another way is rewritten; one it
    cannot express raises, so the pattern is skipped rather than half-applied.
    """
    out = []
    i, n = 0, len(p)
    in_class = False
    while i < n:
        c = p[i]
        if c == "\\" and i + 1 < n:
            nx = p[i + 1]
            if nx == "Q":
                j = p.find("\\E", i + 2)
                out.append(re.escape(p[i + 2:] if j < 0 else p[i + 2:j]))
                i = n if j < 0 else j + 2
                continue
            if nx == "x" and i + 2 < n and p[i + 2] == "{":
                j = p.find("}", i + 3)
                if j < 0:
                    raise ValueError("unterminated \\x{")
                out.append("\\U%08x" % int(p[i + 3:j], 16))
                i = j + 1
                continue
            out.append("\\Z" if nx == "z" else p[i:i + 2])
            i += 2
            continue
        if not in_class:
            out.append(c)
            i += 1
            if c == "[":
                in_class = True
                if p.startswith("^", i):
                    out.append("^")
                    i += 1
                if p.startswith("]", i):
                    out.append("\\]")
                    i += 1
            continue
        if p.startswith("[:", i):
            j = p.find(":]", i + 2)
            if j > 0:
                name = p[i + 2:j]
                if name.startswith("^"):
                    raise ValueError("negated POSIX class")
                if name in _POSIX:
                    out.append(_POSIX[name])
                    i = j + 2
                    continue
        if c == "[":
            out.append("\\[")
        else:
            out.append(c)
            in_class = c != "]"
        i += 1
    return "".join(out)


def compile_custom(specs: Iterable[Any]) -> List[_Recognizer]:
    """Compile customer recognizers independently; a bad one never stops the rest.

    Logs only the entity type of a skipped spec, never its pattern.
    """
    out: List[_Recognizer] = []
    seen = set()
    reserved = {t.upper() for t in _BUILTIN}
    for raw in specs or ():
        s = raw if isinstance(raw, CustomSpec) else CustomSpec.from_dict(raw)
        if s is None:
            continue
        name = s.entity_type
        if (not _CUSTOM_NAME.match(name) or name.upper() in reserved
                or not 3 <= _utf8_len(s.pattern) <= 1024
                or not 0 <= s.group <= 9 or not 0 < s.score <= 1 or name in seen):
            logger.warning("skipping custom recognizer %s: invalid spec", name[:64])
            continue
        if s.shadow:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rx = re.compile(_translate_re2(s.pattern), re.ASCII)
        except (re.error, ValueError, OverflowError) as e:
            logger.warning("skipping custom recognizer %s: pattern does not compile (%s)", name, type(e).__name__)
            continue
        if s.group > rx.groups:
            logger.warning("skipping custom recognizer %s: capture group out of range", name)
            continue
        seen.add(name)
        out.append(_Recognizer(name, rx, _always(s.score), group=s.group))
    return out


def _route(types: Optional[Sequence[str]], custom: Sequence[_Recognizer]) -> List[_Recognizer]:
    if not types:
        return list(_STRUCTURED)
    by_name = {r.entity: r for r in custom}
    out = []
    for t in types:
        r = _BUILTIN.get(t) or by_name.get(t)
        if r is not None:
            out.append(r)
    return out


def _resolve_overlaps(entities: List[Entity]) -> List[Entity]:
    if len(entities) <= 1:
        return entities
    # Length ties break on UTF-8 bytes, as the Go engine measures them.
    ranked = sorted(entities, key=lambda e: (-e.score, -e.byte_len, e.start))
    starts: List[int] = []
    kept: List[Entity] = []
    for e in ranked:
        i = bisect.bisect_left(starts, e.start)
        if i > 0 and kept[i - 1].end > e.start:
            continue
        if i < len(kept) and kept[i].start < e.end:
            continue
        starts.insert(i, e.start)
        kept.insert(i, e)
    return kept


def _detect(text: str, recognizers: Sequence[_Recognizer]) -> List[Entity]:
    found = []
    for r in recognizers:
        found.extend(e for e in r.find(text) if e.score >= THRESHOLD)
    return _resolve_overlaps(found)


def _apply(text: str, entities: Sequence[Entity], op: Operator) -> str:
    parts = []
    pos = 0
    for e in entities:
        parts.append(text[pos:e.start])
        parts.append(op.apply(text[e.start:e.end], e.type))
        pos = e.end
    parts.append(text[pos:])
    return "".join(parts)


def detect(text: str, types: Optional[Sequence[str]] = None, custom: Iterable[Any] = ()) -> List[Entity]:
    """Find the named types in text, overlap-resolved, in document order.

    No types means every structured type (not secrets, not custom). A type that
    is neither built in nor a supplied custom recognizer needs a model and is
    skipped.
    """
    return _detect(text, _route(types, compile_custom(custom)))


def redact(
    text: str,
    types: Optional[Sequence[str]] = None,
    operator: Optional[Operator] = None,
    custom: Iterable[Any] = (),
) -> Tuple[str, Dict[str, int]]:
    """Return the redacted text and a count per entity type (never values)."""
    return _redact_with(text, _route(types, compile_custom(custom)), operator or Operator())


def _redact_with(text: str, recognizers: Sequence[_Recognizer], op: Operator) -> Tuple[str, Dict[str, int]]:
    entities = _detect(text, recognizers)
    counts: Dict[str, int] = {}
    for e in entities:
        counts[e.type] = counts.get(e.type, 0) + 1
    return _apply(text, entities, op), counts


@dataclass
class RedactionPolicy:
    """What this machine's Golem agent says an SDK must redact before signing."""

    enabled: bool = False
    entity_types: Tuple[str, ...] = ()
    operator: Operator = field(default_factory=Operator)
    custom: Tuple[CustomSpec, ...] = ()
    _recognizers: List[_Recognizer] = field(default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._recognizers = _route(self.entity_types, compile_custom(self.custom))

    @classmethod
    def from_dict(cls, d: Any) -> "RedactionPolicy":
        if not isinstance(d, dict):
            raise ValueError("redaction policy is not an object")
        types = d.get("entity_types") or []
        custom = d.get("custom") or []
        return cls(
            enabled=d.get("enabled") is True,
            entity_types=tuple(t for t in types if isinstance(t, str)) if isinstance(types, list) else (),
            operator=Operator.from_dict(d.get("operator")),
            custom=tuple(s for s in (CustomSpec.from_dict(c) for c in custom) if s is not None)
            if isinstance(custom, list) else (),
        )

    def redact_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        return _redact_with(text, self._recognizers, self.operator)

    def redact_value(self, value: Any, counts: Dict[str, int]) -> Any:
        """Redact every string inside value, keys included, adding to counts.

        Returns a new structure; the caller's objects are left as they were.
        """
        if isinstance(value, str):
            out, c = self.redact_text(value)
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
            return out
        if isinstance(value, dict):
            red: Dict[Any, Any] = {}
            for k, v in value.items():
                rk = self.redact_value(k, counts) if isinstance(k, str) else k
                base, i = rk, 2
                while rk in red:
                    rk = f"{base}#{i}"
                    i += 1
                red[rk] = self.redact_value(v, counts)
            return red
        if isinstance(value, (list, tuple)):
            return type(value)(self.redact_value(v, counts) for v in value)
        return value


def counts_metadata(counts: Dict[str, int]) -> str:
    """Encoded as the Golem agent encodes its own: compact JSON, sorted keys."""
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


class PolicyCache:
    """Holds the last policy fetched, refreshing it at most every ``ttl`` seconds.

    A failed fetch keeps the last good policy. Before any fetch has succeeded,
    retries come sooner (``retry``) so a Golem agent that starts after the SDK
    is picked up quickly.
    """

    def __init__(self, fetch: Callable[[], Any], ttl: float = 30.0, retry: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self._fetch = fetch
        self._ttl = ttl
        self._retry = retry
        self._clock = clock
        self._lock = threading.Lock()
        self._policy: Optional[RedactionPolicy] = None
        self._fetched = False
        self._next = 0.0

    def current(self) -> Optional[RedactionPolicy]:
        with self._lock:
            now = self._clock()
            if self._fetched and now < self._next:
                return self._policy
            self._fetched = True
            try:
                self._policy = RedactionPolicy.from_dict(self._fetch())
                self._next = now + self._ttl
            except Exception as e:
                logger.debug("redaction policy unavailable (%s)", type(e).__name__)
                self._next = now + (self._ttl if self._policy is not None else self._retry)
            return self._policy
