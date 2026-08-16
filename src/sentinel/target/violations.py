"""Deterministic parsing and severity classification of violation text.

The `violations` column is semi-structured. Findings §6 measured the format over
497,208 code-era entries: 100% begin with ``NN.`` and 99.8% contain
``- Comments:``, so the shape is

    NN. UPPERCASE VIOLATION TITLE - Comments: free text   |   NN. ...

Two design decisions here are the opposite of the obvious ones, and both are
measured rather than assumed:

**The violation number does not classify severity.** Findings §7.1: item 10 is
42% Priority Foundation, 11% Priority and 47% unlabelled, because the same
numbered item covers "no hot water at the hand sink" (Priority) and "no
hand-washing sign" (not). The number is parsed and kept for audit only.

**A municipal code is not required.** Findings §7.3: 21,281 entries carry a
PRIORITY marker with no ``7-38-xxx`` code, and reading them shows they are
genuine — ``PRIORITY FOUNDATION VIOLATION. NO CITATION ISSUED.`` is a real
violation that was simply not cited. Requiring a code would trade ~21,281 false
negatives for a few hundred false positives.

So classification is marker-based, with a small, enumerated list of narrative
contexts excluded. No LLM, no embeddings, no fuzzy matching: every decision is a
regular expression a reviewer can argue with.
"""

from __future__ import annotations

import re

from sentinel.target.models import Severity, ViolationEntry

# Entries are separated by a pipe. Findings §6: splitting on it is safe because
# every resulting entry is numbered and none is empty.
ENTRY_SEPARATOR = "|"

# Sentence-ish boundaries used to isolate narrative clauses. Deliberately not a
# real sentence tokenizer: the text is uppercase, abbreviation-heavy and
# inconsistently punctuated, so anything cleverer would be less predictable.
_CHUNK_SPLIT = re.compile(r"[.;]")

_ENTRY_NUMBER = re.compile(r"^\s*(\d+)\s*\.")
_COMMENTS_SPLIT = re.compile(r"-\s*COMMENTS\s*:", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

# Narrative contexts in which a PRIORITY marker is not a classification of the
# violation being recorded. Each pattern is justified by a real example in
# findings §7.4, and the combined effect is measured there: 74 entries and 10
# inspection labels out of 137,598 and 58,427 respectively.
NARRATIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    # "A 90 day grace period was given for all new priority and priority
    # foundation violations" -- standard notice attached to unrelated entries.
    ("grace_period_days", r"\d+\s*-?\s*DAY\s+GRACE\s+PERIOD"),
    ("grace_period", r"GRACE\s+PERIOD"),
    # "...OR CITATION PRIORITY FOUNDATION WILL BE ISSUED" -- a warning about a
    # citation that has not happened.
    ("future_citation", r"WILL\s+BE\s+ISSUED"),
    # "NO PRIORITY FOUNDATION VIOLATION 7-38-030(c)" -- explicit negation.
    ("negation", r"\bNO\s+PRIORITY"),
)

_NARRATIVE = re.compile("|".join(f"(?:{p})" for _, p in NARRATIVE_PATTERNS))

# Priority Foundation must be tested before bare Priority, because it contains
# it. Tolerates the observed run-together typo (PRIORITY VIOLATIONI) and any
# amount of internal whitespace.
_PRIORITY_FOUNDATION = re.compile(r"PRIORITY\s+FOUNDATION")
_PRIORITY = re.compile(r"PRIORITY")

# How much surrounding text to keep as evidence for a positive classification.
_EVIDENCE_WINDOW = 60


def split_violations(raw: str | None) -> list[str]:
    """Split the raw violations blob into trimmed, non-empty entries."""
    if raw is None:
        return []
    return [part.strip() for part in raw.split(ENTRY_SEPARATOR) if part.strip()]


def _narrative_free_chunks(text: str) -> list[str]:
    """Split into chunks and drop those that are narrative rather than a finding.

    Span-based rather than entry-based, and that distinction matters. Findings
    §7.4 shows a genuine Priority Foundation citation co-occurring with
    ``A CITATION WILL BE ISSUED IF NOT CORRECTED``; dropping the whole entry
    would lose a real positive, while dropping only the offending clause leaves
    the citation intact.
    """
    return [chunk for chunk in _CHUNK_SPLIT.split(text) if not _NARRATIVE.search(chunk)]


def classify(text: str) -> tuple[Severity, str | None]:
    """Classify one violation entry, returning its severity and the evidence span.

    Returns ``UNCLASSIFIED`` when no marker survives narrative exclusion. That is
    not a claim the violation is Core (findings §7.2) -- only that no severity
    was recorded.
    """
    upper = text.upper()
    chunks = _narrative_free_chunks(upper)

    for chunk in chunks:
        match = _PRIORITY_FOUNDATION.search(chunk)
        if match is not None:
            return Severity.PRIORITY_FOUNDATION, _evidence(chunk, match)

    for chunk in chunks:
        match = _PRIORITY.search(chunk)
        if match is not None:
            return Severity.PRIORITY, _evidence(chunk, match)

    return Severity.UNCLASSIFIED, None


def _evidence(chunk: str, match: re.Match[str]) -> str:
    """Return a trimmed window around the matched marker, for audit."""
    start = max(0, match.start() - _EVIDENCE_WINDOW // 2)
    end = min(len(chunk), match.end() + _EVIDENCE_WINDOW)
    return _WHITESPACE.sub(" ", chunk[start:end]).strip()


def parse_entry(text: str) -> ViolationEntry:
    """Parse a single violation entry.

    A malformed entry is kept, not discarded: it becomes an ``UNCLASSIFIED``
    entry with ``number=None`` and the whole text as its title. Silently
    dropping unparseable text would hide exactly the cases worth knowing about.
    """
    stripped = text.strip()

    number: int | None = None
    number_match = _ENTRY_NUMBER.match(stripped)
    body = stripped
    if number_match is not None:
        number = int(number_match.group(1))
        body = stripped[number_match.end() :].strip()

    parts = _COMMENTS_SPLIT.split(body, maxsplit=1)
    if len(parts) == 2:
        title, comments = parts[0].strip(), parts[1].strip()
    else:
        title, comments = body, ""

    severity, evidence = classify(stripped)
    return ViolationEntry(
        number=number,
        title=_WHITESPACE.sub(" ", title),
        comments=_WHITESPACE.sub(" ", comments),
        severity=severity,
        evidence=evidence,
    )


def parse_violations(raw: str | None) -> list[ViolationEntry]:
    """Parse the full violations blob into classified entries."""
    return [parse_entry(entry) for entry in split_violations(raw)]


def has_priority_violation(entries: list[ViolationEntry]) -> bool:
    """Whether any entry is a Priority or Priority Foundation violation."""
    return any(entry.is_priority for entry in entries)


def first_priority_evidence(entries: list[ViolationEntry]) -> str | None:
    """The evidence span of the first priority entry, preferring Foundation.

    Deterministic: Priority Foundation entries are checked before plain Priority
    ones, and within each the first in document order wins.
    """
    for severity in (Severity.PRIORITY_FOUNDATION, Severity.PRIORITY):
        for entry in entries:
            if entry.severity is severity and entry.evidence:
                return entry.evidence
    return None
