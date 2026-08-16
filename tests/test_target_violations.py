"""Violation parsing and severity classification.

The narrative-exclusion tests are the important ones. They are the only place
where the parser decides that text containing the word PRIORITY is *not* a
priority violation, so each exclusion has a test asserting it fires and — more
importantly — a test asserting it does not fire on a genuine citation that
happens to sit near the same words.
"""

from __future__ import annotations

import pytest

from sentinel.target.models import Severity
from sentinel.target.violations import (
    NARRATIVE_PATTERNS,
    classify,
    first_priority_evidence,
    has_priority_violation,
    parse_entry,
    parse_violations,
    split_violations,
)

# --- splitting -----------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "|", " | | "])
def test_empty_input_yields_no_entries(raw: str | None) -> None:
    assert split_violations(raw) == []


def test_single_entry() -> None:
    assert split_violations("3. SOMETHING - Comments: TEXT") == ["3. SOMETHING - Comments: TEXT"]


def test_multiple_entries_are_split_on_pipe() -> None:
    raw = "3. A - Comments: X | 10. B - Comments: Y | 55. C - Comments: Z"
    assert len(split_violations(raw)) == 3


def test_entries_are_trimmed_and_blanks_dropped() -> None:
    assert split_violations("  3. A  |   |  10. B ") == ["3. A", "10. B"]


# --- entry structure -----------------------------------------------------


def test_entry_number_is_parsed() -> None:
    assert parse_entry("22. PROPER COLD HOLDING TEMPERATURES - Comments: X").number == 22


def test_title_and_comments_are_separated() -> None:
    entry = parse_entry("3. MANAGEMENT AND REPORTING - Comments: NO POLICY ON SITE.")
    assert entry.title == "MANAGEMENT AND REPORTING"
    assert entry.comments == "NO POLICY ON SITE."


def test_entry_without_comments_section_keeps_its_title() -> None:
    """Findings §6: 1,098 code-era entries are a title with no observation."""
    entry = parse_entry("47. FOOD CONTACT SURFACES CLEANABLE")
    assert entry.title == "FOOD CONTACT SURFACES CLEANABLE"
    assert entry.comments == ""


def test_internal_whitespace_is_collapsed() -> None:
    entry = parse_entry("5. A   B    C - Comments:  X    Y ")
    assert entry.title == "A B C"
    assert entry.comments == "X Y"


def test_malformed_entry_is_kept_not_discarded() -> None:
    """Silently dropping unparseable text would hide the cases worth seeing."""
    entry = parse_entry("this has no number and no comments marker")
    assert entry.number is None
    assert entry.severity is Severity.UNCLASSIFIED
    assert entry.title == "this has no number and no comments marker"


def test_multi_digit_numbers_are_parsed() -> None:
    assert parse_entry("64. SOMETHING - Comments: X").number == 64


# --- classification ------------------------------------------------------


def test_priority_foundation_is_recognized() -> None:
    severity, evidence = classify("3. X - Comments: NO POLICY. PRIORITY FOUNDATION 7-38-010.")
    assert severity is Severity.PRIORITY_FOUNDATION
    assert evidence is not None and "PRIORITY FOUNDATION" in evidence


def test_plain_priority_is_recognized() -> None:
    severity, _ = classify("22. X - Comments: TEMP 49F. PRIORITY 7-38-005, CITATION ISSUED")
    assert severity is Severity.PRIORITY


def test_priority_foundation_wins_over_bare_priority() -> None:
    """PRIORITY FOUNDATION contains PRIORITY, so ordering matters."""
    severity, _ = classify("X - Comments: PRIORITY FOUNDATION 7-38-010")
    assert severity is Severity.PRIORITY_FOUNDATION


def test_unlabelled_entry_is_unclassified_not_core() -> None:
    """Findings §7.2: 72% of entries carry no label; that is absence of evidence."""
    severity, evidence = classify("55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR UNDER SINK.")
    assert severity is Severity.UNCLASSIFIED
    assert evidence is None


def test_classification_is_case_insensitive() -> None:
    lower, _ = classify("x - comments: priority foundation 7-38-010")
    upper, _ = classify("X - COMMENTS: PRIORITY FOUNDATION 7-38-010")
    assert lower is upper is Severity.PRIORITY_FOUNDATION


def test_marker_without_a_municipal_code_still_counts() -> None:
    """Findings §7.3: requiring a code would create ~21,281 false negatives."""
    severity, _ = classify(
        "1. PERSON IN CHARGE - Comments: NONE PRESENT. PRIORITY FOUNDATION VIOLATION. "
        "NO CITATION ISSUED."
    )
    assert severity is Severity.PRIORITY_FOUNDATION


def test_run_together_typo_is_tolerated() -> None:
    severity, _ = classify("X - Comments: PRIORITY VIOLATIONI 7-38-005")
    assert severity is Severity.PRIORITY


def test_extra_whitespace_between_marker_words() -> None:
    severity, _ = classify("X - Comments: PRIORITY   FOUNDATION 7-38-010")
    assert severity is Severity.PRIORITY_FOUNDATION


# --- narrative exclusions ------------------------------------------------


def test_grace_period_boilerplate_is_excluded() -> None:
    """The violation may be real, but the only PRIORITY mention is a notice."""
    severity, _ = classify(
        "58. ALLERGEN TRAINING AS REQUIRED - Comments: Observed food allergen requirements "
        "not met. A 90 day grace period was given for all new priority and priority "
        "foundation violations."
    )
    assert severity is Severity.UNCLASSIFIED


def test_future_citation_warning_is_excluded() -> None:
    severity, _ = classify(
        "X - Comments: INSTRUCTED MANAGER HE MUST PROVIDE OR CITATION PRIORITY FOUNDATION "
        "WILL BE ISSUED #7-38-010"
    )
    assert severity is Severity.UNCLASSIFIED


def test_explicit_negation_is_excluded() -> None:
    severity, _ = classify(
        "10. HANDWASHING SINKS - Comments: INSTRUCTED TO REPAIR AND MAINTAIN. "
        "NO PRIORITY FOUNDATION VIOLATION 7-38-030(c)"
    )
    assert severity is Severity.UNCLASSIFIED


def test_exclusion_is_span_based_not_entry_based() -> None:
    """The case that makes entry-level exclusion wrong.

    A genuine citation co-occurring with a forward-looking warning must stay
    positive; only the offending clause is dropped (findings §7.4).
    """
    severity, _ = classify(
        "10. HANDWASHING SINKS - Comments: MUST INSTALL A HANDWASH SINK. "
        "PRIORITY FOUNDATION VIOLATION#: 7-38-030(C). NO CITATION ISSUED. "
        "A CITATION WILL BE ISSUED IF NOT CORRECTED BY THE NEXT ROUTINE INSPECTION."
    )
    assert severity is Severity.PRIORITY_FOUNDATION


def test_no_citation_issued_does_not_exclude() -> None:
    """`NO CITATION` is not a narrative pattern -- 43,093 entries use it."""
    severity, _ = classify("X - Comments: PRIORITY FOUNDATION 7-38-010. NO CITATION ISSUED.")
    assert severity is Severity.PRIORITY_FOUNDATION


@pytest.mark.parametrize("name,_pattern", NARRATIVE_PATTERNS)
def test_every_narrative_pattern_is_documented(name: str, _pattern: str) -> None:
    """Each exclusion carries a name so the contract can enumerate it."""
    assert name and name.replace("_", "").isalnum()


# --- whole-blob parsing --------------------------------------------------


def test_parse_violations_classifies_each_entry_independently() -> None:
    raw = (
        "10. HANDWASHING - Comments: NO HOT WATER. PRIORITY 7-38-030. "
        "| 55. PHYSICAL FACILITIES - Comments: DIRTY FLOOR."
    )
    entries = parse_violations(raw)
    assert [e.severity for e in entries] == [Severity.PRIORITY, Severity.UNCLASSIFIED]


def test_has_priority_violation_is_true_if_any_entry_qualifies() -> None:
    raw = "55. A - Comments: DIRTY. | 3. B - Comments: PRIORITY FOUNDATION 7-38-010."
    assert has_priority_violation(parse_violations(raw))


def test_has_priority_violation_is_false_when_none_qualify() -> None:
    raw = "55. A - Comments: DIRTY FLOOR. | 47. B - Comments: BROKEN GASKET."
    assert not has_priority_violation(parse_violations(raw))


def test_no_entries_means_no_priority_violation() -> None:
    assert not has_priority_violation(parse_violations(None))


def test_evidence_prefers_priority_foundation_then_document_order() -> None:
    raw = "22. A - Comments: PRIORITY 7-38-005. | 3. B - Comments: PRIORITY FOUNDATION 7-38-010."
    evidence = first_priority_evidence(parse_violations(raw))
    assert evidence is not None and "PRIORITY FOUNDATION" in evidence


def test_evidence_is_none_when_nothing_is_positive() -> None:
    assert first_priority_evidence(parse_violations("55. A - Comments: DIRTY.")) is None


def test_evidence_is_whitespace_normalized() -> None:
    raw = "3. A - Comments: NO   POLICY.   PRIORITY    FOUNDATION   7-38-010."
    evidence = first_priority_evidence(parse_violations(raw))
    assert evidence is not None
    assert "  " not in evidence


# --- determinism ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "3. A - Comments: PRIORITY FOUNDATION 7-38-010.",
        "55. B - Comments: DIRTY FLOOR.",
        "10. C - Comments: A 90 day grace period was given for all new priority violations.",
        None,
        "",
    ],
)
def test_parsing_is_deterministic(raw: str | None) -> None:
    first = parse_violations(raw)
    second = parse_violations(raw)
    assert first == second


def test_parsing_an_entry_twice_gives_the_same_severity() -> None:
    text = "3. A - Comments: PRIORITY FOUNDATION 7-38-010."
    assert classify(text)[0] is classify(text)[0]
