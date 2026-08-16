"""Normalization rules.

Each rule gets its own test because each rule was a separate decision, and a
regression in any one of them silently changes which establishments merge.
Several tests assert that something is *not* normalized away: those encode the
findings that descriptive words and store numbers carry identity.
"""

from __future__ import annotations

import pytest

from sentinel.entity.normalize import (
    extract_unit,
    normalize_address,
    normalize_facility_type,
    normalize_geo,
    normalize_license,
    normalize_name,
    normalize_zip,
    strip_corporate_suffixes,
)

# --- names ---------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_blank_names_are_unusable(raw: str | None) -> None:
    assert normalize_name(raw).key is None


def test_name_is_case_folded() -> None:
    assert normalize_name("Joe's Pizza").key == normalize_name("JOE'S PIZZA").key


def test_name_unicode_is_folded_to_ascii() -> None:
    assert normalize_name("CAFÉ").key == "CAFE"


def test_possessive_folds_before_punctuation_is_stripped() -> None:
    # The ordering matters: stripping punctuation first would yield "MCDONALD S"
    # and collide with every other possessive name.
    assert normalize_name("MCDONALD'S").key == "MCDONALDS"
    assert normalize_name("MCDONALDS").key == "MCDONALDS"


def test_ampersand_becomes_and() -> None:
    assert normalize_name("FISH & CHIPS").key == normalize_name("FISH AND CHIPS").key


def test_whitespace_is_collapsed() -> None:
    assert normalize_name("ILLINOIS   SPORTSERVICE").key == "ILLINOIS SPORTSERVICE"


@pytest.mark.parametrize(
    "suffix", ["LLC", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LP", "PLLC"]
)
def test_trailing_corporate_suffix_is_stripped(suffix: str) -> None:
    assert normalize_name(f"ABC RESTAURANT {suffix}").key == "ABC RESTAURANT"


def test_stacked_corporate_suffixes_are_stripped() -> None:
    assert normalize_name("ABC RESTAURANT CO LLC").key == "ABC RESTAURANT"


def test_corporate_token_inside_a_name_is_kept() -> None:
    # "INC" here is part of the name, not a legal form.
    assert normalize_name("INC MAGAZINE CAFE").key == "INC MAGAZINE CAFE"


def test_suffix_stripping_never_empties_a_name() -> None:
    assert normalize_name("LLC").key == "LLC"


def test_leading_the_is_stripped() -> None:
    assert normalize_name("THE RED LION PUB").key == "RED LION PUB"


@pytest.mark.parametrize("descriptive", ["RESTAURANT", "CAFE", "MART", "GRILL", "BAKERY"])
def test_descriptive_trailing_words_are_kept(descriptive: str) -> None:
    """Findings §4.1: these are more common than most corporate suffixes and
    genuinely distinguish businesses. JOE'S GRILL is not JOE'S MART."""
    assert normalize_name(f"JOES {descriptive}").key == f"JOES {descriptive}"


def test_store_numbers_are_preserved_and_tracked() -> None:
    """Findings §4.2: the single most valuable anti-merge signal."""
    result = normalize_name("SUBWAY 1234")
    assert result.key == "SUBWAY 1234"
    assert result.digits == frozenset({"1234"})
    assert normalize_name("SUBWAY 5678").key != result.key


def test_name_tokens_are_exposed_as_a_set() -> None:
    assert normalize_name("TONYS TACO SHOP").tokens == frozenset({"TONYS", "TACO", "SHOP"})


def test_name_without_digits_has_an_empty_digit_set() -> None:
    assert normalize_name("PLAIN DINER").digits == frozenset()


@pytest.mark.parametrize(
    "raw",
    [
        "JOE'S PIZZA",
        "The Red Lion Pub & Grille, Inc.",
        "CAFÉ CENTRAL",
        "SUBWAY #1234",
        "A",
        "LLC",
    ],
)
def test_name_normalization_is_idempotent(raw: str) -> None:
    once = normalize_name(raw).key
    assert once is not None
    assert normalize_name(once).key == once


# --- licences ------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "0"])
def test_unusable_licences_return_none(raw: str | None) -> None:
    """Findings §2: '0' carries 323 distinct business names."""
    assert normalize_license(raw) is None


def test_ordinary_licence_passes_through() -> None:
    assert normalize_license("2147539") == "2147539"


def test_licence_is_trimmed() -> None:
    assert normalize_license("  2147539 ") == "2147539"


# --- zips ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("60601", "60601"),
        ("60601-1234", "60601"),
        ("  60601 ", "60601"),
        ("00000", None),
        ("606", None),
        ("", None),
        (None, None),
        ("ABCDE", None),
    ],
)
def test_zip_normalization(raw: str | None, expected: str | None) -> None:
    assert normalize_zip(raw) == expected


# --- addresses -----------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_addresses_are_unusable(raw: str | None) -> None:
    assert normalize_address(raw, "60601").key is None


def test_address_case_and_whitespace_are_normalized() -> None:
    """Findings §7: this alone collapses 39% of distinct address strings."""
    assert (
        normalize_address("333 W 35th St  ", "60616").key
        == normalize_address("333 W 35TH ST", "60616").key
    )


def test_missing_street_suffix_does_not_split_an_address() -> None:
    """Findings §7.1: 1901 W MADISON / AVE / ST are all the United Center."""
    keys = {
        normalize_address(a, "60612").key
        for a in ("1901 W MADISON", "1901 W MADISON ST", "1901 W MADISON AVE")
    }
    assert len(keys) == 1


def test_suffix_is_retained_as_an_attribute() -> None:
    assert normalize_address("1901 W MADISON AVE", "60612").suffix == "AVE"


def test_leading_st_is_not_treated_as_a_street_suffix() -> None:
    result = normalize_address("123 N ST CLAIR ST", "60611")
    assert result.street_body == "N ST CLAIR"
    assert result.suffix == "ST"


@pytest.mark.parametrize(
    "raw",
    ["4749-4753 N ROCKWELL ST", "4749-51 N ROCKWELL ST", "4749 - 4753 N ROCKWELL ST"],
)
def test_ranged_house_numbers_key_on_the_low_end(raw: str) -> None:
    """Findings §7: 2,231 addresses use a range, in three different styles."""
    result = normalize_address(raw, "60625")
    assert result.house_number == 4749
    assert "ranged" in result.flags
    assert result.key == normalize_address("4749 N ROCKWELL ST", "60625").key


def test_fractional_house_numbers_key_on_the_whole_number() -> None:
    result = normalize_address("2502 1/2 W DEVON AVE", "60659")
    assert result.house_number == 2502
    assert "fractional" in result.flags


def test_split_ordinals_are_rejoined() -> None:
    assert (
        normalize_address("2100 W 22 ND PL", "60608").key
        == normalize_address("2100 W 22ND PL", "60608").key
    )


def test_address_without_a_house_number_has_no_key() -> None:
    result = normalize_address("MERCHANDISE MART PLAZA", "60654")
    assert result.key is None
    assert "no_house_number" in result.flags


@pytest.mark.parametrize("directional", ["N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"])
def test_directionals_are_canonicalized(directional: str) -> None:
    result = normalize_address(f"100 {directional} MAIN ST", "60601")
    assert result.directional in {"N", "S", "E", "W"}


def test_different_directionals_produce_different_keys() -> None:
    """Chicago's grid puts 123 N MAIN and 123 S MAIN miles apart."""
    assert (
        normalize_address("123 N MAIN ST", "60601").key
        != normalize_address("123 S MAIN ST", "60601").key
    )


def test_zip_is_part_of_the_address_key() -> None:
    assert (
        normalize_address("123 N MAIN ST", "60601").key
        != normalize_address("123 N MAIN ST", "60602").key
    )


@pytest.mark.parametrize(
    ("raw", "unit"),
    [
        ("222 W ERIE STE 610", "610"),
        ("222 W ERIE SUITE 610", "610"),
        ("2 W 103RD ST STE B", "B"),
        ("100 N MAIN ST APT 4", "4"),
        ("100 N MAIN ST UNIT B", "B"),
        ("100 N MAIN ST RM 12", "12"),
        ("100 N MAIN ST FL 8", "8"),
    ],
)
def test_unit_designators_are_extracted(raw: str, unit: str) -> None:
    result = normalize_address(raw, "60601")
    assert result.unit == unit


def test_unit_is_removed_from_the_address_body() -> None:
    with_unit = normalize_address("222 W ERIE STE 610", "60654")
    without = normalize_address("222 W ERIE", "60654")
    assert with_unit.key == without.key


def test_stacked_unit_markers_are_all_removed() -> None:
    """The data stacks them: 2333 N MILWAUKEE AVE BLDG 1STFL.& BSMT."""
    result = normalize_address("2333 N MILWAUKEE AVE BLDG 1STFL.& BSMT.", "60647")
    assert result.key == normalize_address("2333 N MILWAUKEE AVE", "60647").key


def test_extract_unit_returns_the_remaining_text() -> None:
    rest, unit = extract_unit("222 W ERIE STE 610")
    assert rest == "222 W ERIE"
    assert unit == "610"


def test_extract_unit_on_an_address_without_one() -> None:
    rest, unit = extract_unit("222 W ERIE")
    assert rest == "222 W ERIE"
    assert unit is None


@pytest.mark.parametrize(
    "raw",
    [
        "1901 W MADISON ST",
        "4749-4753 N ROCKWELL ST",
        "222 W ERIE STE 610",
        "2100 W 22 ND PL",
        "123 N ST CLAIR ST",
    ],
)
def test_address_key_is_stable_under_repeated_normalization(raw: str) -> None:
    """Re-normalizing an already-normalized address must not shift the key."""
    first = normalize_address(raw, "60601")
    assert first.key is not None
    rebuilt = f"{first.house_number} {first.street_body}"
    assert normalize_address(rebuilt, "60601").key == first.key


# --- geo -----------------------------------------------------------------


def test_valid_coordinates_are_usable() -> None:
    geo = normalize_geo("41.8781", "-87.6298")
    assert geo.usable
    assert geo.key == "41.8781,-87.6298"


@pytest.mark.parametrize(
    ("lat", "lon"), [(None, "-87.6"), ("41.8", None), ("", "-87.6"), (None, None)]
)
def test_missing_coordinates_are_unusable(lat: str | None, lon: str | None) -> None:
    geo = normalize_geo(lat, lon)
    assert not geo.usable
    assert geo.reason == "missing"


def test_unparseable_coordinates_report_a_reason() -> None:
    geo = normalize_geo("not-a-number", "-87.6298")
    assert not geo.usable
    assert geo.reason == "unparseable"


def test_coordinates_outside_chicago_are_rejected() -> None:
    geo = normalize_geo("51.5074", "-0.1278")  # London
    assert not geo.usable
    assert geo.reason == "outside_bounding_box"


def test_geo_key_uses_the_raw_strings() -> None:
    """Findings §8: comparison is exact string equality, so trailing-zero
    variants must stay distinct rather than silently unify as floats."""
    assert normalize_geo("41.8781", "-87.6298").key != normalize_geo("41.87810", "-87.6298").key


# --- facility type -------------------------------------------------------


def test_facility_type_is_case_folded() -> None:
    assert normalize_facility_type("Restaurant") == normalize_facility_type("RESTAURANT")


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_facility_type_is_unknown(raw: str | None) -> None:
    """Findings §9: 5,345 rows are blank; blank must never read as a mismatch."""
    assert normalize_facility_type(raw) is None


# --- helpers -------------------------------------------------------------


def test_strip_corporate_suffixes_keeps_a_single_token() -> None:
    assert strip_corporate_suffixes(["INC"]) == ["INC"]


def test_strip_corporate_suffixes_is_trailing_only() -> None:
    assert strip_corporate_suffixes(["INC", "MAGAZINE"]) == ["INC", "MAGAZINE"]
