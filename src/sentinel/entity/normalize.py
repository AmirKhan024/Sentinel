"""Normalization of the raw identity fields.

Every rule here was chosen against measurements in
``docs/analysis/entity_resolution_findings.md``, and several are deliberately
the opposite of the usual advice:

- Digits are **kept** in names. ``SUBWAY 1234`` and ``SUBWAY 5678`` are
  different franchise locations (§4.2).
- The street suffix is **excluded** from the address key rather than
  canonicalized. The dataset has no long-form suffixes, but it does have
  addresses whose suffix is missing or contradictory (§7.1).
- Descriptive trailing words (``RESTAURANT``, ``CAFE``, ``MART``) are **not**
  stripped even though they are more common than most corporate suffixes. They
  distinguish businesses; ``LLC`` does not (§4.1).

Everything is a pure function of its arguments, so each rule is independently
testable and the whole pipeline is trivially deterministic.
"""

from __future__ import annotations

import re
import unicodedata

from sentinel.entity.models import Geo, NormalizedAddress, NormalizedName

# Licence values that are present but meaningless. §2: the literal '0' is
# attached to 323 distinct business names across 364 addresses, so treating it
# as an identifier would fuse all of them.
LICENSE_SENTINELS = frozenset({"0"})

# Trailing legal-form tokens. §4.1: each of these occurs as the final token of
# at least 100 distinct names, and none of them distinguishes one business from
# another. Deliberately excludes RESTAURANT/CAFE/MART/GRILL, which are more
# frequent but *are* distinguishing.
CORPORATE_SUFFIXES = frozenset(
    {
        "LLC",
        "L L C",
        "INC",
        "INCORPORATED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LTD",
        "LP",
        "LLP",
        "PC",
        "PLLC",
        "DBA",
    }
)

# §7: exactly one address in 20,312 uses a long-form directional. Retained for
# safety and future data, documented as near-zero-yield.
DIRECTIONALS: dict[str, str] = {
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "NORTHEAST": "NE",
    "NORTHWEST": "NW",
    "SOUTHEAST": "SE",
    "SOUTHWEST": "SW",
    "N": "N",
    "S": "S",
    "E": "E",
    "W": "W",
    "NE": "NE",
    "NW": "NW",
    "SE": "SE",
    "SW": "SW",
}

# §7: no long forms occur in the data. This maps them anyway so that a future
# snapshot containing them does not silently produce a second address key.
STREET_SUFFIXES: dict[str, str] = {
    "STREET": "ST",
    "ST": "ST",
    "AVENUE": "AVE",
    "AVENU": "AVE",
    "AVEN": "AVE",
    "AVE": "AVE",
    "AV": "AVE",
    "BOULEVARD": "BLVD",
    "BOUL": "BLVD",
    "BLVD": "BLVD",
    "ROAD": "RD",
    "RD": "RD",
    "DRIVE": "DR",
    "DR": "DR",
    "PLACE": "PL",
    "PL": "PL",
    "COURT": "CT",
    "CT": "CT",
    "PARKWAY": "PKWY",
    "PKWY": "PKWY",
    "PKY": "PKWY",
    "TERRACE": "TER",
    "TER": "TER",
    "LANE": "LN",
    "LN": "LN",
    "HIGHWAY": "HWY",
    "HWY": "HWY",
    "SQUARE": "SQ",
    "SQ": "SQ",
    "PLAZA": "PLZ",
    "PLZ": "PLZ",
    "CIRCLE": "CIR",
    "CIR": "CIR",
    "EXPRESSWAY": "EXPY",
    "EXPY": "EXPY",
    "TRAIL": "TRL",
    "TRL": "TRL",
}

# §7: unit markers occur on 1.96% of distinct addresses. Rare, but a suite
# disagreement is a genuine "different establishment" signal, so units are
# extracted and kept rather than stripped and forgotten.
UNIT_MARKERS = (
    "SUITE",
    "STE",
    "APARTMENT",
    "APT",
    "UNIT",
    "ROOM",
    "RM",
    "FLOOR",
    "FL",
    "BUILDING",
    "BLDG",
    "BASEMENT",
    "BSMT",
    "SPACE",
    "SPC",
    "REAR",
    "LL",
)
_UNIT_PATTERN = re.compile(
    r"\b(?:" + "|".join(UNIT_MARKERS) + r")\b\.?\s*([A-Z0-9][A-Z0-9\-]*)?",
)

# Chicago bounding box. §8 measured zero rows outside it, so anything outside is
# a defect rather than a distant establishment.
LAT_MIN, LAT_MAX = 41.60, 42.10
LON_MIN, LON_MAX = -87.95, -87.50

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_POSSESSIVE = re.compile(r"'\s*S\b")
_LEADING_HOUSE = re.compile(r"^(\d+)")
_RANGED_HOUSE = re.compile(r"^(\d+)\s*-\s*\d+")
_FRACTIONAL = re.compile(r"^(\d+)\s+1\s*/\s*2\b")
_ORDINAL_SPLIT = re.compile(r"\b(\d+)\s+(ST|ND|RD|TH)\b")
_ORDINAL_SUFFIX = re.compile(r"\b(\d+)(?:ST|ND|RD|TH)\b")
_DIGIT_TOKEN = re.compile(r"\d")


def _fold_ascii(value: str) -> str:
    """NFKD-fold to ASCII so CAFE and CAFÉ compare equal."""
    decomposed = unicodedata.normalize("NFKD", value)
    return decomposed.encode("ascii", "ignore").decode("ascii")


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def normalize_license(raw: str | None) -> str | None:
    """Return a usable licence number, or None if it carries no information.

    §2: 19 rows are null and 831 carry the '0' sentinel. Those 850 rows (0.27%)
    resolve on name and address alone.
    """
    value = _blank_to_none(raw)
    if value is None:
        return None
    if value in LICENSE_SENTINELS:
        return None
    return value


def strip_corporate_suffixes(tokens: list[str]) -> list[str]:
    """Remove trailing legal-form tokens, iteratively, never emptying the name.

    Trailing-only is deliberate: ``INC`` in ``INC MAGAZINE CAFE`` is part of the
    name, not a legal form. A leading ``THE`` is also dropped, since it varies
    freely across records for the same business.
    """
    result = list(tokens)
    while len(result) > 1 and result[-1] in CORPORATE_SUFFIXES:
        result.pop()
    while len(result) > 1 and result[0] == "THE":
        result.pop(0)
    return result


def normalize_name(raw: str | None) -> NormalizedName:
    """Reduce a business name to a comparable key, tokens and digit set.

    Order matters. Possessives are folded before punctuation becomes
    whitespace, so ``MCDONALD'S`` becomes ``MCDONALDS`` rather than splitting
    into ``MCDONALD S`` and colliding with every other possessive name.
    """
    value = _blank_to_none(raw)
    if value is None:
        return NormalizedName(key=None)

    text = _fold_ascii(value).upper()
    text = _POSSESSIVE.sub("S", text)
    text = text.replace("&", " AND ")
    text = _NON_ALNUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return NormalizedName(key=None)

    tokens = strip_corporate_suffixes(text.split(" "))
    if not tokens:
        return NormalizedName(key=None)

    key = " ".join(tokens)
    # Digits are retained and tracked separately: a store-number disagreement is
    # the strongest anti-merge signal available (§4.2, §12).
    digits = frozenset(t for t in tokens if _DIGIT_TOKEN.search(t))
    return NormalizedName(key=key, tokens=frozenset(tokens), digits=digits)


def normalize_zip(raw: str | None) -> str | None:
    """Take the leading five digits; reject blanks, 00000 and non-numeric forms."""
    value = _blank_to_none(raw)
    if value is None:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 5:
        return None
    five = digits[:5]
    if five == "00000":
        return None
    return five


def extract_unit(address: str) -> tuple[str, str | None]:
    """Split unit designators out of an address, returning (rest, unit).

    Runs before any other address rule, because later steps would destroy the
    markers. §7: only 1.96% of addresses carry one, but where present a
    disagreement genuinely separates two establishments.

    Markers are removed repeatedly rather than once, because the data stacks
    them: ``2333 N MILWAUKEE AVE BLDG 1STFL.& BSMT.`` carries three. Captured
    values are sorted before joining so the result does not depend on the order
    they happened to be written in.
    """
    rest = address
    values: list[str] = []
    # Bounded so a pathological string cannot loop; the observed maximum is 3.
    for _ in range(8):
        match = _UNIT_PATTERN.search(rest)
        if match is None:
            break
        captured = match.group(1)
        if captured:
            values.append(captured)
        rest = rest[: match.start()] + " " + rest[match.end() :]
        rest = _WHITESPACE.sub(" ", rest).strip()
    unit = " ".join(sorted(set(values))) if values else None
    return rest, unit


def canonical_directional(token: str) -> str | None:
    return DIRECTIONALS.get(token)


def canonical_street_suffix(token: str) -> str | None:
    return STREET_SUFFIXES.get(token)


def normalize_address(raw: str | None, raw_zip: str | None) -> NormalizedAddress:
    """Split an address into house number, directional, street body and unit.

    The returned ``key`` is ``house|directional street|zip`` and deliberately
    omits the street suffix (§7.1). It is None whenever the address cannot be
    pinned to a house number, in which case the record can only match on licence.
    """
    zip_key = normalize_zip(raw_zip)
    value = _blank_to_none(raw)
    if value is None:
        return NormalizedAddress(key=None, zip_key=zip_key)

    text = _fold_ascii(value).upper()
    text = _WHITESPACE.sub(" ", text).strip()

    text, unit = extract_unit(text)

    flags: set[str] = set()

    # House number, in the three range styles the data actually contains
    # (§7: 4749-4753, 4749-51, 3000 -3002, plus 1234 1/2 fractionals).
    house_number: int | None = None
    fractional = _FRACTIONAL.match(text)
    ranged = _RANGED_HOUSE.match(text)
    leading = _LEADING_HOUSE.match(text)
    if fractional is not None:
        house_number = int(fractional.group(1))
        flags.add("fractional")
        text = text[fractional.end() :]
    elif ranged is not None:
        house_number = int(ranged.group(1))
        flags.add("ranged")
        text = text[ranged.end() :]
    elif leading is not None:
        house_number = int(leading.group(1))
        text = text[leading.end() :]
    else:
        flags.add("no_house_number")

    text = _NON_ALNUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()

    # Ordinals: the data contains both '22ND' and '22 ND' for the same street.
    text = _ORDINAL_SPLIT.sub(r"\1\2", text)
    text = _ORDINAL_SUFFIX.sub(r"\1", text)

    tokens = [t for t in text.split(" ") if t]

    directional: str | None = None
    if tokens:
        candidate = canonical_directional(tokens[0])
        if candidate is not None and len(tokens) > 1:
            directional = candidate
            tokens = tokens[1:]

    suffix: str | None = None
    if len(tokens) > 1:
        candidate_suffix = canonical_street_suffix(tokens[-1])
        if candidate_suffix is not None:
            suffix = candidate_suffix
            tokens = tokens[:-1]

    street_body = " ".join(tokens) or None

    key: str | None = None
    if house_number is not None and street_body is not None:
        prefix = f"{directional} " if directional else ""
        key = f"{house_number}|{prefix}{street_body}|{zip_key or ''}"

    return NormalizedAddress(
        key=key,
        house_number=house_number,
        street_body=(
            f"{directional} {street_body}" if directional and street_body else street_body
        ),
        directional=directional,
        suffix=suffix,
        unit=unit,
        zip_key=zip_key,
        flags=frozenset(flags),
    )


def normalize_geo(raw_lat: str | None, raw_lon: str | None) -> Geo:
    """Parse a coordinate pair, recording why it is unusable when it is.

    The key is built from the *raw* strings rather than the parsed floats.
    §8: coordinates are a deterministic function of the address string with zero
    variance, so exact string equality is the correct comparison and avoids any
    float-formatting ambiguity. The floats are parsed only to range-check.
    """
    lat_text = _blank_to_none(raw_lat)
    lon_text = _blank_to_none(raw_lon)
    if lat_text is None or lon_text is None:
        return Geo(key=None, reason="missing")

    try:
        lat = float(lat_text)
        lon = float(lon_text)
    except ValueError:
        return Geo(key=None, reason="unparseable")

    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return Geo(key=None, lat=lat, lon=lon, reason="outside_bounding_box")

    return Geo(key=f"{lat_text},{lon_text}", lat=lat, lon=lon)


def normalize_facility_type(raw: str | None) -> str | None:
    """Case-fold a facility type. Blank stays None, meaning 'unknown'.

    §9: blank appears on 5,345 rows and must never be read as a mismatch.
    """
    value = _blank_to_none(raw)
    if value is None:
        return None
    text = _fold_ascii(value).upper()
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip() or None
