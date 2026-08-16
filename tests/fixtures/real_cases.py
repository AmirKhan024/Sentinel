"""Real edge cases harvested from the Chicago snapshot.

Every value here was copied verbatim from
``food_inspections_20260816T070911Z.parquet`` (sha256 ``7d3c4069...ad38``) while
inspecting the first full resolution run. They are literal Python rather than a
CSV or JSON fixture because the repository's ``.gitignore`` excludes ``*.csv``
and ``*.jsonl`` project-wide, so a data-file fixture would silently fail to
commit.

Each case carries a ``why`` that is surfaced in the assertion message, so a
future failure explains what real-world situation just regressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RealCase:
    """Two real records and whether they should resolve to one establishment."""

    case_id: str
    why: str
    should_merge: bool
    left: dict[str, Any]
    right: dict[str, Any]
    tags: tuple[str, ...] = field(default=())


def _row(
    dba_name: str,
    aka_name: str,
    address: str,
    zip_code: str,
    license_: str,
    *,
    latitude: str | None = "41.8781",
    longitude: str | None = "-87.6298",
    facility_type: str = "Restaurant",
) -> dict[str, Any]:
    return {
        "dba_name": dba_name,
        "aka_name": aka_name,
        "address": address,
        "zip": zip_code,
        "license_": license_,
        "latitude": latitude,
        "longitude": longitude,
        "facility_type": facility_type,
    }


REAL_CASES: list[RealCase] = [
    # --- must merge: real establishments split by recording noise ---------
    RealCase(
        case_id="united_center_suffix_disagreement",
        why=(
            "The United Center is recorded as 1901 W MADISON ST and 1901 W MADISON AVE. "
            "Chicago has one W Madison; the suffix is simply inconsistent, and excluding "
            "it from the address key is what keeps the venue whole."
        ),
        should_merge=True,
        left=_row(
            "LEVY RESTAURANTS AT UNITED CENTER",
            "LEVY RESTAURANTS AT UNITED CENTER",
            "1901 W MADISON AVE ",
            "60612",
            "2522718",
            latitude="41.8807",
            longitude="-87.6742",
        ),
        right=_row(
            "LEVY RESTAURANTS AT UNITED CENTER",
            "LEVY RESTAURANTS AT UNITED CENTER",
            "1901 W MADISON ST ",
            "60612",
            "2601805",
            latitude="41.8807",
            longitude="-87.6742",
        ),
        tags=("address", "suffix"),
    ),
    RealCase(
        case_id="kedzie_trailing_whitespace",
        why=(
            "NOONOKABAB at 4707-4713 N KEDZIE AVE appears with and without a trailing "
            "space. Whitespace alone accounts for most of the 39% address collapse."
        ),
        should_merge=True,
        left=_row(
            "NOONOKABAB",
            "NOONOKABAB",
            "4707-4713 N KEDZIE AVE ",
            "60625",
            "2488196",
            latitude="41.9679",
            longitude="-87.7086",
        ),
        right=_row(
            "NOONOKABAB",
            "NOONOKABAB",
            "4707-4713 N KEDZIE AVE",
            "60625",
            "2627692",
            latitude="41.9679",
            longitude="-87.7086",
        ),
        tags=("address", "whitespace", "ranged"),
    ),
    RealCase(
        case_id="dominos_corporate_and_descriptive_name",
        why=(
            "DOMINO'S and DOMINO'S PIZZA at 4039 W 26TH ST across four licences are one "
            "pizzeria. Licence inequality must not count against a match: 18.47% of "
            "establishments hold more than one."
        ),
        should_merge=True,
        left=_row(
            "DOMINO'S",
            "DOMINO'S",
            "4039 W 26TH ST ",
            "60623",
            "2886342",
            latitude="41.8442",
            longitude="-87.7280",
        ),
        right=_row(
            "DOMINO'S PIZZA",
            "DOMINO'S PIZZA",
            "4039 W 26TH ST",
            "60623",
            "36834",
            latitude="41.8442",
            longitude="-87.7280",
        ),
        tags=("name", "containment", "licence"),
    ),
    RealCase(
        case_id="lake_shore_drive_rename",
        why=(
            "Chicago renamed Lake Shore Drive to Jean Baptiste Pointe DuSable Lake Shore "
            "Drive in 2021. No string rule bridges that, but the geocoder assigns both "
            "spellings the same coordinate."
        ),
        should_merge=True,
        left=_row(
            "MUSEUM KITCHEN",
            "MUSEUM KITCHEN",
            "5700 S LAKE SHORE DR ",
            "60637",
            "1224021",
            latitude="41.79142028209799",
            longitude="-87.58014776868883",
        ),
        right=_row(
            "MUSEUM KITCHEN",
            "MUSEUM KITCHEN",
            "5700 S JEAN BAPTISTE POINTE DUSABLE LAKE SHORE DR",
            "60637",
            "2845503",
            latitude="41.79142028209799",
            longitude="-87.58014776868883",
        ),
        tags=("geo", "rename"),
    ),
    RealCase(
        case_id="marianos_legal_entity_vs_trade_name",
        why=(
            "dba_name is the holding company (1918 WINTER STREET ILLINOIS LLC) while "
            "aka_name is the store (MARIANO'S). Reading dba_name alone would split the "
            "premises every time the entity is renamed."
        ),
        should_merge=True,
        left=_row(
            "1918 WINTER STREET ILLINOIS LLC",
            "MARIANO'S",
            "5353 N ELSTON AVE",
            "60630",
            "2977884",
            latitude="41.9724",
            longitude="-87.7566",
        ),
        right=_row(
            "MARIANO'S",
            "MARIANO'S",
            "5353 N ELSTON AVE",
            "60630",
            "2977885",
            latitude="41.9724",
            longitude="-87.7566",
        ),
        tags=("aka", "name"),
    ),
    RealCase(
        case_id="illinois_sportservice_punctuation",
        why=(
            "Licence 14616 carries seven spellings of ILLINOIS SPORTSERVICE INC at the "
            "White Sox park, differing only in punctuation, spacing and the comma before "
            "INC."
        ),
        should_merge=True,
        left=_row(
            "ILLINOIS SPORTSERVICE, INC",
            "ILLINOIS SPORTSERVICE, INC",
            "333 W 35TH ST ",
            "60616",
            "14616",
            latitude="41.8300",
            longitude="-87.6338",
        ),
        right=_row(
            "ILLINOIS SPORT SERVICE INC.",
            "ILLINOIS SPORT SERVICE INC.",
            "333 W 35TH ST",
            "60616",
            "14616",
            latitude="41.8300",
            longitude="-87.6338",
        ),
        tags=("name", "punctuation"),
    ),
    # --- must NOT merge: different premises that look similar -------------
    RealCase(
        case_id="ohare_host_international_concessionaire",
        why=(
            "HOST INTERNATIONAL INC is the O'Hare concessionaire, not a restaurant. It "
            "appears on Starbucks in Terminal 3 and Johnny Rockets in Terminal 2, which "
            "are physically separate premises. Merging on the operator name pooled about "
            "twenty restaurants' histories in the first run."
        ),
        should_merge=False,
        left=_row(
            "HOST INTERNATIONAL INC",
            "STARBUCKS  (T3 H2)",
            "11601 W TOUHY AVE ",
            "60666",
            "34150",
            latitude="42.0088",
            longitude="-87.9069",
        ),
        right=_row(
            "HOST INTERNATIONAL INC",
            "JOHNNY ROCKETS (T2/NEAR SECURITY)",
            "11601 W TOUHY AVE",
            "60666",
            "34183",
            latitude="42.0088",
            longitude="-87.9069",
        ),
        tags=("aka", "mega_address", "false_merge"),
    ),
    RealCase(
        case_id="walgreen_and_chick_fil_a_share_an_address",
        why=(
            "122 S MICHIGAN AVE houses both WALGREEN #15567 and CHICK-FIL-A. A shared "
            "address is not identity evidence on its own."
        ),
        should_merge=False,
        left=_row(
            "WALGREEN #15567",
            "WALGREENS",
            "122 S Michigan AVE BLDG ",
            "60603",
            "2328487",
            latitude="41.8804",
            longitude="-87.6245",
            facility_type="Grocery Store",
        ),
        right=_row(
            "CHICK-FIL-A MILLENNIUM PARK",
            "CHICK-FIL-A",
            "122 S MICHIGAN AVE",
            "60603",
            "2929502",
            latitude="41.8804",
            longitude="-87.6245",
        ),
        tags=("mega_address", "false_merge"),
    ),
    RealCase(
        case_id="business_succession_at_one_address_different_licences",
        why=(
            "KARMA MINI MART and MB & S MARKET occupied 6214 S ASHLAND AVE under "
            "different licences. Same address, same facility type, unrelated names: "
            "different businesses, and not even ambiguous."
        ),
        should_merge=False,
        left=_row(
            "KARMA MINI MART",
            "KARMA MINI MART",
            "6214 S ASHLAND AVE",
            "60636",
            "2461558",
            latitude="41.7776",
            longitude="-87.6640",
            facility_type="Grocery Store",
        ),
        right=_row(
            "MB & S MARKET,CO.",
            "MB & S MARKET,CO.",
            "6214 S ASHLAND AVE",
            "60636",
            "2604513",
            latitude="41.7776",
            longitude="-87.6640",
            facility_type="Grocery Store",
        ),
        tags=("succession", "false_merge"),
    ),
    RealCase(
        case_id="numbered_franchises_at_one_address",
        why=(
            "Store numbers are the strongest anti-merge signal in this dataset. Two "
            "differently numbered outlets of one chain at one address are different "
            "franchise locations."
        ),
        should_merge=False,
        left=_row(
            "SUBWAY #25917",
            "SUBWAY #25917",
            "122 S MICHIGAN AVE",
            "60603",
            "2500001",
            latitude="41.8804",
            longitude="-87.6245",
        ),
        right=_row(
            "SUBWAY #15567",
            "SUBWAY #15567",
            "122 S MICHIGAN AVE",
            "60603",
            "2500002",
            latitude="41.8804",
            longitude="-87.6245",
        ),
        tags=("digits", "false_merge"),
    ),
    RealCase(
        case_id="same_name_different_addresses_is_a_chain",
        why=(
            "JOHNNY ROCKETS operates at 177 N State St and 30 N La Salle St. There are "
            "247 Subways in this dataset; requiring address equivalence for every "
            "non-licence merge is what makes chains harmless."
        ),
        should_merge=False,
        left=_row(
            "JOHNNY ROCKETS",
            "JOHNNY ROCKETS",
            "177 N State ST ",
            "60601",
            "2048281",
            latitude="41.8846",
            longitude="-87.6280",
        ),
        right=_row(
            "JOHNNY ROCKETS",
            "JOHNNY ROCKETS",
            "30 N LA SALLE ST ",
            "60602",
            "2036580",
            latitude="41.8825",
            longitude="-87.6325",
        ),
        tags=("chain", "false_merge"),
    ),
    RealCase(
        case_id="sentinel_licence_does_not_join_unrelated_places",
        why=(
            "The '0' licence sentinel is attached to 323 distinct business names across "
            "364 addresses. Treating it as an identifier would fuse all of them."
        ),
        should_merge=False,
        left=_row(
            "A PLACE",
            "A PLACE",
            "100 N FIRST ST",
            "60601",
            "0",
            latitude="41.8781",
            longitude="-87.6298",
        ),
        right=_row(
            "B PLACE",
            "B PLACE",
            "200 N SECOND ST",
            "60602",
            "0",
            latitude="41.8900",
            longitude="-87.6400",
        ),
        tags=("licence", "sentinel", "false_merge"),
    ),
]
