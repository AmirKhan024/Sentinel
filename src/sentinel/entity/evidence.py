"""Pairwise evidence rules.

Each candidate pair is reduced to a signal vector, then classified by the first
rule that applies. The rule identifier is recorded on the edge, so every merge
and every declined merge can be traced back to one named reason.

Rule design follows the measurements in
``docs/analysis/entity_resolution_findings.md``:

- **Address equivalence is required for every non-licence merge** (§4.2, §13).
  This is the single most important property: it is what makes 247 Subways safe
  without a chain-name list.
- **Licence equality is supporting evidence, never the key, and licence
  *inequality* is never evidence against** (§3). 18.47% of establishments hold
  more than one licence.
- **Name matching is exact after normalization** over both ``dba_name`` and
  ``aka_name`` (§4, §5). There is no fuzzy tier: it would resolve 0.21% of
  licences while endangering the mega-address cases.
- **No temporal logic** (§11), which also keeps ``inspection_date`` out of the
  matcher entirely.

Vetoes are evaluated before merge rules, so a conflict always beats agreement.
"""

from __future__ import annotations

import math

from sentinel.entity.models import MatchTier, Node, PairSignals, PairVerdict, Thresholds

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres.

    Used only for a cluster-level sanity bound. Findings §8 measured zero
    coordinate spread within an address, so a cluster spanning real distance
    indicates a bug, not a geographic fact.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def names_match(left: Node, right: Node) -> bool:
    """Whether any normalized name of one node equals any name of the other.

    Both ``dba_name`` and ``aka_name`` participate (findings §5): the legal
    entity is frequently renamed while the premises is unchanged, so requiring
    ``dba_name`` agreement alone would split an establishment every time a
    holding company changed hands.
    """
    return bool(left.name_keys & right.name_keys)


def aka_conflict(left: Node, right: Node) -> bool:
    """Whether two nodes carry trade names that corroborate nothing.

    Discovered by inspecting the real output. At O'Hare, ``dba_name`` is often
    the *concessionaire* -- ``HOST INTERNATIONAL INC`` appears on Starbucks,
    Johnny Rockets, Chili's and a dozen other physically separate outlets in
    different terminals. Matching on ``dba_name`` alone chained all of them into
    one establishment, pooling twenty restaurants' inspection histories.

    The distinguishing identity lives in ``aka_name`` (``STARBUCKS (T3 H2)`` vs
    ``JOHNNY ROCKETS (T2/NEAR SECURITY)``). So when both nodes carry a trade
    name and *neither* one appears anywhere in the other's set of names, the
    trade names actively disagree and the shared operator name is not evidence
    of a shared premises.

    Deliberately lenient, in two ways. If either trade name matches any name on
    the other node there is no conflict, which keeps ``SUBWAY`` /
    ``SUBWAY #1234`` and every case where ``aka_name`` simply repeats
    ``dba_name`` unaffected. And if one trade name's tokens are contained in the
    other's, they corroborate rather than conflict -- ``HOT WOK CHINESE
    KITCHEN`` inside ``NEW HOT WOK CHINESE KITCHEN`` is the same restaurant
    renamed, not two different ones.

    A conflict therefore means the trade names are genuinely unrelated:
    ``STARBUCKS (T3 H2)`` and ``GOOSE ISLAND CHICAGO (T3/L10)``.
    """
    left_aka, right_aka = left.aka.key, right.aka.key
    if left_aka is None or right_aka is None:
        return False
    if left_aka in right.name_keys or right_aka in left.name_keys:
        return False
    left_tokens, right_tokens = left.aka.tokens, right.aka.tokens
    corroborates_by_containment = bool(left_tokens and right_tokens) and (
        left_tokens < right_tokens or right_tokens < left_tokens
    )
    return not corroborates_by_containment


def name_containment(left: Node, right: Node) -> bool:
    """Whether one node's name tokens are a strict subset of the other's.

    Catches ``TACO SHOP`` inside ``TONYS TACO SHOP``. Only ever consulted at an
    already-equivalent address, and only as the weakest merge tier.
    """
    for a in (left.name, left.aka):
        for b in (right.name, right.aka):
            if not a.tokens or not b.tokens or a.tokens == b.tokens:
                continue
            if a.tokens < b.tokens or b.tokens < a.tokens:
                return True
    return False


def compute_signals(
    left: Node,
    right: Node,
    thresholds: Thresholds,
    blacklist: frozenset[str] = frozenset(),
) -> PairSignals:
    """Compare two nodes across every signal, without judging them."""
    left_addr, right_addr = left.address, right.address

    same_license = (
        left.license_key is not None
        and right.license_key is not None
        and left.license_key == right.license_key
    )

    same_addr_key = (
        left_addr.key is not None and right_addr.key is not None and left_addr.key == right_addr.key
    )

    same_zip = (
        left_addr.zip_key is not None
        and right_addr.zip_key is not None
        and left_addr.zip_key == right_addr.zip_key
    )

    # Exact coordinate equality stands in for address equality (§8), but only
    # when the coordinate is not a known geocoder artefact and the zip agrees.
    same_coord = (
        left.geo.key is not None
        and right.geo.key is not None
        and left.geo.key == right.geo.key
        and left.geo.key not in blacklist
        and same_zip
    )

    near_addr = False
    if (
        left_addr.house_number is not None
        and right_addr.house_number is not None
        and left_addr.street_body is not None
        and left_addr.street_body == right_addr.street_body
        and same_zip
    ):
        delta = abs(left_addr.house_number - right_addr.house_number)
        near_addr = 0 < delta <= thresholds.near_house_number_delta

    # A store-number disagreement is the strongest anti-merge signal available
    # (§4.2): SUBWAY 1234 and SUBWAY 5678 are different franchise locations.
    left_digits, right_digits = left.all_digits, right.all_digits
    digit_conflict = bool(left_digits) and bool(right_digits) and not (left_digits & right_digits)

    unit_conflict = (
        left_addr.unit is not None
        and right_addr.unit is not None
        and left_addr.unit != right_addr.unit
    )

    dir_conflict = (
        left_addr.directional is not None
        and right_addr.directional is not None
        and left_addr.directional != right_addr.directional
    )

    # None means "unknown", never "mismatch": 5,345 rows have a blank facility
    # type (§9).
    facility_agree: bool | None = None
    if left.facility_type_key is not None and right.facility_type_key is not None:
        facility_agree = left.facility_type_key == right.facility_type_key

    return PairSignals(
        same_license=same_license,
        same_addr_key=same_addr_key,
        same_coord=same_coord,
        near_addr=near_addr,
        same_zip=same_zip,
        name_exact=names_match(left, right),
        name_containment=name_containment(left, right),
        aka_conflict=aka_conflict(left, right),
        digit_conflict=digit_conflict,
        unit_conflict=unit_conflict,
        dir_conflict=dir_conflict,
        facility_agree=facility_agree,
    )


def _veto(signals: PairSignals) -> str | None:
    """Return the identifier of the first veto that fires, if any.

    Vetoes run before merge rules so a genuine conflict always outranks
    agreement, however strong that agreement looks.
    """
    # V1: different directionals on the same street number. Chicago's grid puts
    # 123 N MAIN and 123 S MAIN miles apart.
    if signals.dir_conflict:
        return "V1"
    # V2: different suites at one street address are different establishments.
    # Waived when the licence agrees, since a single licence covering two suites
    # is one operator's premises.
    if signals.unit_conflict and not signals.same_license:
        return "V2"
    # V3: conflicting store numbers (§4.2).
    if signals.digit_conflict and not signals.same_license:
        return "V3"
    # V4: the name evidence that would drive this merge is a shared *operator*
    # name, and the trade names actively disagree. See aka_conflict().
    #
    # Gated on there being name evidence at all. Without that gate the veto
    # fires on any two differently-named neighbours, pre-empting the ordinary
    # no-match rules and -- worse -- blocking legitimate containment merges such
    # as HOT WOK CHINESE KITCHEN / NEW HOT WOK CHINESE KITCHEN, whose trade
    # names naturally differ. Caught by the test suite before it shipped.
    #
    # Waived when the licence agrees, because one licence at one address is a
    # single premises even when the business on it was renamed or replaced.
    name_driven = signals.name_exact or signals.name_containment
    if name_driven and signals.aka_conflict and not signals.same_license:
        return "V4"
    return None


def evaluate_pair(
    left: Node,
    right: Node,
    thresholds: Thresholds,
    blacklist: frozenset[str] = frozenset(),
) -> PairVerdict:
    """Classify a candidate pair. First matching rule wins."""
    if left.node_id > right.node_id:
        left, right = right, left

    signals = compute_signals(left, right, thresholds, blacklist)

    veto = _veto(signals)
    if veto is not None:
        return PairVerdict(left.node_id, right.node_id, MatchTier.NO_MATCH, veto, signals)

    rule, tier = _classify(signals)
    return PairVerdict(left.node_id, right.node_id, tier, rule, signals)


def _classify(signals: PairSignals) -> tuple[str, MatchTier]:
    """Map a clean signal vector to (rule id, tier)."""
    addr_equivalent = signals.addr_equivalent

    # --- strong: two independent identity signals agree ------------------
    # S1: same licence at the same place. Nothing stronger exists.
    if signals.same_license and addr_equivalent:
        return "S1", MatchTier.STRONG
    # S2: same name at the same place, whatever the licences say. This is the
    # rule that repairs the 18.47% of establishments holding several licences
    # (§3.2), and the ABC RESTAURANT / ABC RESTAURANT LLC case.
    if addr_equivalent and signals.name_exact:
        return "S2", MatchTier.STRONG
    # S3: same licence and same name one or two house numbers apart -- an
    # address typo corroborated by two other agreeing signals.
    if signals.same_license and signals.near_addr and signals.name_exact:
        return "S3", MatchTier.STRONG

    # --- probable: one signal agrees, nothing contradicts ------------------
    # P1: same licence, address off by a digit or two, names differ. Weaker
    # than S3 because only the licence corroborates.
    if signals.same_license and signals.near_addr:
        return "P1", MatchTier.PROBABLE
    # P2: one name contains the other at the same place, with no facility-type
    # disagreement.
    if addr_equivalent and signals.name_containment and signals.facility_agree is not False:
        return "P2", MatchTier.PROBABLE

    # --- ambiguous: real evidence, not enough of it. Recorded, never merged --
    # A1: one licence at two genuinely different places. Either a relocation or
    # a multi-site operator; §3.3 shows 86 licences span more than a kilometre.
    # Splitting is the safe error here.
    if signals.same_license and not addr_equivalent and not signals.near_addr:
        return "A1", MatchTier.AMBIGUOUS
    # A2: name containment at one place, but the facility types disagree.
    if addr_equivalent and signals.name_containment:
        return "A2", MatchTier.AMBIGUOUS

    # --- no match ----------------------------------------------------------
    # N2: same place, same kind of business, unrelated names. Inspecting these
    # showed exactly what findings §11 predicts -- successive tenants and
    # neighbouring units (KARMA MINI MART / MB & S MARKET at one address). They
    # are genuinely different businesses, so this is a decision, not a doubt,
    # and classifying it as ambiguous would bury the real review queue under
    # ~108k pairs of ordinary strip-mall neighbours.
    if addr_equivalent and signals.facility_agree:
        return "N2", MatchTier.NO_MATCH
    # N1: same place, nothing else agrees. The common case at dense addresses.
    if addr_equivalent:
        return "N1", MatchTier.NO_MATCH
    return "N0", MatchTier.NO_MATCH
