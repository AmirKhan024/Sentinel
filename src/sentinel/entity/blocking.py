"""Candidate pair generation.

Comparing every node to every other node would be quadratic and pointless: the
vast majority of pairs share nothing. Blocking restricts comparison to nodes
that already agree on something structural.

The block choice is what contains chaining. Every non-licence block requires
location agreement, so two nodes at different places are never even compared on
name. That is what makes 247 Subways (findings §4.2) harmless without any
special-casing of chain names: no rule can reach across addresses.

There is deliberately **no name block**. A block keyed on ``SUBWAY`` would hold
369 licences across 247 addresses and would invite exactly the merge the data
says is wrong.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence

from sentinel.entity.models import Node, OversizedBlock, Thresholds

logger = logging.getLogger(__name__)


def spatial_blocks(nodes: Sequence[Node]) -> dict[str, list[str]]:
    """Block on (zip, house number). The workhorse.

    Small blocks: one street number within one postal code. Nodes lacking either
    component are simply absent, and can then only match via licence.
    """
    blocks: dict[str, list[str]] = {}
    for node in nodes:
        address = node.address
        if address.zip_key is None or address.house_number is None:
            continue
        blocks.setdefault(f"{address.zip_key}|{address.house_number}", []).append(node.node_id)
    return blocks


def coordinate_blocks(nodes: Sequence[Node], blacklist: frozenset[str]) -> dict[str, list[str]]:
    """Block on the exact geocoded coordinate.

    Findings §8: address variants of one place share a single coordinate, which
    catches cases string normalization cannot — including Chicago's 2021 rename
    of Lake Shore Drive. Blacklisted coordinates (those covering too many
    distinct addresses to be one place) are excluded.
    """
    blocks: dict[str, list[str]] = {}
    for node in nodes:
        key = node.geo.key
        if key is None or key in blacklist:
            continue
        blocks.setdefault(key, []).append(node.node_id)
    return blocks


def license_blocks(nodes: Sequence[Node]) -> dict[str, list[str]]:
    """Block on the licence number, sentinels already excluded by normalization."""
    blocks: dict[str, list[str]] = {}
    for node in nodes:
        if node.license_key is None:
            continue
        blocks.setdefault(node.license_key, []).append(node.node_id)
    return blocks


def _pairs_from_blocks(
    blocks: dict[str, list[str]],
    kind: str,
    thresholds: Thresholds,
    oversized: list[OversizedBlock],
) -> Iterator[tuple[str, str]]:
    """Expand blocks into ordered pairs, skipping and recording oversized ones.

    An oversized block is *not* silently truncated. It is recorded so the run
    can report that some comparisons were never made, and its members stay
    unmerged. Skipping is the conservative failure: it splits rather than
    guesses.
    """
    for block_key, members in sorted(blocks.items()):
        if len(members) < 2:
            continue
        if len(members) > thresholds.max_block_size:
            oversized.append(
                OversizedBlock(block_kind=kind, block_key=block_key, size=len(members))
            )
            logger.warning(
                "Skipping oversized %s block %s with %d members", kind, block_key, len(members)
            )
            continue
        ordered = sorted(members)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                yield ordered[i], ordered[j]


def candidate_pairs(
    nodes: Sequence[Node],
    thresholds: Thresholds,
    blacklist: frozenset[str] = frozenset(),
) -> tuple[list[tuple[str, str]], list[OversizedBlock]]:
    """Generate the deduplicated, canonically ordered set of pairs to evaluate.

    Pairs are always (smaller node_id, larger node_id) and are returned sorted,
    so evaluation order — and therefore the audit table — does not depend on
    which block first produced a pair.
    """
    oversized: list[OversizedBlock] = []
    seen: set[tuple[str, str]] = set()

    for kind, blocks in (
        ("spatial", spatial_blocks(nodes)),
        ("coordinate", coordinate_blocks(nodes, blacklist)),
        ("license", license_blocks(nodes)),
    ):
        for pair in _pairs_from_blocks(blocks, kind, thresholds, oversized):
            seen.add(pair)

    pairs = sorted(seen)
    logger.info(
        "Generated %d candidate pairs from %d nodes (%d oversized blocks skipped)",
        len(pairs),
        len(nodes),
        len(oversized),
    )
    return pairs, oversized
