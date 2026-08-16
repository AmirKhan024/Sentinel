"""Union-find.

The determinism tests matter more than the correctness ones. Connected
components are order-independent by construction, but the *label* a component
gets is not, and a label that shifted between runs would shift establishment ids
with it.
"""

from __future__ import annotations

import random

from sentinel.entity.unionfind import UnionFind


def test_singletons_are_their_own_components() -> None:
    union = UnionFind(["a", "b", "c"])
    assert union.components() == {"a": ["a"], "b": ["b"], "c": ["c"]}


def test_simple_chain_forms_one_component() -> None:
    union = UnionFind(["a", "b", "c"])
    union.union("a", "b")
    union.union("b", "c")
    assert union.components() == {"a": ["a", "b", "c"]}


def test_disjoint_groups_stay_separate() -> None:
    union = UnionFind(["a", "b", "c", "d"])
    union.union("a", "b")
    union.union("c", "d")
    assert union.components() == {"a": ["a", "b"], "c": ["c", "d"]}


def test_union_is_idempotent() -> None:
    union = UnionFind(["a", "b"])
    union.union("a", "b")
    union.union("a", "b")
    union.union("b", "a")
    assert union.components() == {"a": ["a", "b"]}


def test_components_are_keyed_by_minimum_member() -> None:
    union = UnionFind(["z", "m", "a"])
    union.union("z", "m")
    union.union("m", "a")
    assert set(union.components()) == {"a"}


def test_component_members_are_sorted() -> None:
    union = UnionFind(["z", "m", "a"])
    union.union("z", "m")
    union.union("m", "a")
    assert union.components()["a"] == ["a", "m", "z"]


def test_find_raises_for_an_unknown_item() -> None:
    union = UnionFind(["a"])
    try:
        union.find("missing")
    except KeyError:
        return
    raise AssertionError("find() should raise KeyError for an unknown item")


def test_add_registers_a_new_item() -> None:
    union = UnionFind(["a"])
    union.add("b")
    assert "b" in union
    assert union.find("b") == "b"


def test_result_is_identical_under_shuffled_edge_order() -> None:
    """The property establishment ids ultimately depend on."""
    items = [f"n{i:02d}" for i in range(20)]
    edges = [("n00", "n01"), ("n01", "n02"), ("n05", "n06"), ("n10", "n11"), ("n11", "n05")]

    baseline = None
    rng = random.Random(20260816)
    for _ in range(10):
        shuffled = edges[:]
        rng.shuffle(shuffled)
        union = UnionFind(items)
        for left, right in shuffled:
            union.union(left, right)
        components = union.components()
        if baseline is None:
            baseline = components
        assert components == baseline


def test_result_is_identical_under_shuffled_item_order() -> None:
    items = [f"n{i:02d}" for i in range(10)]
    edges = [("n00", "n05"), ("n05", "n09")]

    forward = UnionFind(items)
    for left, right in edges:
        forward.union(left, right)

    backward = UnionFind(list(reversed(items)))
    for left, right in edges:
        backward.union(left, right)

    assert forward.components() == backward.components()
