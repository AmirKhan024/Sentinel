"""Disjoint-set union over node identifiers.

Written out rather than pulled in from networkx or scipy. The algorithm is
forty lines, it is the only graph operation Component 2 needs, and keeping it
here means the merge step has no dependency whose version could change the
grouping between runs.

Determinism note: connected components are a property of the edge set and are
therefore independent of union order, but the *representative* chosen for a
component is not. ``components()`` re-labels every component by its minimum
member so the output cannot depend on the order edges arrived in.
"""

from __future__ import annotations

from collections.abc import Iterable


class UnionFind:
    """Union-find with path compression and union by size."""

    __slots__ = ("_parent", "_size")

    def __init__(self, items: Iterable[str]) -> None:
        self._parent: dict[str, str] = {item: item for item in items}
        self._size: dict[str, int] = dict.fromkeys(self._parent, 1)

    def __contains__(self, item: str) -> bool:
        return item in self._parent

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        """Return the representative of an item's set, compressing the path."""
        if item not in self._parent:
            raise KeyError(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Second pass rather than recursion: cluster chains are shallow, but a
        # 314k-row snapshot should never be able to hit a recursion limit.
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        """Merge the sets containing two items. Idempotent."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._size[left_root] < self._size[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]

    def components(self) -> dict[str, list[str]]:
        """Return components keyed by their minimum member, each member sorted.

        Keying by the minimum rather than by whatever the union-by-size
        heuristic left as root is what makes the result independent of input
        order.
        """
        groups: dict[str, list[str]] = {}
        for item in self._parent:
            groups.setdefault(self.find(item), []).append(item)
        return {min(members): sorted(members) for members in groups.values()}
