"""BST over segments keyed by AvgLP_K, supporting O(log N) FindNearest.

PLAN.md line 33: `Insert(BST, key=AvgLP_K, value=s)`. `FindNearest` returns the
segment with the closest key to a query AvgLP_K value.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, Tuple

try:
    from sortedcontainers import SortedList  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sortedcontainers is required for GEAR segment index. "
        "Install with `pip install sortedcontainers`."
    ) from exc


class SegmentBST:
    """Sorted list of (key, segment_id) pairs.

    Ties on key are allowed; FindNearest returns the closest by absolute key
    difference, breaking ties by insertion order.
    """

    def __init__(self):
        self._items: SortedList = SortedList(key=lambda kv: kv[0])
        self._lock = threading.Lock()

    def insert(self, key: float, segment_id: str) -> None:
        with self._lock:
            self._items.add((float(key), segment_id))

    def find_nearest(self, key: float) -> Optional[Tuple[float, str]]:
        """Return (key, segment_id) of the entry closest to `key`, or None if empty."""

        with self._lock:
            if len(self._items) == 0:
                return None
            target = (float(key), "")
            idx = self._items.bisect_left(target)
            candidates = []
            if idx < len(self._items):
                candidates.append(self._items[idx])
            if idx - 1 >= 0:
                candidates.append(self._items[idx - 1])
            best = min(candidates, key=lambda kv: abs(kv[0] - float(key)))
            return best

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
