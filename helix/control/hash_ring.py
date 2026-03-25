from __future__ import annotations

import bisect
import hashlib
import threading
from typing import Optional

from helix.models.worker import WorkerNode


class ConsistentHashRing:
    """
    Consistent hash ring with virtual nodes.

    Properties:
    - O(log N) lookup via bisect on a sorted ring
    - O(K/N) keys move when a node is added/removed (not O(K))
    - Session affinity: same key always maps to same node
    - Weighted nodes: higher weight = more vnodes = more traffic share
    - Thread-safe: all mutations hold a lock (reads are lock-free after copy)
    """

    def __init__(self, vnodes: int = 150) -> None:
        self._vnodes = vnodes
        # Sorted list of (hash_value, worker_id) tuples
        self._ring: list[tuple[int, str]] = []
        # worker_id -> WorkerNode
        self._nodes: dict[str, WorkerNode] = {}
        # Lock for ring mutations (add/remove node)
        self._lock = threading.RLock()

    # ── Hashing ───────────────────────────────────────────────

    @staticmethod
    def _hash(key: str) -> int:
        """SHA-256 based hash — uniform distribution, no clustering."""
        return int(hashlib.sha256(key.encode()).hexdigest(), 16)

    def _vnode_key(self, worker_id: str, index: int) -> str:
        return f"{worker_id}:vnode:{index}"

    # ── Mutations ─────────────────────────────────────────────

    def add_node(self, worker: WorkerNode) -> None:
        """
        Add a worker to the ring.
        Inserts (weight * vnodes) virtual nodes into the sorted ring.
        """
        with self._lock:
            if worker.worker_id in self._nodes:
                return  # Already registered
            self._nodes[worker.worker_id] = worker
            count = self._vnodes * worker.weight
            for i in range(count):
                h = self._hash(self._vnode_key(worker.worker_id, i))
                bisect.insort(self._ring, (h, worker.worker_id))

    def remove_node(self, worker_id: str) -> None:
        """
        Remove a worker from the ring.
        All its vnodes are deleted; traffic redistributes to neighbours.
        """
        with self._lock:
            if worker_id not in self._nodes:
                return
            worker = self._nodes.pop(worker_id)
            count = self._vnodes * worker.weight
            for i in range(count):
                h = self._hash(self._vnode_key(worker_id, i))
                # bisect to find exact position then verify before removing
                idx = bisect.bisect_left(self._ring, (h, worker_id))
                if idx < len(self._ring) and self._ring[idx] == (h, worker_id):
                    self._ring.pop(idx)

    # ── Lookup ────────────────────────────────────────────────

    def get_node(self, key: str) -> Optional[str]:
        """
        Return the worker_id responsible for this key.
        Walks clockwise from the key's hash position.
        Returns None if the ring is empty.
        """
        if not self._ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_left(self._ring, (h,))
        idx = idx % len(self._ring)
        return self._ring[idx][1]

    def get_nodes(self, key: str, n: int) -> list[str]:
        """
        Return up to n distinct worker_ids starting from key's position.
        Used for replication: first node is primary, rest are replicas.
        """
        if not self._ring:
            return []
        h = self._hash(key)
        idx = bisect.bisect_left(self._ring, (h,))
        seen: set[str] = set()
        result: list[str] = []
        for i in range(len(self._ring)):
            worker_id = self._ring[(idx + i) % len(self._ring)][1]
            if worker_id not in seen:
                seen.add(worker_id)
                result.append(worker_id)
            if len(result) >= n:
                break
        return result

    def get_worker(self, key: str) -> Optional[WorkerNode]:
        """Convenience: resolve key directly to WorkerNode."""
        worker_id = self.get_node(key)
        if worker_id is None:
            return None
        return self._nodes.get(worker_id)

    # ── Introspection ─────────────────────────────────────────

    def size(self) -> int:
        """Number of physical nodes in the ring."""
        return len(self._nodes)

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def all_worker_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def all_workers(self) -> list[WorkerNode]:
        return list(self._nodes.values())

    def rebalance_stats(self) -> dict[str, object]:
        """
        Show how evenly keys are distributed across nodes.
        Reports vnode count and % of ring owned per worker.
        Useful for verifying that virtual nodes give even distribution.
        """
        if not self._ring or not self._nodes:
            return {}

        total = len(self._ring)
        counts: dict[str, int] = {wid: 0 for wid in self._nodes}
        for _, worker_id in self._ring:
            counts[worker_id] += 1

        return {
            "total_vnodes": total,
            "physical_nodes": len(self._nodes),
            "distribution": {
                wid: {
                    "vnodes": count,
                    "ring_pct": round(count / total * 100, 2),
                }
                for wid, count in counts.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"ConsistentHashRing("
            f"nodes={len(self._nodes)}, "
            f"vnodes_per_node={self._vnodes}, "
            f"ring_size={len(self._ring)})"
        )
