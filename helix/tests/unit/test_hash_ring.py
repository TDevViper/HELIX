from helix.control.hash_ring import ConsistentHashRing
from helix.models.worker import BackendType, WorkerNode


def _make_worker(wid: str) -> WorkerNode:
    return WorkerNode(worker_id=wid, base_url=f"http://{wid}", backend_type=BackendType.OLLAMA)


def test_add_and_lookup():
    ring = ConsistentHashRing(vnodes=50)
    ring.add_node(_make_worker("w1"))
    ring.add_node(_make_worker("w2"))
    assert ring.get_node("session-abc") in {"w1", "w2"}


def test_session_affinity():
    ring = ConsistentHashRing(vnodes=50)
    ring.add_node(_make_worker("w1"))
    ring.add_node(_make_worker("w2"))
    key = "session-xyz"
    assert ring.get_node(key) == ring.get_node(key)


def test_remove_node():
    ring = ConsistentHashRing(vnodes=50)
    ring.add_node(_make_worker("w1"))
    ring.add_node(_make_worker("w2"))
    ring.remove_node("w1")
    assert ring.get_node("anything") == "w2"


def test_empty_ring_returns_none():
    ring = ConsistentHashRing()
    assert ring.get_node("key") is None
