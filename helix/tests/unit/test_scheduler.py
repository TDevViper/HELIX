import pytest

from helix.control.memory_monitor import MemoryMonitor
from helix.control.scheduler import PreemptiveScheduler
from helix.models.request import HelixRequest, UserTier


@pytest.mark.asyncio
async def test_enqueue_and_queue_depth():
    monitor = MemoryMonitor()
    sched = PreemptiveScheduler(memory_monitor=monitor)
    req = HelixRequest(model="llama3", messages=[])
    await sched.enqueue(req)
    assert sched.queue_depth() == 1


@pytest.mark.asyncio
async def test_priority_ordering():
    monitor = MemoryMonitor()
    sched = PreemptiveScheduler(memory_monitor=monitor)
    low = HelixRequest(model="llama3", messages=[], user_tier=UserTier.FREE)
    high = HelixRequest(model="llama3", messages=[], user_tier=UserTier.PREMIUM)
    await sched.enqueue(low)
    await sched.enqueue(high)
    _, _, top = sched._queue[0]
    assert top.priority_score() >= low.priority_score()
