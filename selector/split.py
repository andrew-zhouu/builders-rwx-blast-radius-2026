"""Split selected tests over workers to minimise (largest load + 2 s per worker)."""
from __future__ import annotations

WORKER_MS = 2000
MAX_WORKERS = 8


def _lpt(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    loads = [0] * k
    bins: list[list[int]] = [[] for _ in range(k)]
    for num, ms in items:  # sorted by ms descending
        i = loads.index(min(loads))
        loads[i] += ms
        bins[i].append(num)
    return max(loads), bins


def best_split(durations: dict[int, int]) -> tuple[int, list[list[int]]]:
    """durations: test number -> milliseconds. Returns (cost_ms, workers)."""
    if not durations:
        return 0, []
    items = sorted(durations.items(), key=lambda kv: (-kv[1], kv[0]))
    best: tuple[int, list[list[int]]] | None = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        load, bins = _lpt(items, k)
        cost = load + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, [b for b in bins if b])
    assert best is not None
    return best


def split_cost(durations: dict[int, int]) -> int:
    return best_split(durations)[0]
