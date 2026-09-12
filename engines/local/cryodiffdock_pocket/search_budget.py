"""Shared cohort-level budget helpers for matched search experiments.

Candidate budgets include the original input poses.  A cohort-wide budget is
allocated deterministically: every complex keeps all of its original poses.
The remaining allowance is either distributed evenly or scaled in proportion
to a reference full run's native candidate capacities.  This makes the *total*
number of scored candidates the controlled quantity without requiring equal
per-complex pool sizes.
"""

from __future__ import annotations

from collections.abc import Sequence


def allocate_total_candidate_budget(
    total_budget: int | None,
    original_counts: Sequence[int],
    *,
    capacities: Sequence[int] | None = None,
) -> list[int | None]:
    """Return a per-complex cap whose sum is exactly ``total_budget``.

    ``None`` means unlimited.  Every cap includes that complex's original
    candidates, so a budget smaller than the original pool is rejected rather
    than silently dropping diffusion predictions.
    """

    counts = [int(value) for value in original_counts]
    if not counts:
        return []
    if any(value <= 0 for value in counts):
        raise ValueError("original candidate counts must be positive")
    if total_budget is None:
        return [None] * len(counts)
    total = int(total_budget)
    if total <= 0:
        raise ValueError("total candidate budget must be positive")
    minimum = sum(counts)
    if total < minimum:
        raise ValueError(
            f"total candidate budget {total} is smaller than the {minimum} "
            "required original candidates"
        )
    if capacities is None:
        quotient, remainder = divmod(total - minimum, len(counts))
        return [
            count + quotient + int(index < remainder)
            for index, count in enumerate(counts)
        ]

    caps = [int(value) for value in capacities]
    if len(caps) != len(counts):
        raise ValueError("capacity and original-count lengths differ")
    if any(capacity < original for capacity, original in zip(caps, counts)):
        raise ValueError("candidate capacity cannot be smaller than original count")
    if total > sum(caps):
        raise ValueError(
            f"total candidate budget {total} exceeds native capacity {sum(caps)}"
        )
    target_extra = total - minimum
    capacity_extra = [capacity - original for capacity, original in zip(caps, counts)]
    available_extra = sum(capacity_extra)
    if target_extra == 0:
        return counts
    if available_extra <= 0:
        raise ValueError("native candidate capacity contains no generated poses")
    exact = [target_extra * value / available_extra for value in capacity_extra]
    allocated_extra = [
        min(capacity, int(exact_value))
        for capacity, exact_value in zip(capacity_extra, exact)
    ]
    remaining = target_extra - sum(allocated_extra)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - int(exact[index])), index),
    )
    for index in order:
        if remaining <= 0:
            break
        if allocated_extra[index] < capacity_extra[index]:
            allocated_extra[index] += 1
            remaining -= 1
    if remaining:
        raise AssertionError("largest-remainder candidate allocation did not close")
    return [
        original + extra for original, extra in zip(counts, allocated_extra)
    ]


def per_complex_wall_clock_budget(
    total_budget_seconds: float | None,
    num_complexes: int,
) -> float | None:
    """Split a cohort wall-clock ceiling evenly across manifest rows."""

    if total_budget_seconds is None:
        return None
    total = float(total_budget_seconds)
    if total <= 0:
        raise ValueError("total wall-clock budget must be positive")
    if int(num_complexes) <= 0:
        raise ValueError("num_complexes must be positive")
    return total / int(num_complexes)
