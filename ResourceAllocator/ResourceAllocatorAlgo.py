import random
from typing import Optional
from datetime import datetime, timedelta


class ResourceAllocatorAlgo:
    """
    I used these strategies:
    - random
    - round_robin
    - earliest_available
    """

    def __init__(
            self,
            resource_pool: list,
            permissions: Optional[dict] = None,
            availability_model=None,
            strategy: str = "random",
            sim_start_time: Optional[datetime] = None
    ):
        self.resource_pool = [str(r) for r in resource_pool]
        self.permissions = permissions or {}
        self.availability_model = availability_model
        self.strategy = strategy

        self.round_robin_index = 0
        self.resource_next_free = {
            r: (sim_start_time if sim_start_time is not None else datetime.min)
            for r in self.resource_pool
        }

    def get_allowed_resources(self, activity: str) -> list:
        if not self.permissions:
            return list(self.resource_pool)

        parts = [p.strip() for p in activity.split("&")]
        allowed_sets = []

        for p in parts:
            allowed = set(str(r) for r in self.permissions.get(p, []))
            allowed_sets.append(allowed)

        if not allowed_sets:
            return list(self.resource_pool)

        allowed = set.intersection(*allowed_sets) if allowed_sets else set()
        return [r for r in self.resource_pool if r in allowed]

    def feasible_start(self, resource: str, ready_time: datetime) -> datetime:
        candidate_time = max(ready_time, self.resource_next_free.get(resource, ready_time))

        if self.availability_model is not None:
            return self.availability_model.next_available(resource, candidate_time)

        return candidate_time

    def assign(self, activity: str, ready_time: datetime, duration: timedelta):
        candidates = self.get_allowed_resources(activity)

        if not candidates:
            return None

        if self.strategy == "random":
            r = random.choice(candidates)

        elif self.strategy == "round_robin":
            ordered = sorted(candidates)
            r = ordered[self.round_robin_index % len(ordered)]
            self.round_robin_index += 1

        elif self.strategy == "earliest_available":
            r = min(
                candidates,
                key=lambda res: self.feasible_start(res, ready_time)
            )

        else:
            raise ValueError(f"Unknown resource allocation strategy: {self.strategy}")

        planned_start = ready_time
        actual_start = self.feasible_start(r, ready_time)
        finish_time = actual_start + duration
        delay_seconds = (actual_start - planned_start).total_seconds()
        was_delayed = delay_seconds > 0

        self.resource_next_free[r] = finish_time

        return {
            "resource": r,
            "planned_start": planned_start,
            "actual_start": actual_start,
            "finish_time": finish_time,
            "delay_seconds": delay_seconds,
            "was_delayed": was_delayed,
        }