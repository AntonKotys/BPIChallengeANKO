"""
K-Batching Resource Allocator (Task 2.1 - Batch Allocation)

Implements the K-Batching approach from:
D. D. Zeng and J. L. Zhao, "Effective role resolution in workflow management,"
INFORMS J. Comput., vol. 17, no. 3, pp. 374-387, 2005.

Accumulates K tasks in a pending queue, then solves a Parallel Machines
Scheduling Problem to optimally assign all K tasks to available resources.

Two batch solving strategies are provided:
  - "lpt" (Longest Processing Time first): greedy 4/3-approximation for
    makespan minimization on parallel machines.
  - "min_cost": optimal one-to-one assignment via the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment). Falls back to LPT when the
    batch size exceeds the number of available resources.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict


class BatchAllocator:
    """
    K-Batching resource allocation strategy.

    Wraps a base ResourceAllocatorAlgo instance for shared resource tracking,
    permission resolution, and availability-aware feasible start computation.

    Parameters
    ----------
    base_allocator : ResourceAllocatorAlgo
        Base allocator whose resource_next_free, permissions, and availability
        model are reused.
    batch_size_k : int
        Number of tasks to accumulate before solving the batch.
    max_wait_seconds : float or None
        Maximum time (in simulation seconds) a task may wait in the queue
        before a flush is forced, even if the batch is not full. Prevents
        starvation. None means no time-based flushing.
    batch_strategy : str
        Solving strategy: "lpt" (default) or "min_cost".
    """

    def __init__(
        self,
        base_allocator,
        batch_size_k: int = 5,
        max_wait_seconds: Optional[float] = None,
        batch_strategy: str = "lpt",
    ):
        self.base = base_allocator
        self.batch_size_k = batch_size_k
        self.max_wait_seconds = max_wait_seconds
        self.batch_strategy = batch_strategy
        self.pending_tasks: List[Dict] = []
        self.stats = {
            "batches_solved": 0,
            "tasks_assigned": 0,
            "forced_flushes": 0,
            "total_tasks_submitted": 0,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def submit_task(
        self,
        activity: str,
        ready_time: datetime,
        duration: timedelta,
        case_id: str,
    ):
        """Add a task to the pending batch queue."""
        self.pending_tasks.append({
            "activity": activity,
            "ready_time": ready_time,
            "duration": duration,
            "case_id": case_id,
        })
        self.stats["total_tasks_submitted"] += 1

    def should_flush(self, current_time: datetime) -> bool:
        """Return True if the batch should be flushed now."""
        if len(self.pending_tasks) >= self.batch_size_k:
            return True
        if self.max_wait_seconds is not None and self.pending_tasks:
            oldest = min(t["ready_time"] for t in self.pending_tasks)
            if (current_time - oldest).total_seconds() >= self.max_wait_seconds:
                return True
        return False

    def flush_batch(self, force: bool = False) -> List[Optional[Dict]]:
        """
        Solve the batch assignment and return a list of assignment dicts.

        Each dict has the same keys as ResourceAllocatorAlgo.assign() plus
        "case_id" and "activity" to identify the originating task.
        Returns None entries for tasks that could not be assigned.
        """
        if not self.pending_tasks:
            return []
        if not force and len(self.pending_tasks) < self.batch_size_k:
            return []

        tasks = self.pending_tasks[:]
        self.pending_tasks.clear()

        if force:
            self.stats["forced_flushes"] += 1

        if self.batch_strategy == "min_cost":
            results = self._solve_min_cost(tasks)
        else:
            results = self._solve_lpt(tasks)

        self.stats["batches_solved"] += 1
        self.stats["tasks_assigned"] += sum(1 for r in results if r is not None)

        return results

    def get_stats(self) -> Dict:
        """Return batch allocation statistics."""
        return dict(self.stats)

    # ------------------------------------------------------------------
    # Batch solving strategies
    # ------------------------------------------------------------------

    def _solve_lpt(self, tasks: list) -> list:
        """
        Longest Processing Time first greedy scheduling.

        Sort tasks by decreasing duration and greedily assign each to the
        resource with the earliest feasible completion time.  This is a
        well-known 4/3-approximation for makespan minimization on identical
        parallel machines (Graham 1969).
        """
        sorted_indices = sorted(
            range(len(tasks)),
            key=lambda i: tasks[i]["duration"].total_seconds(),
            reverse=True,
        )

        results: List[Optional[Dict]] = [None] * len(tasks)

        for idx in sorted_indices:
            task = tasks[idx]
            candidates = self.base.get_allowed_resources(task["activity"])
            if not candidates:
                continue

            best_res = None
            best_finish = None
            best_start = None

            for res in candidates:
                start = self.base.feasible_start(res, task["ready_time"])
                finish = start + task["duration"]
                if best_finish is None or finish < best_finish:
                    best_res = res
                    best_finish = finish
                    best_start = start

            if best_res is None:
                continue

            planned_start = task["ready_time"]
            delay_seconds = (best_start - planned_start).total_seconds()

            self.base.resource_next_free[best_res] = best_finish

            results[idx] = {
                "resource": best_res,
                "planned_start": planned_start,
                "actual_start": best_start,
                "finish_time": best_finish,
                "delay_seconds": delay_seconds,
                "was_delayed": delay_seconds > 0,
                "case_id": task["case_id"],
                "activity": task["activity"],
            }

        return results

    def _solve_min_cost(self, tasks: list) -> list:
        """
        Minimum total completion time via the Hungarian algorithm.

        Builds a cost matrix where cost[i][j] = expected completion time
        of task i on resource j.  Uses scipy.optimize.linear_sum_assignment
        for optimal one-to-one matching.

        Falls back to LPT when the batch size exceeds the number of
        available resources (Hungarian requires one-to-one matching).
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return self._solve_lpt(tasks)

        task_candidate_sets = []
        all_candidates_set = set()
        for task in tasks:
            cands = set(self.base.get_allowed_resources(task["activity"]))
            task_candidate_sets.append(cands)
            all_candidates_set.update(cands)

        unique_resources = sorted(all_candidates_set)
        n_tasks = len(tasks)
        n_resources = len(unique_resources)

        if n_resources == 0:
            return [None] * n_tasks

        if n_tasks > n_resources:
            return self._solve_lpt(tasks)

        INF = 1e15
        max_dim = max(n_tasks, n_resources)
        cost = np.full((max_dim, max_dim), INF)

        start_cache = {}
        for i, task in enumerate(tasks):
            for j, res in enumerate(unique_resources):
                if res in task_candidate_sets[i]:
                    start = self.base.feasible_start(res, task["ready_time"])
                    finish = start + task["duration"]
                    cost[i][j] = (finish - task["ready_time"]).total_seconds()
                    start_cache[(i, j)] = start

        row_ind, col_ind = linear_sum_assignment(cost)

        assignment_map = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_tasks:
                assignment_map[r] = c

        results: List[Optional[Dict]] = [None] * n_tasks
        for i, task in enumerate(tasks):
            j = assignment_map.get(i)
            if j is None or j >= n_resources or cost[i][j] >= INF:
                continue

            res = unique_resources[j]
            planned_start = task["ready_time"]
            actual_start = start_cache.get(
                (i, j), self.base.feasible_start(res, planned_start)
            )
            finish_time = actual_start + task["duration"]
            delay_seconds = (actual_start - planned_start).total_seconds()

            self.base.resource_next_free[res] = finish_time

            results[i] = {
                "resource": res,
                "planned_start": planned_start,
                "actual_start": actual_start,
                "finish_time": finish_time,
                "delay_seconds": delay_seconds,
                "was_delayed": delay_seconds > 0,
                "case_id": task["case_id"],
                "activity": task["activity"],
            }

        return results
