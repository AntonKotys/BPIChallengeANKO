"""
Score-based Value Function Approximation (SVFA) for Resource Allocation
(Task 2.1 Advanced - Learning-based Allocation)

Implements the SVFA approach from:
J. Middelhuis, R. Lo Bianco, E. Sherzer, Z. Bukhsh, I. Adan, R. M. Dijkman,
"Learning policies for resource allocation in business processes,"
Inf. Syst., vol. 128, art. 102492, 2025.

The allocator scores every possible (resource, task) pair using a learned
linear combination of hand-crafted features, then assigns the pair with
the lowest score.  Weights are optimized via Bayesian optimization.

Score function (Equation 2 of the paper):
    Score(r, k) = w1 * MeanAssignment(r, k)
                + w2 * VarAssignment(r, k)
                + w3 * ActivityRank(r, k)
                + w4 * ResourceRank(r, k)
                - w5 * ProbFin(r, k)
                - w6 * QueueLength(k)

Features:
    f1  MeanAssignment  - expected processing time of assigning r to k
    f2  VarAssignment   - variance of that processing time
    f3  ActivityRank    - rank of task k among all waiting tasks for resource r
    f4  ResourceRank    - rank of resource r among available resources for task k
    f5  ProbFin         - probability that case c finishes after activity k
    f6  QueueLength     - number of waiting tasks of the same activity type

If no assignment's score is below threshold w7, the agent POSTPONES.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

from scipy.stats import lognorm, gamma, weibull_min


_DIST_MAP = {"lognorm": lognorm, "gamma": gamma, "weibull_min": weibull_min}


class SVFAllocator:
    """
    SVFA resource allocator.

    Wraps a base ResourceAllocatorAlgo for shared resource tracking.

    Parameters
    ----------
    base_allocator : ResourceAllocatorAlgo
        Provides resource_next_free, permissions, and availability model.
    weights : list[float]
        Seven weights [w1 .. w7].  w1-w6 scale features, w7 is the
        postpone threshold.
    processing_stats : dict
        {activity_name: {"mean": float, "var": float}} or
        {(resource, activity): {"mean": float, "var": float}}.
        If keyed by activity only, every resource is assumed identical.
    prob_fin_map : dict
        {activity_name: float} probability the case finishes after this
        activity (ProbFin feature).
    """

    DEFAULT_WEIGHTS = [1.0, 0.5, 0.5, 0.5, 2.0, 1.0, 1e9]

    def __init__(
        self,
        base_allocator,
        weights: Optional[List[float]] = None,
        processing_stats: Optional[Dict] = None,
        prob_fin_map: Optional[Dict[str, float]] = None,
    ):
        self.base = base_allocator
        self.weights = list(weights) if weights else list(self.DEFAULT_WEIGHTS)
        self.processing_stats = processing_stats or {}
        self.prob_fin_map = prob_fin_map or {}
        self.pending_tasks: List[Dict] = []
        self.stats = {
            "assignments_made": 0,
            "postponements": 0,
            "decision_steps": 0,
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
        """Add a task to the waiting queue."""
        self.pending_tasks.append({
            "activity": activity,
            "ready_time": ready_time,
            "duration": duration,
            "case_id": case_id,
        })
        self.stats["total_tasks_submitted"] += 1

    def make_assignments(self, current_time: datetime) -> List[Dict]:
        """
        Score all possible (resource, task) pairs, then greedily assign
        best-scoring pairs.  Pre-computes ranks and caches lookups to
        avoid the O(n^2 * m) bottleneck of per-pair rank iteration.

        Returns a list of assignment dicts compatible with the engine's
        schedule_event().
        """
        self.stats["decision_steps"] += 1

        available = set(self._get_available_resources(current_time))
        if not available or not self.pending_tasks:
            return []

        w = self.weights

        # Step 1: Cache allowed resources per activity type
        allowed_cache: Dict[str, set] = {}
        unique_activities: set = set()
        for task in self.pending_tasks:
            act = task["activity"]
            unique_activities.add(act)
            if act not in allowed_cache:
                allowed_cache[act] = set(self.base.get_allowed_resources(act))

        # Step 2: Cache mean/var for eligible (resource, activity) pairs
        mean_cache: Dict[tuple, float] = {}
        var_cache: Dict[tuple, float] = {}
        for act in unique_activities:
            eligible = allowed_cache[act] & available
            for res in eligible:
                key = (res, act)
                mean_cache[key] = self._get_mean(res, act)
                var_cache[key] = self._get_var(res, act)

        # Step 3: Queue lengths per activity type
        queue_lengths: Dict[str, int] = {}
        for t in self.pending_tasks:
            queue_lengths[t["activity"]] = queue_lengths.get(t["activity"], 0) + 1

        # Step 4: ProbFin per activity (handles compound activities via "&")
        prob_fin_cache: Dict[str, float] = {}
        for act in unique_activities:
            pf = self.prob_fin_map.get(act, 0.0)
            for part in act.split("&"):
                p = part.strip()
                if p in self.prob_fin_map and p != act:
                    pf = max(pf, self.prob_fin_map[p])
            prob_fin_cache[act] = pf

        # Step 5: ActivityRank — rank tasks by mean processing time per resource
        activity_ranks: Dict[tuple, int] = {}
        for res in available:
            acts_for_res = []
            for act in unique_activities:
                key = (res, act)
                if key in mean_cache:
                    acts_for_res.append((mean_cache[key], act))
            acts_for_res.sort()
            for rank, (_, act) in enumerate(acts_for_res, 1):
                activity_ranks[(res, act)] = rank

        # Step 6: ResourceRank — rank resources by mean processing time per activity
        resource_ranks: Dict[tuple, int] = {}
        for act in unique_activities:
            ress_for_act = []
            for res in (allowed_cache[act] & available):
                key = (res, act)
                if key in mean_cache:
                    ress_for_act.append((mean_cache[key], res))
            ress_for_act.sort()
            for rank, (_, res) in enumerate(ress_for_act, 1):
                resource_ranks[(res, act)] = rank

        # Step 7: Score every (resource, task) pair
        scored_pairs = []
        for task_idx, task in enumerate(self.pending_tasks):
            act = task["activity"]
            pf = prob_fin_cache.get(act, 0.0)
            ql = queue_lengths.get(act, 0)
            eligible = allowed_cache[act] & available
            for res in eligible:
                key = (res, act)
                score = (
                    w[0] * mean_cache[key]
                    + w[1] * var_cache[key]
                    + w[2] * activity_ranks.get(key, 1)
                    + w[3] * resource_ranks.get(key, 1)
                    - w[4] * pf
                    - w[5] * ql
                )
                scored_pairs.append((score, task_idx, res))

        scored_pairs.sort()

        # Step 8: Greedy assignment — best score first, skip conflicts
        assigned_resources: set = set()
        assigned_task_indices: set = set()
        assignments: List[Dict] = []

        for score, task_idx, res in scored_pairs:
            if score >= w[6]:
                self.stats["postponements"] += 1
                break
            if res in assigned_resources or task_idx in assigned_task_indices:
                continue

            task = self.pending_tasks[task_idx]
            planned_start = task["ready_time"]
            actual_start = self.base.feasible_start(res, planned_start)
            if actual_start < current_time:
                actual_start = current_time
            finish_time = actual_start + task["duration"]
            delay_seconds = (actual_start - planned_start).total_seconds()

            self.base.resource_next_free[res] = finish_time

            assignments.append({
                "resource": res,
                "planned_start": planned_start,
                "actual_start": actual_start,
                "finish_time": finish_time,
                "delay_seconds": delay_seconds,
                "was_delayed": delay_seconds > 0,
                "case_id": task["case_id"],
                "activity": task["activity"],
            })
            self.stats["assignments_made"] += 1
            assigned_resources.add(res)
            assigned_task_indices.add(task_idx)

        for idx in sorted(assigned_task_indices, reverse=True):
            self.pending_tasks.pop(idx)

        return assignments

    def force_assign_remaining(self, current_time: datetime) -> List[Dict]:
        """
        Assign all remaining pending tasks using earliest-available logic
        (no postponement).  Called at the end of the simulation to flush
        any tasks still waiting in the queue.
        """
        assignments: List[Dict] = []
        while self.pending_tasks:
            task = self.pending_tasks[0]
            candidates = self.base.get_allowed_resources(task["activity"])
            if not candidates:
                self.pending_tasks.pop(0)
                continue

            best_res = min(
                candidates,
                key=lambda r: self.base.feasible_start(r, task["ready_time"]),
            )
            self.pending_tasks.pop(0)

            planned_start = task["ready_time"]
            actual_start = self.base.feasible_start(best_res, planned_start)
            finish_time = actual_start + task["duration"]
            delay_seconds = (actual_start - planned_start).total_seconds()

            self.base.resource_next_free[best_res] = finish_time

            assignments.append({
                "resource": best_res,
                "planned_start": planned_start,
                "actual_start": actual_start,
                "finish_time": finish_time,
                "delay_seconds": delay_seconds,
                "was_delayed": delay_seconds > 0,
                "case_id": task["case_id"],
                "activity": task["activity"],
            })
        return assignments

    def get_stats(self) -> Dict:
        return dict(self.stats)

    # ------------------------------------------------------------------
    # Score computation (Section 4.2 of the paper)
    # ------------------------------------------------------------------

    def _compute_score(
        self, resource: str, task: Dict, available_resources: List[str]
    ) -> float:
        w = self.weights
        activity = task["activity"]

        mean_pt = self._get_mean(resource, activity)
        var_pt = self._get_var(resource, activity)

        activity_rank = self._activity_rank(resource, task)
        resource_rank = self._resource_rank(resource, task, available_resources)

        prob_fin = self.prob_fin_map.get(activity, 0.0)
        for part in activity.split("&"):
            p = part.strip()
            if p in self.prob_fin_map and p != activity:
                prob_fin = max(prob_fin, self.prob_fin_map[p])

        queue_length = sum(
            1 for t in self.pending_tasks if t["activity"] == activity
        )

        score = (
            w[0] * mean_pt
            + w[1] * var_pt
            + w[2] * activity_rank
            + w[3] * resource_rank
            - w[4] * prob_fin
            - w[5] * queue_length
        )
        return score

    def _activity_rank(self, resource: str, task: Dict) -> int:
        """
        Rank activity instance `task` among all waiting tasks by
        MeanAssignment(resource, k') — lowest mean = rank 1.
        """
        current_mean = self._get_mean(resource, task["activity"])
        rank = 1
        for other in self.pending_tasks:
            if other is task:
                continue
            other_mean = self._get_mean(resource, other["activity"])
            if other_mean < current_mean:
                rank += 1
        return rank

    def _resource_rank(
        self, resource: str, task: Dict, available_resources: List[str]
    ) -> int:
        """
        Rank `resource` among all available resources by
        MeanAssignment(r', task.activity) — lowest mean = rank 1.
        """
        candidates = self.base.get_allowed_resources(task["activity"])
        eligible_available = [r for r in available_resources if r in candidates]
        current_mean = self._get_mean(resource, task["activity"])
        rank = 1
        for r in eligible_available:
            if r == resource:
                continue
            r_mean = self._get_mean(r, task["activity"])
            if r_mean < current_mean:
                rank += 1
        return rank

    def _get_mean(self, resource: str, activity: str) -> float:
        """Look up mean processing time for (resource, activity)."""
        key_ra = (resource, activity)
        if key_ra in self.processing_stats:
            return self.processing_stats[key_ra].get("mean", 60.0)
        if activity in self.processing_stats:
            return self.processing_stats[activity].get("mean", 60.0)
        parts = [p.strip() for p in activity.split("&")]
        total = 0.0
        for p in parts:
            key_rp = (resource, p)
            if key_rp in self.processing_stats:
                total += self.processing_stats[key_rp].get("mean", 60.0)
            elif p in self.processing_stats:
                total += self.processing_stats[p].get("mean", 60.0)
            else:
                total += 60.0
        return total

    def _get_var(self, resource: str, activity: str) -> float:
        """Look up variance of processing time for (resource, activity)."""
        key_ra = (resource, activity)
        if key_ra in self.processing_stats:
            return self.processing_stats[key_ra].get("var", 0.0)
        if activity in self.processing_stats:
            return self.processing_stats[activity].get("var", 0.0)
        parts = [p.strip() for p in activity.split("&")]
        total = 0.0
        for p in parts:
            key_rp = (resource, p)
            if key_rp in self.processing_stats:
                total += self.processing_stats[key_rp].get("var", 0.0)
            elif p in self.processing_stats:
                total += self.processing_stats[p].get("var", 0.0)
        return total

    def _get_available_resources(self, current_time: datetime) -> List[str]:
        """Return resources whose next_free time <= current_time."""
        return [
            r for r in self.base.resource_pool
            if self.base.resource_next_free.get(r, datetime.min) <= current_time
        ]

    # ------------------------------------------------------------------
    # Static helpers for building processing_stats and prob_fin_map
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stats_from_distributions(
        dist_json_path: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and variance per activity from distributions.json.
        Returns {activity: {"mean": ..., "var": ...}}.
        """
        with open(dist_json_path) as f:
            dist_data = json.load(f)

        stats: Dict[str, Dict[str, float]] = {}
        for act_name, info in dist_data.items():
            dist_name = info.get("best_dist")
            params = info.get("params", [])
            dist_cls = _DIST_MAP.get(dist_name)
            if dist_cls is None:
                stats[act_name] = {"mean": 60.0, "var": 0.0}
                continue
            try:
                mean_val = float(dist_cls.mean(*params))
                var_val = float(dist_cls.var(*params))
                if np.isnan(mean_val) or np.isinf(mean_val):
                    mean_val = 60.0
                if np.isnan(var_val) or np.isinf(var_val):
                    var_val = 0.0
            except Exception:
                mean_val, var_val = 60.0, 0.0
            stats[act_name] = {"mean": mean_val, "var": var_val}
        return stats

    @staticmethod
    def compute_stats_from_csv(
        csv_path: str,
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Compute mean and variance of processing times per (resource, activity)
        from an event log CSV.

        Approximates processing time as the time between consecutive events
        within the same case.  Returns {(resource, activity): {"mean", "var"}}.
        """
        df = pd.read_csv(csv_path)
        ts_col = "time:timestamp" if "time:timestamp" in df.columns else "timestamp"
        df[ts_col] = pd.to_datetime(df[ts_col], format="ISO8601", utc=True)
        df = df.sort_values(["case:concept:name", ts_col])

        df["_prev_ts"] = df.groupby("case:concept:name")[ts_col].shift(1)
        df["_duration_s"] = (df[ts_col] - df["_prev_ts"]).dt.total_seconds()
        df = df.dropna(subset=["_duration_s"])
        df = df[df["_duration_s"] > 0]

        stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (res, act), grp in df.groupby(["org:resource", "concept:name"]):
            vals = grp["_duration_s"]
            stats[(str(res), act)] = {
                "mean": float(vals.mean()),
                "var": float(vals.var()) if len(vals) > 1 else 0.0,
            }
        return stats

    @staticmethod
    def compute_prob_fin_from_csv(csv_path: str) -> Dict[str, float]:
        """
        Estimate ProbFin(activity) from historical data.

        ProbFin = fraction of traces where `activity` was the last event
        before the trace ended, relative to total occurrences of `activity`.
        """
        df = pd.read_csv(csv_path)
        ts_col = "time:timestamp" if "time:timestamp" in df.columns else "timestamp"
        df[ts_col] = pd.to_datetime(df[ts_col], format="ISO8601", utc=True)
        df = df.sort_values(["case:concept:name", ts_col])

        last_activities = df.groupby("case:concept:name")["concept:name"].last()
        last_counts = last_activities.value_counts()

        total_counts = df["concept:name"].value_counts()

        prob_fin: Dict[str, float] = {}
        for act in total_counts.index:
            n_last = last_counts.get(act, 0)
            n_total = total_counts[act]
            prob_fin[act] = float(n_last / n_total) if n_total > 0 else 0.0
        return prob_fin

    @staticmethod
    def compute_prob_fin_from_model(process_model: dict) -> Dict[str, float]:
        """
        Estimate ProbFin from the process model structure.

        If an activity has "END" among its successors, ProbFin =
        1 / len(successors).  Otherwise ProbFin = 0.
        """
        prob_fin: Dict[str, float] = {}
        for act, successors in process_model.items():
            if "END" in successors:
                prob_fin[act] = 1.0 / len(successors)
            else:
                prob_fin[act] = 0.0
        return prob_fin
