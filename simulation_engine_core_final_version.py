import heapq
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import json
import numpy as np
import random

import os

from sklearn.cluster import AgglomerativeClustering

from datetime import timedelta
from scipy.stats import lognorm, gamma, weibull_min

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from ResourceAllocator.ResourceAllocatorAlgo import ResourceAllocatorAlgo
from ResourceAllocator.BatchAllocator import BatchAllocator
from ResourceAllocator.SVFAllocator import SVFAllocator
# Import Task 1.4 module


from task_1_4_next_activity import ExpertActivityPredictor
from task_1_5_rolling_stochastic_availability import RollingStochasticAvailabilityModel
from DynamicSpawnRates.DynamicArrivalModel import DynamicArrivalModel, fit_dynamic_arrival_model

# ============================================================
# CONFIGURATION (minimal)
# ============================================================

SIM_START_TIME = datetime(2016, 2, 1, 10, 0, 0)
START_ACTIVITY = "A_Create Application"
NUM_CASES = 100
MAX_EVENTS_PER_CASE = 200
LAMBDA_RATE = 1 / 1089.18
MAX_OFFERS_PER_CASE = 3
with open("distributions.json", "r") as f:
    DIST_DATA = json.load(f)

DIST_MAP = {
    "lognorm": lognorm,
    "gamma": gamma,
    "weibull_min": weibull_min
}
# Resource permissions configuration
USE_ADVANCED_PERMISSIONS = True

output_dir = "sim_outputs"
os.makedirs(output_dir, exist_ok=True)

# Global seed for reproducibility of random behaviour
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
print(f"[SEED] Global seed set to {GLOBAL_SEED}")

# ============================================================
# ROUTING MODEL
# ============================================================

PROCESS_MODEL = {
    # --- Start ---
    "A_Create Application": [
        "A_Submitted",
        "W_Complete application & A_Concept"
    ],

    "A_Submitted": [
        "W_Handle leads",
        "W_Complete application & A_Concept"
    ],

    "W_Handle leads": [
        "W_Complete application & A_Concept"
    ],

    "W_Complete application & A_Concept": [
        "A_Accepted"
    ],

    "A_Accepted": [
        "O_Create Offer & O_Created"
    ],

    "O_Create Offer & O_Created": [
        "O_Sent (mail and online)"
    ],

    "O_Sent (mail and online)": [
        "W_Call after offers & A_Complete",
        "O_Create Offer & O_Created"
    ],

    "W_Call after offers & A_Complete": [
        "W_Validate application & A_Validating",
        "O_Create Offer & O_Created",
        "A_Cancelled & O_Cancelled"
    ],

    # --- Validation ---
    "W_Validate application & A_Validating": [
        "O_Returned"
    ],

    "O_Returned": [
        "A_Incomplete",
        "END"
    ],

    "A_Incomplete": [
        "A_Validating",
        "END"
    ],

    "A_Validating": [
        "A_Incomplete",
        "END"
    ],

    "A_Cancelled & O_Cancelled": [
        "END"
    ]
}
# ============================================================
# GATEWAY SEMANTICS
# ============================================================
GATEWAYS = {
    # XOR-splits (choose exactly 1 outgoing)
    "A_Create Application": "xor",
    "A_Submitted": "xor",
    "O_Returned": "xor",
    "A_Incomplete": "xor",
    "A_Validating": "xor",

    # Inclusive OR-split
    "W_Call after offers & A_Complete": "or",
    "O_Sent (mail and online)": "or",
}


# ============================================================
# 1.1 SIMULATION ENGINE CORE
# ============================================================

class SimEvent:
    """Single discrete event, ordered by timestamp for priority queue."""

    def __init__(
            self,
            time: datetime,
            case_id: str,
            activity: str,
            resource: Optional[str] = None,
            # --- NEW (debug for Task 1.5) ---
            planned_start: Optional[datetime] = None,  # ready time without availability
            actual_start: Optional[datetime] = None,  # after availability adjustment
            delay_seconds: float = 0.0,
            was_delayed: bool = False,
    ):
        self.time = time
        self.case_id = case_id
        self.activity = activity
        self.resource = resource

        # --- NEW (debug for Task 1.5) ---
        self.planned_start = planned_start
        self.actual_start = actual_start
        self.delay_seconds = delay_seconds
        self.was_delayed = was_delayed

    def __lt__(self, other):
        return self.time < other.time


class SimulationEngine:
    """
    1.1 Simulation Engine Core
    - maintains event queue
    - processes events in timestamp order
    - routes to next activities using PROCESS_MODEL
    - logs executed events
    """

    def __init__(self, process_model: dict, start_time: datetime, gateways: Optional[dict] = None,
                 predictor: Optional[ExpertActivityPredictor] = None,
                 availability_model: Optional[RollingStochasticAvailabilityModel] = None,
                 resource_pool: Optional[list] = None, permissions: Optional[dict] = None,
                 allocation_strategy: str = "random",
                 batch_size_k: int = 5, batch_max_wait_seconds: float = None,
                 svfa_weights: Optional[list] = None,
                 svfa_processing_stats: Optional[dict] = None,
                 svfa_prob_fin_map: Optional[dict] = None):
        self.model = process_model
        self.gateways = gateways or {}
        self.now = start_time
        self.event_queue = []
        self.log_rows = []
        self.case_context = {}
        # I added the change here from the previous file
        self.availability_model = availability_model
        self.resource_pool = resource_pool or []
        self.permissions = permissions or {}  # Task 1.6
        self.allocation_strategy = allocation_strategy

        # For k_batch / svfa, use earliest_available as base strategy for
        # non-batched calls (e.g. spawn_cases); queue-based allocation
        # happens only inside run()
        queue_strategies = ("k_batch", "svfa")
        base_strategy = (
            "earliest_available" if allocation_strategy in queue_strategies
            else allocation_strategy
        )
        self.allocator = ResourceAllocatorAlgo(
            resource_pool=self.resource_pool,
            permissions=self.permissions,
            availability_model=self.availability_model,
            strategy=base_strategy,
            sim_start_time=start_time
        )

        # K-Batch allocator wraps the base allocator (Task 2.1)
        self.batch_allocator = None
        if allocation_strategy == "k_batch":
            self.batch_allocator = BatchAllocator(
                base_allocator=self.allocator,
                batch_size_k=batch_size_k,
                max_wait_seconds=batch_max_wait_seconds,
            )

        # SVFA allocator wraps the base allocator (Task 2.1 Advanced)
        self.svfa_allocator = None
        if allocation_strategy == "svfa":
            self.svfa_allocator = SVFAllocator(
                base_allocator=self.allocator,
                weights=svfa_weights,
                processing_stats=svfa_processing_stats,
                prob_fin_map=svfa_prob_fin_map,
            )

        # TASK 1.4: Activity predictor for XOR branching
        self.predictor = predictor
        self.case_traces: Dict[str, List[str]] = {}  # Track trace per case

    def schedule_event(
            self,
            time: datetime,
            case_id: str,
            activity: str,
            resource: Optional[str] = None,
            # --- NEW (debug for Task 1.5) ---
            planned_start: Optional[datetime] = None,
            actual_start: Optional[datetime] = None,
            delay_seconds: float = 0.0,
            was_delayed: bool = False,
    ):
        heapq.heappush(
            self.event_queue,
            SimEvent(
                time=time,
                case_id=case_id,
                activity=activity,
                resource=resource,
                planned_start=planned_start,
                actual_start=actual_start,
                delay_seconds=delay_seconds,
                was_delayed=was_delayed,
            )
        )

    def log_event(self, case_id: str, activity: str, timestamp: datetime, resource: Optional[str],
                  # --- NEW (debug for Task 1.5) ---
                  planned_start: Optional[datetime] = None,
                  actual_start: Optional[datetime] = None,
                  delay_seconds: float = 0.0,
                  was_delayed: bool = False):
        self.log_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": timestamp.isoformat(),
            "org:resource": resource,

            # --- NEW (debug for Task 1.5) ---
            "planned_start": planned_start.isoformat() if planned_start is not None else None,
            "actual_start": actual_start.isoformat() if actual_start is not None else None,
            "delay_seconds": delay_seconds,
            "was_delayed": was_delayed,
        })

        # TASK 1.4: Track executed activities for prediction
        if case_id not in self.case_traces:
            self.case_traces[case_id] = []
        self.case_traces[case_id].append(activity)

    def route_next(self, activity: str, case_id: str) -> List[str]:
        """
        Decide which outgoing activities to schedule based on gateway semantics.

        TASK 1.4 INTEGRATION:
        For XOR gateways, uses ExpertActivityPredictor to sample next activity
        based on learned probabilities, instead of random.choice().

        for OR Gateways limit the number of offers up to 3 - cover 97.99% of historical cases

        Args:
            activity: Current activity that just completed
            case_id: Case identifier (needed to access trace history)

        Returns:
            List of activities to schedule next


        """
        outgoing = self.model.get(activity, [])
        if not outgoing:
            return []

        gateway_type = self.gateways.get(activity)

        # Get trace history for this case (TASK 1.4)
        trace = self.case_traces.get(case_id, [])

        # --- SPECIAL CASE: OR-split after sending offer : restrict the number of offer creations ---
        if activity == "O_Sent (mail and online)":
            # count offers already created in this case
            offer_count = sum(
                1 for a in trace
                if a.strip().startswith("O_Create Offer")
            )

            or_candidates = outgoing[:]  # both outgoing branches are candidates here

            # if offer limit reached, remove offer-creation branch
            if offer_count >= MAX_OFFERS_PER_CASE:
                or_candidates = [
                    a for a in or_candidates
                    if not a.strip().startswith("O_Create Offer")
                ]

            if not or_candidates:
                return []

            # inclusive OR among remaining branches
            k = random.randint(1, len(or_candidates))
            return random.sample(or_candidates, k)


        # --- SPECIAL CASE: OR-split + exclusive cancellation option ---
        if activity == "W_Call after offers & A_Complete": # !!! This part is not process model agnostic !!!
            cancel_act = "A_Cancelled & O_Cancelled"

            # --- offer loop bounding --- Anton
            offer_count = sum(
                1 for a in trace
                if a.startswith("O_Create Offer")
            )

            or_candidates = [a for a in outgoing if a != cancel_act]

            # if offer limit reached, remove offer-creation branch
            if offer_count >= MAX_OFFERS_PER_CASE:
                or_candidates = [
                    a for a in or_candidates
                    if not a.startswith("O_Create Offer")
                ]

            # decide cancel probability
            if self.predictor:
                dist = self.predictor.get_next_activity_distribution(trace, outgoing)
                # A_Cancelled after W_Call after offers in 8,539 cases of 191,092 occurrences in historical event log
                # 4.47% cancellation per occurrence of W_Call after offers
                cancel_prob = dist.get(cancel_act, 0.047) # !!! Default 0.2 must be customized properly !!! (Anton adjusted already)
            else:
                cancel_prob = 0.047

            # 1) cancel ONLY (exclusive)
            if cancel_act in outgoing and random.random() < cancel_prob:
                return [cancel_act]

            # 2) otherwise normal inclusive-OR among NON-cancel branches
            if not or_candidates:
                return []
            k = random.randint(1, len(or_candidates))
            return random.sample(or_candidates, k)

        # --- XOR GATEWAY: TASK 1.4 IMPLEMENTATION ---
        if gateway_type == "xor":
            if self.predictor:
                # Use predictor to sample based on learned probabilities
                sampled = self.predictor.sample_next_activity(
                    prefix_activities=trace,
                    enabled_next=outgoing
                )
                if sampled:
                    return [sampled]

            # Fallback to random if predictor unavailable or returns None
            return [random.choice(outgoing)]

        # --- OR GATEWAY: Random non-empty subset ---
        if gateway_type == "or":
            k = random.randint(1, len(outgoing))
            return random.sample(outgoing, k)

        # --- DEFAULT: Schedule all outgoing (sequence/parallel) ---
        return outgoing

    # ---------------------------
    # 1.1 Resource Allocation Heuristics
    # ---------------------------

    def allocate_resource(self, activity: str, t: datetime, duration: timedelta):
        """ Allocating a resource using the selected heuristic."""
        return self.allocator.assign(activity, t, duration)

    def run(self, duration_function):
        # Outer loop handles force-flush of remaining batch tasks at end
        run_again = True
        while run_again:
            run_again = False
            while self.event_queue:
                event = heapq.heappop(self.event_queue)
                self.now = event.time

                # init per-case context
                ctx = self.case_context.setdefault(
                    event.case_id,
                    {"event_count": 0, "ended": False, "open_tokens": 1}
                )

                # if case already ended, ignore everything
                if ctx["ended"]:
                    continue

                # log execution
                self.log_event(
                    event.case_id,
                    event.activity,
                    event.time,
                    event.resource,
                    planned_start=getattr(event, "planned_start", None),
                    actual_start=getattr(event, "actual_start", None),
                    delay_seconds=getattr(event, "delay_seconds", 0.0),
                    was_delayed=getattr(event, "was_delayed", False),
                )

                # IMPORTANT: consume one token for the event we just executed
                ctx["open_tokens"] -= 1

                # Safety (shouldn't normally happen if we stop scheduling END as an event)
                if event.activity == "END":
                    # Treat END as "this token finished" (we already decremented open_tokens above).
                    # Only end the case if this was the last active token.
                    if ctx["open_tokens"] == 0:
                        ctx["ended"] = True
                    continue

                ctx["event_count"] += 1
                # limit events per case (anti-infinite loop)
                if ctx["event_count"] > MAX_EVENTS_PER_CASE:
                    # terminate the case forcefully
                    ctx["ended"] = True
                    continue

                # route to next activities using gateway semantics
                # Route to next activities (TASK 1.4 applied here for XOR)
                next_activities = self.route_next(event.activity, event.case_id)

                # schedule successors (produce tokens)
                for next_act in next_activities:
                    # Treat END as token completion (DO NOT schedule END as an event)
                    if next_act == "END":
                        continue

                    dur = duration_function(next_act, event.time, ctx)

                    # K-Batch mode: submit task to batch queue instead of
                    # immediate assignment.  The token stays alive (open_tokens
                    # is incremented) and will be scheduled when the batch
                    # flushes via _flush_batch().
                    if self.batch_allocator is not None:
                        self.batch_allocator.submit_task(
                            next_act, event.time, dur, event.case_id
                        )
                        ctx["open_tokens"] += 1
                        continue

                    # SVFA mode: submit task to SVFA queue.  Assignments are
                    # made after all successors are queued via _svfa_assign().
                    if self.svfa_allocator is not None:
                        self.svfa_allocator.submit_task(
                            next_act, event.time, dur, event.case_id
                        )
                        ctx["open_tokens"] += 1
                        continue

                    """From here"""
                    assignment = self.allocate_resource(next_act, event.time, dur)

                    if assignment is None:
                        continue

                    r = assignment["resource"]
                    planned_start = assignment["planned_start"]
                    actual_start = assignment["actual_start"]
                    delay_seconds = assignment["delay_seconds"]
                    was_delayed = assignment["was_delayed"]
                    next_time = assignment["finish_time"]

                    """Changed up to here"""

                    # We schedule COMPLETION timestamp
                    self.schedule_event(
                        time=next_time,
                        case_id=event.case_id,
                        activity=next_act,
                        resource=r,
                        planned_start=planned_start,
                        actual_start=actual_start,
                        delay_seconds=delay_seconds,
                        was_delayed=was_delayed,
                    )

                    # produced a new live token
                    ctx["open_tokens"] += 1

                # K-Batch: flush batch if enough tasks have accumulated
                if self.batch_allocator is not None:
                    self._flush_batch()

                # SVFA: try to assign immediately after queueing successors
                if self.svfa_allocator is not None:
                    self._svfa_assign()

                # AND-join completion: only end when ALL tokens are done
                if ctx["open_tokens"] == 0:
                    # log END exactly once, at the time the last token completed
                    self.log_event(event.case_id, "END", event.time, resource=None)
                    ctx["ended"] = True

            # Force-flush any remaining batch tasks at end of simulation
            if self.batch_allocator is not None and self.batch_allocator.pending_tasks:
                self._flush_batch(force=True)
                run_again = True

            # Force-assign any remaining SVFA tasks at end of simulation
            if self.svfa_allocator is not None and self.svfa_allocator.pending_tasks:
                self._svfa_force_assign()
                run_again = True

    def _flush_batch(self, force=False):
        """Flush pending K-Batch tasks if the batch is ready or forced."""
        if self.batch_allocator is None:
            return
        if force:
            assignments = self.batch_allocator.flush_batch(force=True)
        elif self.batch_allocator.should_flush(self.now):
            assignments = self.batch_allocator.flush_batch()
        else:
            return
        for a in assignments:
            if a is None:
                continue
            self.schedule_event(
                time=a["finish_time"],
                case_id=a["case_id"],
                activity=a["activity"],
                resource=a["resource"],
                planned_start=a["planned_start"],
                actual_start=a["actual_start"],
                delay_seconds=a["delay_seconds"],
                was_delayed=a["was_delayed"],
            )

    def _svfa_assign(self):
        """Let the SVFA allocator score and assign pending tasks."""
        if self.svfa_allocator is None:
            return
        assignments = self.svfa_allocator.make_assignments(self.now)
        for a in assignments:
            self.schedule_event(
                time=a["finish_time"],
                case_id=a["case_id"],
                activity=a["activity"],
                resource=a["resource"],
                planned_start=a["planned_start"],
                actual_start=a["actual_start"],
                delay_seconds=a["delay_seconds"],
                was_delayed=a["was_delayed"],
            )

    def _svfa_force_assign(self):
        """Force-assign all remaining SVFA pending tasks at end of sim."""
        if self.svfa_allocator is None:
            return
        assignments = self.svfa_allocator.force_assign_remaining(self.now)
        for a in assignments:
            self.schedule_event(
                time=a["finish_time"],
                case_id=a["case_id"],
                activity=a["activity"],
                resource=a["resource"],
                planned_start=a["planned_start"],
                actual_start=a["actual_start"],
                delay_seconds=a["delay_seconds"],
                was_delayed=a["was_delayed"],
            )

    def export_csv(self, path: str):
        df = pd.DataFrame(self.log_rows)
        df = df.sort_values(["case:concept:name", "time:timestamp"])
        df.to_csv(path, index=False)
        print(f"💾 CSV exported to {path}")

    def export_xes(self, path: str):
        df = pd.DataFrame(self.log_rows)
        df = df.sort_values(["case:concept:name", "time:timestamp"])

        log = EventLog()
        for case_id, group in df.groupby("case:concept:name"):
            trace = Trace()
            trace.attributes["concept:name"] = case_id
            for _, row in group.iterrows():
                trace.append(Event({
                    "concept:name": row["concept:name"],
                    "time:timestamp": pd.to_datetime(row["time:timestamp"]),
                    "org:resource": row.get("org:resource", None),
                    # --- NEW (debug for Task 1.5) ---
                    "planned_start": row.get("planned_start", None),
                    "actual_start": row.get("actual_start", None),
                    "delay_seconds": row.get("delay_seconds", 0.0),
                    "was_delayed": row.get("was_delayed", False),
                }))
            log.append(trace)

        xes_exporter.apply(log, path)
        print(f"💾 XES exported to {path}")

    def get_prediction_stats(self) -> dict:
        """Get statistics about prediction usage (for debugging/evaluation)."""
        return {
            "total_cases": len(self.case_traces),
            "avg_trace_length": np.mean([len(t) for t in self.case_traces.values()]) if self.case_traces else 0,
            "predictor_mode": self.predictor.mode if self.predictor else "none"
        }


# ============================================================
# 1.2 INSTANCE SPAWN RATES (Emi's part)
# ============================================================
MAX_INTER_ARRIVAL = 24 * 60 * 60  # I approximated this to 1 day


def spawn_cases_static(engine_n: SimulationEngine,
                       n_cases: int,
                       start_activity: str,
                       start_time: datetime,
                       lambda_rate: float):
    """
    (1.2) Emi implements:
    """
    t = start_time
    for i in range(1, n_cases + 1):
        case_id = f"Case_{i}"
        """Change from here"""
        dur = duration_function(start_activity, t, {"event_count": 0, "ended": False})
        assignment = engine_n.allocate_resource(start_activity, t, dur)

        if assignment is not None:
            engine_n.schedule_event(
                assignment["finish_time"],
                case_id,
                start_activity,
                resource=assignment["resource"],
                planned_start=assignment["planned_start"],
                actual_start=assignment["actual_start"],
                delay_seconds=assignment["delay_seconds"],
                was_delayed=assignment["was_delayed"]
            )
        """Change up to here"""
        inter_arrival_time = np.random.exponential(scale=1 / lambda_rate)
        inter_arrival_time = min(inter_arrival_time, MAX_INTER_ARRIVAL)
        t = t + timedelta(seconds=inter_arrival_time)

def spawn_cases_dynamic(engine_n: SimulationEngine,
                        n_cases: int,
                        start_activity: str,
                        start_time: datetime,
                        arrival_model: DynamicArrivalModel):
    """
    Advanced version: context-dependent dynamic arrival rates.
    """
    t = start_time
    for i in range(1, n_cases + 1):
        case_id = f"Case_{i}"

        dur = duration_function(start_activity, t, {"event_count": 0, "ended": False})
        assignment = engine_n.allocate_resource(start_activity, t, dur)

        if assignment is not None:
            engine_n.schedule_event(
                assignment["finish_time"],
                case_id,
                start_activity,
                resource=assignment["resource"],
                planned_start=assignment["planned_start"],
                actual_start=assignment["actual_start"],
                delay_seconds=assignment["delay_seconds"],
                was_delayed=assignment["was_delayed"]
            )
        inter_arrival_time = arrival_model.sample_interarrival(t, MAX_INTER_ARRIVAL)
        t = t + timedelta(seconds=inter_arrival_time)

# ============================================================
# 1.3 PROCESSING TIMES (Lukas's part) - PLACEHOLDER
# ============================================================

def duration_function(activity: str, timestamp: datetime, case_context: dict) -> timedelta:
    # Combines labels like "X & Y", sum both durations
    parts = [p.strip() for p in activity.split("&")]
    total_seconds = sum(_sample_one(p) for p in parts)
    return timedelta(seconds=total_seconds)


def _sample_one(act_name: str) -> float:
    """Return sampled duration in seconds for a single activity name."""
    info = DIST_DATA.get(act_name)
    if not info:
        return 60.0  # fallback 60s if no distribution

    dist_name = info["best_dist"]
    params = info["params"]
    dist = DIST_MAP.get(dist_name)
    if dist is None:
        return 60.0

    sec = float(dist.rvs(*params))
    sec = max(0.1, min(sec, 60 * 60 * 24 * 60))  # cap at 60 days
    return sec


# ============================================================
# HELPER: Train predictor from historical data
# ============================================================

def train_predictor_from_csv(
        csv_path: str,
        mode: str = "basic",
        context_k: int = 2,
        use_token_replay: bool = False,
        bpmn_path: str = None,
        **kwargs
) -> ExpertActivityPredictor:
    """
    Train an ExpertActivityPredictor from a CSV event log.

    Args:
        csv_path: Path to CSV with case_id/case:concept:name, activity/concept:name, timestamp
        mode: 'basic' or 'advanced'
        context_k: Context window for k-gram
        use_token_replay: If True, use token-based replay for decision point
            identification (Task 1.4 Advanced). Requires bpmn_path.
        bpmn_path: Path to BPMN model file (used when use_token_replay=True)

    Returns:
        Trained predictor
    """
    df = pd.read_csv(csv_path)

    # Parse timestamp
    ts_col = None
    for col in ["timestamp", "time:timestamp"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], format="ISO8601", utc=True)

    predictor = ExpertActivityPredictor(
        mode=mode,
        basic_context_k=context_k,
        use_token_replay=use_token_replay,
        **kwargs
    )
    predictor.fit(df, bpmn_path=bpmn_path)

    return predictor


# 1.6 Resource permissions

def learn_resource_permissions(df: pd.DataFrame) -> dict:
    perms = {}
    for (act, res), _ in df.groupby(["concept:name", "org:resource"]):
        perms.setdefault(act, set()).add(str(res))
    return perms

def learn_advanced_resource_permissions(df: pd.DataFrame, n_roles: int = 10, threshold: float = 0.1) -> dict:
    
    role_permissions = learn_resource_permissions(df)
    
    freq_matrix = pd.crosstab(df['org:resource'], df['concept:name'])
    
    matrix_norm = freq_matrix.div(freq_matrix.sum(axis=1), axis=0).fillna(0)
    
    n_clusters = min(n_roles, len(matrix_norm))
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    roles = clusterer.fit_predict(matrix_norm)
    
    resource_to_role = {str(res): role for res, role in zip(matrix_norm.index, roles)}
    
    for role_id in range(n_clusters):
        role_resources = [res for res, r in resource_to_role.items() if r == role_id]
        if not role_resources:
            continue
        for activity in freq_matrix.columns:
            doers_in_role = sum(1 for res in role_resources if freq_matrix.loc[res, activity] > 0)
            if doers_in_role / len(role_resources) >= threshold:
                if activity not in role_permissions:
                    role_permissions[activity] = set()
                role_permissions[activity].update(role_resources)
                
    return role_permissions

def select_resource(activity: str, resource_pool: list, permissions: dict):
    allowed = permissions.get(activity, [])
    candidates = [r for r in resource_pool if r in allowed]
    return random.choice(candidates) if candidates else None


# ============================================================
# MAIN
# ============================================================
# (Emi): I am changing the method here, so I can choose the heuristic
def run_simulation(
        predictor: Optional[ExpertActivityPredictor] = None,
        n_cases: int = NUM_CASES,
        output_prefix: str = "sim",
        allocation_strategy: str = "random",
        USE_ADVANCED_PERMISSIONS: bool = USE_ADVANCED_PERMISSIONS,
        batch_size_k: int = 5,
        batch_max_wait_seconds: float = None,
        svfa_weights: Optional[list] = None,
        svfa_processing_stats: Optional[dict] = None,
        svfa_prob_fin_map: Optional[dict] = None,
        use_dynamic_arrivals: bool = False,
        arrival_bin_hours: int = 6,
):
    """
    Run the full simulation with optional predictor.

    Args:
        predictor: Trained ExpertActivityPredictor (Task 1.4), or None for random
        n_cases: Number of cases to simulate
        output_prefix: Prefix for output files

    Returns:
        SimulationEngine with results
    """
    print("=" * 60)
    print("  BUSINESS PROCESS SIMULATION")
    print("=" * 60)
    
    if USE_ADVANCED_PERMISSIONS:
        print("[Config] Using ADVANCED permissions with clustering")
        output_prefix += "_advanced_roles"
    else:
        print("[Config] Using BASIC permissions")
        output_prefix += "_basic_roles"

    if predictor:
        print(f"[Config] Using Task 1.4 predictor (mode={predictor.mode})")
    else:
        print("[Config] Using RANDOM branching (no predictor)")
    print(f"[Config] Number of cases: {n_cases}")
    # (Emi) Adding a new output for allocation strategy
    print(f"[Config] Allocation strategy: {allocation_strategy}")
    print()

    # availability = TwoWeekAvailabilityModel.fit_from_csv("bpi2017.csv")
    availability = RollingStochasticAvailabilityModel.fit_from_csv("bpi2017.csv")

    # optional:
    availability.window_days = 28  # 4 weeks rolling
    availability.min_points = 5
    availability.condition_on_weekday = False

    # resource pool from log (simple)
    df = pd.read_csv("bpi2017.csv", usecols=["concept:name", "org:resource"]).dropna()
    resource_pool = df["org:resource"].astype(str).unique().tolist()
    
    # 
    if USE_ADVANCED_PERMISSIONS:
        permissions = learn_advanced_resource_permissions(df)
    else:
        permissions = learn_resource_permissions(df)
     

    
    
    

    # Create engine with predictor
    # (Emi): changing this as well to add the allocation strategy in the end
    # For SVFA: compute processing stats and ProbFin from data if not provided
    if allocation_strategy == "svfa":
        if svfa_processing_stats is None:
            svfa_processing_stats = SVFAllocator.compute_stats_from_distributions(
                "distributions.json"
            )
        if svfa_prob_fin_map is None:
            svfa_prob_fin_map = SVFAllocator.compute_prob_fin_from_model(PROCESS_MODEL)

    engine = SimulationEngine(
        PROCESS_MODEL,
        SIM_START_TIME,
        gateways=GATEWAYS,
        predictor=predictor,  # TASK 1.4
        availability_model=availability,
        resource_pool=resource_pool,
        permissions=permissions,  # 1.6
        allocation_strategy=allocation_strategy,
        batch_size_k=batch_size_k,
        batch_max_wait_seconds=batch_max_wait_seconds,
        svfa_weights=svfa_weights,
        svfa_processing_stats=svfa_processing_stats,
        svfa_prob_fin_map=svfa_prob_fin_map,
    )

    # 1.2: Spawn cases
    print("[1.2] Spawning case arrivals...")
    if use_dynamic_arrivals:
        print(f"[1.2] Using DYNAMIC arrival model with {arrival_bin_hours}-hour bins")
        arrival_model = fit_dynamic_arrival_model("bpi2017.csv", bin_hours=arrival_bin_hours)
        spawn_cases_dynamic(engine, n_cases, START_ACTIVITY, SIM_START_TIME, arrival_model)
    else:
        print("[1.2] Using STATIC exponential arrival model")
        spawn_cases_static(engine, n_cases, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)

    # 1.1 + 1.4: Run simulation (uses predictor in route_next)
    print("[1.1] Running discrete event simulation...")
    engine.run(duration_function)

    #print("[1.6] Permission fallback count:", getattr(engine, "permission_fallback_count", 0))

    # K-Batch stats
    if engine.batch_allocator is not None:
        bstats = engine.batch_allocator.get_stats()
        print(f"[K-Batch] Batches solved: {bstats['batches_solved']}, "
              f"Tasks assigned: {bstats['tasks_assigned']}, "
              f"Forced flushes: {bstats['forced_flushes']}")

    # SVFA stats
    if engine.svfa_allocator is not None:
        sstats = engine.svfa_allocator.get_stats()
        print(f"[SVFA] Assignments: {sstats['assignments_made']}, "
              f"Postponements: {sstats['postponements']}, "
              f"Decision steps: {sstats['decision_steps']}")

    # Export results
    print("[Export] Saving logs...")

    full_output_path = os.path.join(output_dir, output_prefix)
    
    engine.export_csv(f"{full_output_path}.csv")
    engine.export_xes(f"{full_output_path}.xes")

    # Print stats
    stats = engine.get_prediction_stats()
    print()
    print(f"[Stats] Total cases simulated: {stats['total_cases']}")
    print(f"[Stats] Average trace length: {stats['avg_trace_length']:.1f}")
    print(f"[Stats] Predictor mode: {stats['predictor_mode']}")

    return engine


if __name__ == "__main__":
    # Example 1: Run without predictor (random branching)
    # print("\n>>> Running simulation WITHOUT predictor (random)...")
    # engine_random = run_simulation(predictor=None, n_cases=NUM_CASES, output_prefix="sim_random")

    # print("\n" + "=" * 60)

    # Example 2: If you have training data, train predictor first
    # Uncomment the following to train from your event log:

    print("\n>>> Training predictor from historical data...")
    predictor = train_predictor_from_csv(
        "bpi2017.csv",  # Your historical event log
        mode="basic",
        context_k=2
    )

    #print("\n>>> Running simulation WITH predictor (learned probabilities)...")
    #engine_pred = run_simulation(predictor=predictor, n_cases=NUM_CASES, output_prefix="sim_predicted")
    #print("\n✅ Simulation complete!")
    print("\n>>> Running simulation WITH predictor and RANDOM allocation...")
    engine_random = run_simulation(
        predictor=predictor,
        n_cases=NUM_CASES,
        output_prefix="sim_predicted_random",
        allocation_strategy="random"
    )

    print("\n>>> Running simulation WITH predictor and ROUND ROBIN allocation...")
    engine_rr = run_simulation(
        predictor=predictor,
        n_cases=NUM_CASES,
        output_prefix="sim_predicted_round_robin",
        allocation_strategy="round_robin"
    )

    print("\n>>> Running simulation WITH predictor and EARLIEST AVAILABLE allocation...")
    engine_ea = run_simulation(
        predictor=predictor,
        n_cases=NUM_CASES,
        output_prefix="sim_predicted_earliest_available",
        allocation_strategy="earliest_available"
    )

    print("\n>>> Running simulation WITH predictor and DYNAMIC ARRIVALS...")
    engine_dynamic_arrivals = run_simulation(
        predictor=predictor,
        n_cases=NUM_CASES,
        output_prefix="sim_predicted_dynamic_arrivals",
        allocation_strategy="random",
        use_dynamic_arrivals=True,
        arrival_bin_hours=6,
    )

    # Example 4: SVFA allocation (Task 2.1 Advanced - Learning-based)
    print("\n>>> Running simulation WITH predictor and SVFA allocation...")
    engine_svfa = run_simulation(
        predictor=predictor,
        n_cases=NUM_CASES,
        output_prefix="sim_predicted_svfa",
        allocation_strategy="svfa",
    )

    # Example 5: K-Batch allocation (Task 2.1 - Batch Allocation)
    print("\n>>> Running simulation WITH predictor and K-BATCH allocation (K=5)...")
    engine_batch = run_simulation(
        predictor=predictor,
        n_cases=NUM_CASES,
        output_prefix="sim_predicted_k_batch",
        allocation_strategy="k_batch",
        batch_size_k=5,
        batch_max_wait_seconds=3600.0,
    )

    print("\n✅ Simulation complete!")

    # Example 3: Train with token replay (Task 1.4 Advanced)
    # Identifies decision points via conformance-checking token replay on Petri net.
    # Uncomment to use:

    # import os
    # BPMN_PATH = "bpianko9.0.bpmn"
    # if os.path.exists(BPMN_PATH):
    #     print("\n>>> Training predictor WITH token replay (Advanced)...")
    #     predictor_replay = train_predictor_from_csv(
    #         "bpi2017.csv",
    #         mode="basic",
    #         context_k=2,
    #         use_token_replay=True,
    #         bpmn_path=BPMN_PATH,
    #     )
    #     print("\n>>> Running simulation WITH token replay predictor...")
    #     engine_replay = run_simulation(
    #         predictor=predictor_replay,
    #         n_cases=NUM_CASES,
    #         output_prefix="sim_token_replay",
    #     )
    #     print("\n✅ Token replay simulation complete!")
    # else:
    #     print(f"\n⚠️  BPMN file not found: {BPMN_PATH}")

# if __name__ == "__main__":
#     engine = SimulationEngine(PROCESS_MODEL, SIM_START_TIME, gateways=GATEWAYS)
#
#     # 1.2: create initial events (case arrivals)
#     spawn_cases(engine, NUM_CASES, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)
#
#     # 1.1: run core DES loop (uses 1.3 duration function)
#     engine.run(duration_function)
#
#     # export
#     engine.export_csv("sim.csv")
#     engine.export_xes("sim.xes")
