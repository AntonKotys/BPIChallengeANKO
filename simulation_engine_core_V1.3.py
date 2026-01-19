"""
Simulation Engine Core V2.0 - Complete Pipeline

This simulation engine integrates:
- Task 1.1: Discrete Event Simulation Core
- Task 1.2: Instance Spawn Rates (Poisson arrivals)
- Task 1.3: Processing Times (fitted distributions)
- Task 1.4: Next Activity Prediction (k-gram for XOR gateways)
- Task 1.5: Rolling Stochastic Availability (optional)
- Task 1.6: Resource Permissions

FEATURES:
- Runs BOTH trained predictor and random simulations
- Properly handles OR gateways with parallel branches
- Generates comparison report against historical data
- Runs verification tests

USAGE:
    python simulation_engine_core_V2.py [n_cases]

    Example: python simulation_engine_core_V2.py 100
"""

import heapq
from datetime import datetime
from typing import Optional, List, Dict, Set
import os
import sys

import pandas as pd
import json
import numpy as np
import random
from datetime import timedelta
from scipy.stats import lognorm, gamma, weibull_min

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

# Import Task 1.4 module
from task_1_4_next_activity import ExpertActivityPredictor

# Try to import Task 1.5 (may not be available)
try:
    from task_1_5_rolling_stochastic_availability import RollingStochasticAvailabilityModel
    HAS_AVAILABILITY = True
except ImportError:
    RollingStochasticAvailabilityModel = None
    HAS_AVAILABILITY = False

# ============================================================
# CONFIGURATION
# ============================================================

SIM_START_TIME = datetime(2016, 2, 1, 10, 0, 0)
START_ACTIVITY = "A_Create Application"
NUM_CASES = 100
MAX_EVENTS_PER_CASE = 200
LAMBDA_RATE = 1 / 1089.18

# Load distribution data (with fallback)
try:
    with open("distributions.json", "r") as f:
        DIST_DATA = json.load(f)
except FileNotFoundError:
    DIST_DATA = {}

DIST_MAP = {
    "lognorm": lognorm,
    "gamma": gamma,
    "weibull_min": weibull_min
}

# ============================================================
# PROCESS MODEL (ORIGINAL - defines valid transitions)
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
    ],

    # OR gateway - can spawn multiple parallel branches
    "W_Call after offers & A_Complete": [
        "W_Validate application & A_Validating",
        "O_Create Offer & O_Created",
        "A_Cancelled & O_Cancelled"
    ],

    # --- Validation Loop ---
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

    # Inclusive OR-split (choose random non-empty subset - PARALLEL)
    "W_Call after offers & A_Complete": "or",
}


# ============================================================
# SIMULATION EVENT
# ============================================================

class SimEvent:
    """Single discrete event, ordered by timestamp for priority queue."""

    def __init__(
            self,
            time: datetime,
            case_id: str,
            activity: str,
            branch_id: Optional[int] = None,
            resource: Optional[str] = None,
    ):
        self.time = time
        self.case_id = case_id
        self.activity = activity
        self.branch_id = branch_id
        self.resource = resource

    def __lt__(self, other):
        return self.time < other.time


# ============================================================
# SIMULATION ENGINE
# ============================================================

class SimulationEngine:
    """
    Discrete Event Simulation Engine with Task 1.4 Integration.

    TASK 1.4 KEY FEATURE:
    - For XOR gateways, uses ExpertActivityPredictor to sample next activity
    - Predictor uses TRACE HISTORY for k-gram context
    - Predictor ONLY returns valid outgoing activities

    OR GATEWAY HANDLING:
    - OR gateways can spawn multiple parallel branches
    - Each branch executes independently
    """

    def __init__(
        self,
        process_model: dict,
        start_time: datetime,
        gateways: Optional[dict] = None,
        predictor: Optional[ExpertActivityPredictor] = None,
        resource_pool: Optional[list] = None,
        permissions: Optional[dict] = None
    ):
        self.model = process_model
        self.gateways = gateways or {}
        self.now = start_time
        self.event_queue = []
        self.log_rows = []
        self.case_context = {}
        self.resource_pool = resource_pool or []
        self.permissions = permissions or {}

        # TASK 1.4: Activity predictor for XOR branching
        self.predictor = predictor

        # Track executed activities per case (TRACE HISTORY)
        self.case_traces: Dict[str, List[str]] = {}

        # Track parallel branches per case
        self.next_branch_id: Dict[str, int] = {}

    def schedule_event(self, time: datetime, case_id: str, activity: str,
                       branch_id: Optional[int] = None, resource: Optional[str] = None):
        heapq.heappush(self.event_queue, SimEvent(time, case_id, activity, branch_id, resource))

    def log_event(self, case_id: str, activity: str, timestamp: datetime,
                  branch_id: Optional[int], resource: Optional[str]):
        """Log an executed event and update trace history."""
        self.log_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": timestamp.isoformat(),
            "branch_id": branch_id,
            "org:resource": resource,
        })

        # Update trace history for predictor
        if case_id not in self.case_traces:
            self.case_traces[case_id] = []
        self.case_traces[case_id].append(activity)

    def _get_next_branch_id(self, case_id: str) -> int:
        """Get next available branch ID for a case."""
        if case_id not in self.next_branch_id:
            self.next_branch_id[case_id] = 0
        bid = self.next_branch_id[case_id]
        self.next_branch_id[case_id] += 1
        return bid

    def route_next(self, activity: str, case_id: str, current_branch_id: Optional[int] = None) -> List[tuple]:
        """
        Decide which outgoing activities to schedule based on gateway semantics.

        For XOR: Uses predictor (trained) or random (baseline)
        For OR: Spawns parallel branches

        Returns: List of (activity, branch_id) tuples
        """
        outgoing = self.model.get(activity, [])
        if not outgoing:
            return []

        gateway_type = self.gateways.get(activity)
        trace = self.case_traces.get(case_id, [])

        # --- SPECIAL CASE: OR-split with exclusive cancellation ---
        if activity == "W_Call after offers & A_Complete":
            cancel_act = "A_Cancelled & O_Cancelled"
            or_candidates = [a for a in outgoing if a != cancel_act]

            # Determine cancel probability
            if self.predictor:
                dist = self.predictor.get_next_activity_distribution(trace, outgoing)
                cancel_prob = dist.get(cancel_act, 0.2)
            else:
                cancel_prob = 0.2

            # Cancel is exclusive
            if cancel_act in outgoing and random.random() < cancel_prob:
                return [(cancel_act, current_branch_id)]

            # Otherwise, OR split among remaining branches
            if not or_candidates:
                return []

            k = random.randint(1, len(or_candidates))
            selected = random.sample(or_candidates, k)

            # Each branch gets its own ID for parallel tracking
            result = []
            for act in selected:
                branch_id = self._get_next_branch_id(case_id) if len(selected) > 1 else current_branch_id
                result.append((act, branch_id))
            return result

        # --- XOR GATEWAY: TASK 1.4 ---
        if gateway_type == "xor":
            if self.predictor:
                sampled = self.predictor.sample_next_activity(
                    prefix_activities=trace,
                    enabled_next=outgoing
                )
                if sampled and sampled in outgoing:
                    return [(sampled, current_branch_id)]

            # Fallback to random
            return [(random.choice(outgoing), current_branch_id)]

        # --- OR GATEWAY: Random non-empty subset (parallel) ---
        if gateway_type == "or":
            k = random.randint(1, len(outgoing))
            selected = random.sample(outgoing, k)
            result = []
            for act in selected:
                branch_id = self._get_next_branch_id(case_id) if len(selected) > 1 else current_branch_id
                result.append((act, branch_id))
            return result

        # --- DEFAULT: Schedule all outgoing ---
        return [(act, current_branch_id) for act in outgoing]

    def _select_resource(self, activity: str) -> Optional[str]:
        """Select a resource for the activity."""
        allowed = self.permissions.get(activity, set())
        candidates = [r for r in self.resource_pool if r in allowed]
        if candidates:
            return random.choice(candidates)
        if allowed:
            return random.choice(list(allowed))
        return None

    def run(self, duration_function):
        """Run the discrete event simulation."""
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.now = event.time

            ctx = self.case_context.setdefault(event.case_id, {"event_count": 0, "ended": False})

            if ctx["ended"]:
                continue

            self.log_event(event.case_id, event.activity, event.time, event.branch_id, event.resource)

            if event.activity == "END":
                ctx["ended"] = True
                continue

            ctx["event_count"] += 1
            if ctx["event_count"] > MAX_EVENTS_PER_CASE:
                ctx["ended"] = True
                continue

            next_items = self.route_next(event.activity, event.case_id, event.branch_id)

            for next_act, branch_id in next_items:
                dur = duration_function(next_act, event.time, ctx)
                r = self._select_resource(next_act)
                self.schedule_event(event.time + dur, event.case_id, next_act, branch_id, r)

    def export_csv(self, path: str):
        df = pd.DataFrame(self.log_rows)
        df = df.sort_values(["case:concept:name", "time:timestamp"])
        df.to_csv(path, index=False)
        print(f"  💾 CSV exported: {path}")

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
                }))
            log.append(trace)

        xes_exporter.apply(log, path)
        print(f"  💾 XES exported: {path}")

    def get_stats(self) -> dict:
        return {
            "total_cases": len(self.case_traces),
            "avg_trace_length": np.mean([len(t) for t in self.case_traces.values()]) if self.case_traces else 0,
            "predictor_mode": self.predictor.mode if self.predictor else "random"
        }


# ============================================================
# CASE SPAWNING (Task 1.2)
# ============================================================

MAX_INTER_ARRIVAL = 24 * 60 * 60  # 1 day max


def spawn_cases(engine: SimulationEngine, n_cases: int, start_activity: str,
                start_time: datetime, lambda_rate: float):
    """Spawn cases according to exponential inter-arrival times."""
    t = start_time

    for i in range(1, n_cases + 1):
        case_id = f"Case_{i}"
        branch_id = engine._get_next_branch_id(case_id)
        r = engine._select_resource(start_activity)
        dur = duration_function(start_activity, t, {"event_count": 0, "ended": False})

        engine.schedule_event(t + dur, case_id, start_activity, branch_id, r)

        inter_arrival = min(np.random.exponential(scale=1 / lambda_rate), MAX_INTER_ARRIVAL)
        t = t + timedelta(seconds=inter_arrival)


# ============================================================
# PROCESSING TIMES (Task 1.3)
# ============================================================

def duration_function(activity: str, timestamp: datetime, case_context: dict) -> timedelta:
    """Sample activity duration from fitted distributions."""
    parts = [p.strip() for p in activity.split("&")]
    total_seconds = sum(_sample_one(p) for p in parts)
    return timedelta(seconds=total_seconds)


def _sample_one(act_name: str) -> float:
    """Sample duration in seconds for a single activity."""
    info = DIST_DATA.get(act_name)
    if not info:
        return 60.0

    dist_name = info["best_dist"]
    params = info["params"]
    dist = DIST_MAP.get(dist_name)
    if dist is None:
        return 60.0

    sec = float(dist.rvs(*params))
    return max(0.1, min(sec, 60 * 60 * 24 * 60))


# ============================================================
# RESOURCE PERMISSIONS (Task 1.6)
# ============================================================

def learn_resource_permissions(df: pd.DataFrame) -> dict:
    """Learn which resources can perform which activities."""
    perms = {}
    for (act, res), _ in df.groupby(["concept:name", "org:resource"]):
        perms.setdefault(act, set()).add(str(res))
    return perms


# ============================================================
# PREDICTOR TRAINING
# ============================================================

def train_predictor_from_csv(csv_path: str, mode: str = "basic", context_k: int = 2) -> ExpertActivityPredictor:
    """Train an ExpertActivityPredictor from a CSV event log."""
    df = pd.read_csv(csv_path)

    ts_col = None
    for col in ["timestamp", "time:timestamp"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], format="ISO8601", utc=True)

    predictor = ExpertActivityPredictor(mode=mode, basic_context_k=context_k)
    predictor.fit(df)

    return predictor


# ============================================================
# SIMULATION RUNNERS
# ============================================================

def run_trained_simulation(predictor: ExpertActivityPredictor, n_cases: int,
                          resource_pool: list, permissions: dict) -> SimulationEngine:
    """Run simulation with trained predictor."""
    print("\n  Running simulation WITH trained predictor...")

    engine = SimulationEngine(
        PROCESS_MODEL, SIM_START_TIME, gateways=GATEWAYS,
        predictor=predictor, resource_pool=resource_pool, permissions=permissions
    )

    spawn_cases(engine, n_cases, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)
    engine.run(duration_function)
    engine.export_csv("sim_predicted.csv")
    engine.export_xes("sim_predicted.xes")

    stats = engine.get_stats()
    print(f"  ✓ {stats['total_cases']} cases, avg {stats['avg_trace_length']:.1f} events/case")

    return engine


def run_random_simulation(n_cases: int, resource_pool: list, permissions: dict) -> SimulationEngine:
    """Run simulation with random branching."""
    print("\n  Running simulation with RANDOM branching...")

    engine = SimulationEngine(
        PROCESS_MODEL, SIM_START_TIME, gateways=GATEWAYS,
        predictor=None, resource_pool=resource_pool, permissions=permissions
    )

    spawn_cases(engine, n_cases, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)
    engine.run(duration_function)
    engine.export_csv("sim_random.csv")
    engine.export_xes("sim_random.xes")

    stats = engine.get_stats()
    print(f"  ✓ {stats['total_cases']} cases, avg {stats['avg_trace_length']:.1f} events/case")

    return engine


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_full_pipeline(training_data_path: str = "bpi2017.csv", n_cases: int = 100,
                     run_verification: bool = True, generate_report: bool = True):
    """
    Run the complete simulation pipeline:
    1. Train predictor from historical data
    2. Run simulation with trained predictor
    3. Run simulation with random branching
    4. Run verification tests
    5. Generate comparison report
    """
    print("\n" + "█" * 70)
    print("█  COMPLETE SIMULATION PIPELINE")
    print("█" * 70)

    # Load resources and permissions
    print("\n─" * 70)
    print("SETUP: Loading resources and permissions")
    print("─" * 70)

    resource_pool = []
    permissions = {}
    try:
        df = pd.read_csv(training_data_path, usecols=["concept:name", "org:resource"]).dropna()
        resource_pool = df["org:resource"].astype(str).unique().tolist()
        permissions = learn_resource_permissions(df)
        print(f"  ✓ Loaded {len(resource_pool)} resources")
    except Exception as e:
        print(f"  ⚠ Could not load resources: {e}")

    # STEP 1: Train predictor
    print("\n" + "─" * 70)
    print("STEP 1: Training predictor from historical data")
    print("─" * 70)

    predictor = train_predictor_from_csv(training_data_path, mode="basic", context_k=2)
    print(f"  ✓ Predictor trained (k-gram context, k=2)")
    print(f"  ✓ Activities learned: {len(predictor.activities_set)}")

    # STEP 2: Run trained simulation
    print("\n" + "─" * 70)
    print("STEP 2: Trained predictor simulation")
    print("─" * 70)

    engine_trained = run_trained_simulation(predictor, n_cases, resource_pool, permissions)

    # STEP 3: Run random simulation
    print("\n" + "─" * 70)
    print("STEP 3: Random branching simulation")
    print("─" * 70)

    engine_random = run_random_simulation(n_cases, resource_pool, permissions)

    # STEP 4: Verification
    if run_verification:
        print("\n" + "─" * 70)
        print("STEP 4: Verification tests")
        print("─" * 70)

        try:
            import verify_task_1_4
            verify_task_1_4.main()
        except ImportError:
            print("  ⚠ verify_task_1_4.py not found")
        except Exception as e:
            print(f"  ⚠ Verification error: {e}")

    # STEP 5: Generate report
    if generate_report:
        print("\n" + "─" * 70)
        print("STEP 5: Generating comparison report")
        print("─" * 70)

        try:
            from branch_comparison_report import generate_comparison_report, print_report

            report = generate_comparison_report(
                historical_path=training_data_path,
                predicted_path="sim_predicted.csv",
                random_path="sim_random.csv",
                output_path="gateway_comparison_report.txt"
            )
            print_report(report)
        except ImportError:
            print("  ⚠ branch_comparison_report.py not found")
        except Exception as e:
            print(f"  ⚠ Report error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "█" * 70)
    print("█  PIPELINE COMPLETE")
    print("█" * 70)
    print("\n  Output files:")
    print("    • sim_predicted.csv / .xes  - Trained predictor simulation")
    print("    • sim_random.csv / .xes     - Random branching simulation")
    if generate_report:
        print("    • gateway_comparison_report.txt - Comparison analysis")

    return engine_trained, engine_random


if __name__ == "__main__":
    n_cases = 100
    if len(sys.argv) > 1:
        try:
            n_cases = int(sys.argv[1])
        except:
            pass

    run_full_pipeline(
        training_data_path="bpi2017.csv",
        n_cases=n_cases,
        run_verification=True,
        generate_report=True
    )