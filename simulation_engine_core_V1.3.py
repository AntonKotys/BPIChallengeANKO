"""
Simulation Engine Core V1.6 - Improved Predictor Accuracy

FIXES from V1.5:
1. Reduced Laplace smoothing to avoid inflating rare/unseen transitions
2. Fixed process model to ensure validation loop is reachable
3. Better handling of 100%/0% splits in historical data
"""

import heapq
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
import json
import numpy as np
import random
from datetime import timedelta
from scipy.stats import lognorm, gamma, weibull_min

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from task_1_4_next_activity import ExpertActivityPredictor
from task_1_5_rolling_stochastic_availability import RollingStochasticAvailabilityModel

# ============================================================
# CONFIGURATION
# ============================================================

SIM_START_TIME = datetime(2016, 2, 1, 10, 0, 0)
START_ACTIVITY = "A_Create Application"
NUM_CASES = 100
MAX_EVENTS_PER_CASE = 200
LAMBDA_RATE = 1 / 1089.18

with open("distributions.json", "r") as f:
    DIST_DATA = json.load(f)

DIST_MAP = {
    "lognorm": lognorm,
    "gamma": gamma,
    "weibull_min": weibull_min
}

# ============================================================
# ACTIVITY MAPPING: Historical → Simulation
# ============================================================

ACTIVITY_MAPPING = {
    # Direct matches
    "A_Create Application": "A_Create Application",
    "A_Submitted": "A_Submitted",
    "W_Handle leads": "W_Handle leads",
    "O_Returned": "O_Returned",
    "O_Sent (mail and online)": "O_Sent (mail and online)",
    "O_Sent (online only)": "O_Sent (mail and online)",
    "A_Accepted": "A_Accepted",
    "A_Incomplete": "A_Incomplete",
    "A_Validating": "A_Validating",

    # Combined activities
    "W_Complete application": "W_Complete application & A_Concept",
    "A_Concept": "W_Complete application & A_Concept",
    "O_Create Offer": "O_Create Offer & O_Created",
    "O_Created": "O_Create Offer & O_Created",
    "W_Call after offers": "W_Call after offers & A_Complete",
    "A_Complete": "W_Call after offers & A_Complete",
    "W_Validate application": "W_Validate application & A_Validating",
    "A_Cancelled": "A_Cancelled & O_Cancelled",
    "O_Cancelled": "A_Cancelled & O_Cancelled",

    # End states
    "A_Denied": "END",
    "O_Refused": "END",
    "O_Accepted": "END",

    # Incomplete handling -> goes to validation
    "W_Call incomplete files": "W_Validate application & A_Validating",

    # Other
    "A_Pending": "A_Pending",
    "W_Assess potential fraud": "W_Assess potential fraud",
    "W_Personal Loan collection": "W_Personal Loan collection",
    "W_Shortened completion ": "W_Shortened completion",
}


def map_activity(activity: str) -> str:
    return ACTIVITY_MAPPING.get(activity, activity)


def map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'activity' in df.columns:
        df['activity'] = df['activity'].apply(map_activity)
    if 'concept:name' in df.columns:
        df['concept:name'] = df['concept:name'].apply(map_activity)
    return df


# ============================================================
# PROCESS MODEL - Fixed to match actual flow
# ============================================================

PROCESS_MODEL = {
    "A_Create Application": [
        "A_Submitted",
        "W_Complete application & A_Concept"
    ],
    "A_Submitted": [
        "W_Handle leads",
        "W_Complete application & A_Concept"  # Rare but possible
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
    # ???????????
    # VALIDATION LOOP - This is where most cases go through multiple times
    "W_Validate application & A_Validating": [
        "O_Returned",  # After validation, decision is made
        "A_Incomplete",  # Can also go to incomplete
        "A_Validating",  # Internal validation step
    ],
    "O_Returned": [
        "W_Validate application & A_Validating",  # 97.7% - revalidation
        "END"  # 2.2% - case ends (accepted/denied)
    ],
    "A_Incomplete": [
        "W_Validate application & A_Validating",  # 96.6% - back to validation
        "END"  # Rare
    ],
    "A_Validating": [
        "O_Returned",  # 53.2% - goes to returned
        "W_Validate application & A_Validating",  # 46.2% - continues validation
        "END"  # 0.6%
    ],
    # ???????????/
    "A_Cancelled & O_Cancelled": [
        "END"
    ]
}

GATEWAYS = {
    "A_Create Application": "xor",
    "A_Submitted": "xor",
    "O_Returned": "xor",
    "A_Incomplete": "xor",
    "A_Validating": "xor",
    "W_Validate application & A_Validating": "xor",  # Added!
    "W_Call after offers & A_Complete": "or",
    "O_Sent (mail and online)": "or",
}


# ============================================================
# SIMULATION ENGINE
# ============================================================

class SimEvent:
    def __init__(self, time: datetime, case_id: str, activity: str,
                 resource: Optional[str] = None,
                 planned_start: Optional[datetime] = None,
                 actual_start: Optional[datetime] = None,
                 delay_seconds: float = 0.0,
                 was_delayed: bool = False):
        self.time = time
        self.case_id = case_id
        self.activity = activity
        self.resource = resource
        self.planned_start = planned_start
        self.actual_start = actual_start
        self.delay_seconds = delay_seconds
        self.was_delayed = was_delayed

    def __lt__(self, other):
        return self.time < other.time


class SimulationEngine:
    def __init__(self, process_model: dict, start_time: datetime, gateways: Optional[dict] = None,
                 predictor: Optional[ExpertActivityPredictor] = None,
                 availability_model: Optional[RollingStochasticAvailabilityModel] = None,
                 resource_pool: Optional[list] = None, permissions: Optional[dict] = None):
        self.model = process_model
        self.gateways = gateways or {}
        self.now = start_time
        self.event_queue = []
        self.log_rows = []
        self.case_context = {}
        self.availability_model = availability_model
        self.resource_pool = resource_pool or []
        self.permissions = permissions or {}
        self.predictor = predictor
        self.case_traces: Dict[str, List[str]] = {}
        self.permission_fallback_count = 0

    def schedule_event(self, time: datetime, case_id: str, activity: str,
                       resource: Optional[str] = None,
                       planned_start: Optional[datetime] = None,
                       actual_start: Optional[datetime] = None,
                       delay_seconds: float = 0.0,
                       was_delayed: bool = False):
        heapq.heappush(self.event_queue, SimEvent(
            time, case_id, activity, resource,
            planned_start, actual_start, delay_seconds, was_delayed
        ))

    def log_event(self, case_id: str, activity: str, timestamp: datetime, resource: Optional[str],
                  planned_start: Optional[datetime] = None,
                  actual_start: Optional[datetime] = None,
                  delay_seconds: float = 0.0,
                  was_delayed: bool = False):
        self.log_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": timestamp.isoformat(),
            "org:resource": resource,
            "planned_start": planned_start.isoformat() if planned_start else None,
            "actual_start": actual_start.isoformat() if actual_start else None,
            "delay_seconds": delay_seconds,
            "was_delayed": was_delayed,
        })
        if case_id not in self.case_traces:
            self.case_traces[case_id] = []
        self.case_traces[case_id].append(activity)

    def route_next(self, activity: str, case_id: str) -> List[str]:
        outgoing = self.model.get(activity, [])
        if not outgoing:
            return []

        gateway_type = self.gateways.get(activity)
        trace = self.case_traces.get(case_id, [])

        # Special case: OR-split with cancellation
        if activity == "W_Call after offers & A_Complete":
            cancel_act = "A_Cancelled & O_Cancelled"
            or_candidates = [a for a in outgoing if a != cancel_act]

            if self.predictor:
                dist = self.predictor.get_next_activity_distribution(
                    trace, enabled_next=outgoing, current_activity=activity
                )
                cancel_prob = dist.get(cancel_act, 0.2)
            else:
                cancel_prob = 0.2

            if cancel_act in outgoing and random.random() < cancel_prob:
                return [cancel_act]

            if not or_candidates:
                return []

            if self.predictor:
                sampled = self.predictor.sample_next_activity(
                    prefix_activities=trace,
                    enabled_next=or_candidates,
                    current_activity=activity
                )
                if sampled and sampled in or_candidates:
                    return [sampled]

            return [random.choice(or_candidates)]

        # XOR GATEWAY
        if gateway_type == "xor":
            if self.predictor:
                sampled = self.predictor.sample_next_activity(
                    prefix_activities=trace,
                    enabled_next=outgoing,
                    current_activity=activity
                )
                if sampled and sampled in outgoing:
                    return [sampled]
            return [random.choice(outgoing)]

        # OR GATEWAY - pick one ????????
        if gateway_type == "or":
            if self.predictor:
                sampled = self.predictor.sample_next_activity(
                    prefix_activities=trace,
                    enabled_next=outgoing,
                    current_activity=activity
                )
                if sampled and sampled in outgoing:
                    return [sampled]
            return [random.choice(outgoing)]

        return outgoing

    def run(self, duration_function):
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.now = event.time

            ctx = self.case_context.setdefault(event.case_id, {"event_count": 0, "ended": False})

            if ctx["ended"]:
                continue

            self.log_event(
                event.case_id, event.activity, event.time, event.resource,
                planned_start=getattr(event, "planned_start", None),
                actual_start=getattr(event, "actual_start", None),
                delay_seconds=getattr(event, "delay_seconds", 0.0),
                was_delayed=getattr(event, "was_delayed", False),
            )

            if event.activity == "END":
                ctx["ended"] = True
                continue

            ctx["event_count"] += 1

            if ctx["event_count"] > MAX_EVENTS_PER_CASE:
                ctx["ended"] = True
                continue

            next_activities = self.route_next(event.activity, event.case_id)

            for next_act in next_activities:
                dur = duration_function(next_act, event.time, ctx)

                r = None
                if self.availability_model is not None:
                    eligible = self.availability_model.eligible_resources(event.time)
                    if eligible:
                        if self.permissions:
                            parts = [p.strip() for p in next_act.split("&")]
                            allowed_sets = [set(self.permissions.get(p, [])) for p in parts]
                            allowed = set.intersection(*allowed_sets) if allowed_sets else set()
                            candidates = [res for res in eligible if res in allowed]
                            if candidates:
                                r = random.choice(candidates)
                            else:
                                self.permission_fallback_count += 1
                                r = random.choice(eligible)
                        else:
                            r = random.choice(eligible)
                    elif self.resource_pool:
                        r = select_resource(next_act, self.resource_pool, self.permissions) if self.permissions else random.choice(self.resource_pool)
                elif self.resource_pool:
                    r = select_resource(next_act, self.resource_pool, self.permissions) if self.permissions else random.choice(self.resource_pool)

                planned_start = event.time
                actual_start = planned_start
                delay_seconds = 0.0
                was_delayed = False

                if r is not None and self.availability_model is not None:
                    actual_start = self.availability_model.next_available(r, planned_start)
                    delay_seconds = (actual_start - planned_start).total_seconds()
                    was_delayed = delay_seconds > 0.0

                self.schedule_event(
                    actual_start + dur, event.case_id, next_act,
                    resource=r, planned_start=planned_start, actual_start=actual_start,
                    delay_seconds=delay_seconds, was_delayed=was_delayed
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
                }))
            log.append(trace)

        xes_exporter.apply(log, path)
        print(f"💾 XES exported to {path}")

    def get_prediction_stats(self) -> dict:
        return {
            "total_cases": len(self.case_traces),
            "avg_trace_length": np.mean([len(t) for t in self.case_traces.values()]) if self.case_traces else 0,
            "predictor_mode": self.predictor.mode if self.predictor else "none"
        }


# ============================================================
# SPAWN CASES
# ============================================================
MAX_INTER_ARRIVAL = 24 * 60 * 60


def spawn_cases(engine_n: SimulationEngine, n_cases: int, start_activity: str,
                start_time: datetime, lambda_rate: float):
    t = start_time
    for i in range(1, n_cases + 1):
        case_id = f"Case_{i}"

        r = None
        if engine_n.availability_model is not None:
            eligible = engine_n.availability_model.eligible_resources(t)
            if eligible:
                if engine_n.permissions:
                    allowed = engine_n.permissions.get(start_activity, [])
                    candidates = [res for res in eligible if res in allowed]
                    if candidates:
                        r = random.choice(candidates)
                    else:
                        engine_n.permission_fallback_count += 1
                        r = random.choice(eligible)
                else:
                    r = random.choice(eligible)
            elif engine_n.resource_pool:
                r = select_resource(start_activity, engine_n.resource_pool, engine_n.permissions) if engine_n.permissions else random.choice(engine_n.resource_pool)
        elif engine_n.resource_pool:
            r = select_resource(start_activity, engine_n.resource_pool, engine_n.permissions) if engine_n.permissions else random.choice(engine_n.resource_pool)

        planned_start = t
        actual_start = planned_start
        delay_seconds = 0.0
        was_delayed = False

        if r and engine_n.availability_model:
            actual_start = engine_n.availability_model.next_available(r, planned_start)
            delay_seconds = (actual_start - planned_start).total_seconds()
            was_delayed = delay_seconds > 0.0

        dur = duration_function(start_activity, actual_start, {"event_count": 0, "ended": False})

        engine_n.schedule_event(
            actual_start + dur, case_id, start_activity,
            resource=r, planned_start=planned_start, actual_start=actual_start,
            delay_seconds=delay_seconds, was_delayed=was_delayed
        )

        inter_arrival_time = np.random.exponential(scale=1 / lambda_rate)
        inter_arrival_time = min(inter_arrival_time, MAX_INTER_ARRIVAL)
        t = t + timedelta(seconds=inter_arrival_time)


# ============================================================
# DURATION FUNCTION
# ============================================================

def duration_function(activity: str, timestamp: datetime, case_context: dict) -> timedelta:
    parts = [p.strip() for p in activity.split("&")]
    total_seconds = sum(_sample_one(p) for p in parts)
    return timedelta(seconds=total_seconds)


def _sample_one(act_name: str) -> float:
    info = DIST_DATA.get(act_name)
    if not info:
        return 60.0
    dist_name = info["best_dist"]
    params = info["params"]
    dist = DIST_MAP.get(dist_name)
    if dist is None:
        return 60.0
    sec = float(dist.rvs(*params))
    sec = max(0.1, min(sec, 60 * 60 * 24 * 60))
    return sec


# ============================================================
# TRAIN PREDICTOR - WITH LOWER SMOOTHING
# ============================================================

def train_predictor_from_csv(
        csv_path: str,
        process_model: Dict[str, List[str]] = None,
        gateways: Dict[str, str] = None,
        mode: str = "basic",
        context_k: int = 2,
        smoothing_alpha: float = 0.01,  # REDUCED from 1.0 to 0.01
        **kwargs
) -> ExpertActivityPredictor:
    """Train predictor with lower smoothing to better match historical data."""
    df = pd.read_csv(csv_path)

    ts_col = None
    for col in ["timestamp", "time:timestamp"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], format="ISO8601", utc=True)

    # Map activity names
    print("[Task 1.4] Mapping historical activity names to simulation names...")
    df = map_dataframe(df)

    if 'concept:name' in df.columns:
        unique_activities = df['concept:name'].unique()
    else:
        unique_activities = df['activity'].unique() if 'activity' in df.columns else []
    print(f"[Task 1.4] Unique activities after mapping: {len(unique_activities)}")

    # Create predictor with LOWER smoothing
    predictor = ExpertActivityPredictor(
        mode=mode,
        basic_context_k=context_k,
        process_model=process_model,
        gateways=gateways,
        smoothing_alpha=smoothing_alpha,  # Lower = less inflation of unseen transitions
        **kwargs
    )

    predictor.fit(df, process_model=process_model, gateways=gateways)

    print(f"[Task 1.4] Trained predictor (smoothing_alpha={smoothing_alpha})")
    print(f"[Task 1.4] Activities with learned transitions: {len(predictor.transition_probs)}")

    return predictor


# ============================================================
# RESOURCE HELPERS
# ============================================================

def learn_resource_permissions(df: pd.DataFrame) -> dict:
    df = map_dataframe(df)
    perms = {}
    act_col = 'concept:name' if 'concept:name' in df.columns else 'activity'
    res_col = 'org:resource' if 'org:resource' in df.columns else 'resource'

    if act_col in df.columns and res_col in df.columns:
        for (act, res), _ in df.groupby([act_col, res_col]):
            perms.setdefault(act, set()).add(str(res))
    return perms


def select_resource(activity: str, resource_pool: list, permissions: dict):
    allowed = permissions.get(activity, [])
    candidates = [r for r in resource_pool if r in allowed]
    return random.choice(candidates) if candidates else None


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(
        predictor: Optional[ExpertActivityPredictor] = None,
        n_cases: int = NUM_CASES,
        output_prefix: str = "sim",
        historical_csv: str = "bpi2017.csv"
):
    print("=" * 60)
    print("  BUSINESS PROCESS SIMULATION")
    print("=" * 60)

    if predictor:
        print(f"[Config] Using Task 1.4 predictor (mode={predictor.mode})")
    else:
        print("[Config] Using RANDOM branching (no predictor)")
    print(f"[Config] Number of cases: {n_cases}")
    print()

    availability = RollingStochasticAvailabilityModel.fit_from_csv(historical_csv)
    availability.window_days = 28
    availability.min_points = 5
    availability.condition_on_weekday = False

    df = pd.read_csv(historical_csv, usecols=["concept:name", "org:resource"]).dropna()
    df = map_dataframe(df)
    resource_pool = df["org:resource"].astype(str).unique().tolist()
    permissions = learn_resource_permissions(df)

    engine = SimulationEngine(
        PROCESS_MODEL,
        SIM_START_TIME,
        gateways=GATEWAYS,
        predictor=predictor,
        availability_model=availability,
        resource_pool=resource_pool,
        permissions=permissions,
    )

    print("[1.2] Spawning case arrivals...")
    spawn_cases(engine, n_cases, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)

    print("[1.1] Running discrete event simulation...")
    engine.run(duration_function)

    print(f"[1.6] Permission fallback count: {engine.permission_fallback_count}")

    print("[Export] Saving logs...")
    engine.export_csv(f"{output_prefix}.csv")
    engine.export_xes(f"{output_prefix}.xes")

    stats = engine.get_prediction_stats()
    print()
    print(f"[Stats] Total cases simulated: {stats['total_cases']}")
    print(f"[Stats] Average trace length: {stats['avg_trace_length']:.1f}")
    print(f"[Stats] Predictor mode: {stats['predictor_mode']}")

    return engine


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import os

    HISTORICAL_CSV = "bpi2017.csv"
    N_CASES = 100

    if not os.path.exists(HISTORICAL_CSV):
        print(f"❌ Historical data file not found: {HISTORICAL_CSV}")
        exit(1)

    print("\n" + "█" * 70)
    print("█  TASK 1.4 SIMULATION BENCHMARK (V1.6 - Improved Accuracy)")
    print("█" * 70)

    # 1. RANDOM
    print("\n" + "=" * 70)
    print(">>> [1/2] Running simulation WITHOUT predictor (RANDOM branching)...")
    print("=" * 70)

    engine_random = run_simulation(
        predictor=None,
        n_cases=N_CASES,
        output_prefix="sim_random",
        historical_csv=HISTORICAL_CSV
    )

    # 2. TRAINED with lower smoothing
    print("\n" + "=" * 70)
    print(">>> [2/2] Training predictor (LOW smoothing) and running simulation...")
    print("=" * 70)

    print("\n>>> Training predictor with reduced Laplace smoothing...")
    predictor = train_predictor_from_csv(
        HISTORICAL_CSV,
        process_model=PROCESS_MODEL,
        gateways=GATEWAYS,
        mode="basic",
        context_k=2,
        smoothing_alpha=0.01  # Much lower than default 1.0
    )

    print("\n>>> Running simulation WITH predictor...")
    engine_trained = run_simulation(
        predictor=predictor,
        n_cases=N_CASES,
        output_prefix="sim_predicted",
        historical_csv=HISTORICAL_CSV
    )

    # 3. Metrics
    print("\n" + "=" * 70)
    print(">>> Running performance metrics comparison...")
    print("=" * 70)

    try:
        from task_1_4_metrics import compare_simulations
        compare_simulations("sim_random.csv", "sim_predicted.csv", HISTORICAL_CSV)
    except ImportError:
        print("\n⚠️  task_1_4_metrics.py not found.")
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "█" * 70)
    print("█  SIMULATION COMPLETE!")
    print("█" * 70)