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
# Import Task 1.4 module

from task_1_4_next_activity import ExpertActivityPredictor, train_predictor_from_log
from task_1_5_resource_availability import TwoWeekAvailabilityModel

# ============================================================
# CONFIGURATION (minimal)
# ============================================================

SIM_START_TIME = datetime(2016, 1, 1, 9, 0, 0)
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
        # "O_Create Offer & O_Created"   # Saifulla, add this line when branching implemented
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
# GATEWAY SEMANTICS (random for now)
# ============================================================
GATEWAYS = {
    # XOR-splits (choose exactly 1 outgoing)
    "A_Create Application": "xor",
    "A_Submitted": "xor",
    "O_Returned": "xor",
    "A_Incomplete": "xor",
    "A_Validating": "xor",

    # Inclusive OR-split (choose random non-empty subset)
    "W_Call after offers & A_Complete": "or",
    # "O_Sent (mail and online)": "or",    Saifulla, add this line when branching implemented
}


# ============================================================
# 1.1 SIMULATION ENGINE CORE
# ============================================================

class SimEvent:
    """Single discrete event, ordered by timestamp for priority queue."""

    def __init__(self, time: datetime, case_id: str, activity: str, resource: Optional[str] = None):
        self.time = time
        self.case_id = case_id
        self.activity = activity
        self.resource = resource

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

    def __init__(self, process_model: dict, start_time: datetime, gateways: Optional[dict] = None,  predictor: Optional[ExpertActivityPredictor] = None,
                 availability_model : Optional[TwoWeekAvailabilityModel] = None,  resource_pool : Optional[list] = None):
        self.model = process_model
        self.gateways = gateways or {}
        self.now = start_time
        self.event_queue = []
        self.log_rows = []
        self.case_context = {}
        self.availability_model = availability_model
        self.resource_pool = resource_pool or []
        # TASK 1.4: Activity predictor for XOR branching
        self.predictor = predictor
        self.case_traces: Dict[str, List[str]] = {}  # Track trace per case# per-case context

    def schedule_event(self, time: datetime, case_id: str, activity: str, resource: Optional[str] = None):
        heapq.heappush(self.event_queue, SimEvent(time, case_id, activity, resource))

    def log_event(self, case_id: str, activity: str, timestamp: datetime, resource: Optional[str]):
        self.log_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": timestamp.isoformat(),
            "org:resource": resource
        })

        # TASK 1.4: Track executed activities for prediction
        if case_id not in self.case_traces:
            self.case_traces[case_id] = []
        self.case_traces[case_id].append(activity)

    # SAIFULLA YOU MAKE YOUR MAGIC HERE
    def route_next(self, activity: str, case_id : str) -> List[str]:
        """
        Decide which outgoing activities to schedule based on gateway semantics.

        TASK 1.4 INTEGRATION:
        For XOR gateways, uses ExpertActivityPredictor to sample next activity
        based on learned probabilities, instead of random.choice().

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

        # --- SPECIAL CASE: OR-split + exclusive cancellation option ---
        if activity == "W_Call after offers & A_Complete":
            cancel_act = "A_Cancelled & O_Cancelled"
            or_candidates = [a for a in outgoing if a != cancel_act]

            # decide cancel probability
            if self.predictor:
                dist = self.predictor.get_next_activity_distribution(trace, outgoing)
                cancel_prob = dist.get(cancel_act, 0.2)
            else:
                cancel_prob = 0.2

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


    def run(self, duration_function):
        """
        duration_function(activity, current_time, case_context) -> timedelta
        """
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.now = event.time

            # get / init per-case context
            ctx = self.case_context.setdefault(event.case_id, {"event_count": 0, "ended": False})

            # if case already ended, ignore everything (prevents leftover queued events)
            if ctx["ended"]:
                continue

            # log execution
            self.log_event(event.case_id, event.activity, event.time, event.resource)

            # END semantics
            if event.activity == "END":
                ctx["ended"] = True
                continue

            ctx["event_count"] += 1

            # limit events per case (anti-infinite loop)
            if ctx["event_count"] > MAX_EVENTS_PER_CASE:
                ctx["ended"] = True
                continue

            # route to next activities using gateway semantics
            # Route to next activities (TASK 1.4 applied here for XOR)
            next_activities = self.route_next(event.activity, event.case_id)

            for next_act in next_activities:
                dur = duration_function(next_act, event.time, ctx)

                # 1.7 placeholder (random resource) but needed for 1.5
                if self.resource_pool:
                    r = random.choice(self.resource_pool)
                else:
                    r = None

                # 1.5 resource availability: delay start until resource is on shift
                start_time = event.time
                if r is not None and self.availability_model is not None:
                    start_time = self.availability_model.next_available(r, start_time)

                self.schedule_event(start_time + dur, event.case_id, next_act, resource=r)

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
                    "org:resource": row.get("org:resource", None)
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
# 1.2 INSTANCE SPAWN RATES (Emi's part) - PLACEHOLDER
# ============================================================
MAX_INTER_ARRIVAL = 24 * 60 * 60  # I approximated this to 1 day


def spawn_cases(engine_n: SimulationEngine,
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
        # RANDOM CHOICE FOR NOW
        r = random.choice(engine_n.resource_pool) if engine_n.resource_pool else None
        if r and engine_n.availability_model:
            t0 = engine_n.availability_model.next_available(r, t)
        else:
            t0 = t
        engine_n.schedule_event(t0, case_id, start_activity, resource=r)

        inter_arrival_time = np.random.exponential(scale=1 / lambda_rate)
        inter_arrival_time = min(inter_arrival_time, MAX_INTER_ARRIVAL)
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
        **kwargs
) -> ExpertActivityPredictor:
    """
    Train an ExpertActivityPredictor from a CSV event log.

    Args:
        csv_path: Path to CSV with case_id/case:concept:name, activity/concept:name, timestamp
        mode: 'basic' or 'advanced'
        context_k: Context window for k-gram

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
        df[ts_col] = pd.to_datetime(df[ts_col])

    predictor = ExpertActivityPredictor(
        mode=mode,
        basic_context_k=context_k,
        **kwargs
    )
    predictor.fit(df)

    return predictor


# ============================================================
# MAIN
# ============================================================
def run_simulation(
        predictor: Optional[ExpertActivityPredictor] = None,
        n_cases: int = NUM_CASES,
        output_prefix: str = "sim"
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

    if predictor:
        print(f"[Config] Using Task 1.4 predictor (mode={predictor.mode})")
    else:
        print("[Config] Using RANDOM branching (no predictor)")
    print(f"[Config] Number of cases: {n_cases}")
    print()

    availability = TwoWeekAvailabilityModel.fit_from_csv("bpi2017.csv")

    # resource pool from log (simple)
    df = pd.read_csv("bpi2017.csv", usecols=["org:resource"]).dropna()
    resource_pool = df["org:resource"].astype(str).unique().tolist()

    # Create engine with predictor
    engine = SimulationEngine(
        PROCESS_MODEL,
        SIM_START_TIME,
        gateways=GATEWAYS,
        predictor=predictor,  # TASK 1.4
        availability_model=availability,
        resource_pool=resource_pool,
    )

    # 1.2: Spawn cases
    print("[1.2] Spawning case arrivals...")
    spawn_cases(engine, n_cases, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)

    # 1.1 + 1.4: Run simulation (uses predictor in route_next)
    print("[1.1] Running discrete event simulation...")
    engine.run(duration_function)

    # Export results
    print("[Export] Saving logs...")
    engine.export_csv(f"{output_prefix}.csv")
    engine.export_xes(f"{output_prefix}.xes")

    # Print stats
    stats = engine.get_prediction_stats()
    print()
    print(f"[Stats] Total cases simulated: {stats['total_cases']}")
    print(f"[Stats] Average trace length: {stats['avg_trace_length']:.1f}")
    print(f"[Stats] Predictor mode: {stats['predictor_mode']}")

    return engine


if __name__ == "__main__":
    # Example 1: Run without predictor (random branching)
    #print("\n>>> Running simulation WITHOUT predictor (random)...")
    #engine_random = run_simulation(predictor=None, n_cases=NUM_CASES, output_prefix="sim_random")

    #print("\n" + "=" * 60)

     # Example 2: If you have training data, train predictor first
     # Uncomment the following to train from your event log:

    print("\n>>> Training predictor from historical data...")
    predictor = train_predictor_from_csv(
        "bpi2017filtered.csv",  # Your historical event log
        mode="basic",
         context_k=2
    )

    print("\n>>> Running simulation WITH predictor (learned probabilities)...")
    engine_pred = run_simulation(predictor=predictor, n_cases=50, output_prefix="sim_predicted")

    print("\n✅ Simulation complete!")



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
