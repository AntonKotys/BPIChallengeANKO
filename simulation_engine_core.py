import heapq
from datetime import datetime
from typing import Optional, List

import pandas as pd
import json
import numpy as np
import random
from datetime import timedelta
from scipy.stats import lognorm, gamma, weibull_min

from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

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

    def __init__(self, time: datetime, case_id: str, activity: str):
        self.time = time
        self.case_id = case_id
        self.activity = activity

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

    def __init__(self, process_model: dict, start_time: datetime, gateways: Optional[dict] = None):
        self.model = process_model
        self.gateways = gateways or {}
        self.now = start_time
        self.event_queue = []
        self.log_rows = []
        self.case_context = {}  # per-case context

    def schedule_event(self, time: datetime, case_id: str, activity: str):
        heapq.heappush(self.event_queue, SimEvent(time, case_id, activity))

    def log_event(self, case_id: str, activity: str, timestamp: datetime):
        self.log_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": timestamp.isoformat()
        })

    # SAIFULLA YOU MAKE YOUR MAGIC HERE
    def route_next(self, activity: str) -> List[str]:
        """
        Decide which outgoing activities to schedule based on gateway semantics.

        RANDOM IS IMPLEMENTED HERE: SAIFULLA YOU MAKE YOUR MAGIC HERE
          - XOR: random.choice(outgoing)
          - OR : random non-empty subset via random.sample(...)
        """
        outgoing = self.model.get(activity, [])
        if not outgoing:
            return []

        gw = self.gateways.get(activity)

        # --- SPECIAL CASE: OR-split + exclusive cancellation option ---
        if activity == "W_Call after offers & A_Complete":
            cancel_act = "A_Cancelled & O_Cancelled"
            or_candidates = [a for a in outgoing if a != cancel_act]

            # 1) sometimes cancel only (mutually exclusive)
            if cancel_act in outgoing and random.random() < 0.2:  # chose here randomly 0.2
                return [cancel_act]  # <-- RANDOM cancel decision HERE

            # 2) otherwise normal OR split among remaining branches
            if or_candidates:
                k = random.randint(1, len(or_candidates))
                return random.sample(or_candidates, k)  # <-- RANDOM OR subset HERE
            return []

        if gw == "xor":
            return [random.choice(outgoing)]  # <-- RANDOM XOR CHOICE HERE

        if gw == "or":
            k = random.randint(1, len(outgoing))
            return random.sample(outgoing, k)  # <-- RANDOM INCLUSIVE(OR) CHOICE HERE

        # default: schedule all (sequence-like)
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
            self.log_event(event.case_id, event.activity, event.time)

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
            next_activities = self.route_next(event.activity)

            for next_act in next_activities:
                dur = duration_function(next_act, event.time, ctx)
                self.schedule_event(event.time + dur, event.case_id, next_act)

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
                    "time:timestamp": pd.to_datetime(row["time:timestamp"])
                }))
            log.append(trace)

        xes_exporter.apply(log, path)
        print(f"💾 XES exported to {path}")


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

    Current placeholder: spawn one case every x seconds.
    """
    t = start_time
    for i in range(1, n_cases + 1):
        case_id = f"Case_{i}"
        engine_n.schedule_event(t, case_id, start_activity)
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
# MAIN
# ============================================================

if __name__ == "__main__":
    engine = SimulationEngine(PROCESS_MODEL, SIM_START_TIME, gateways=GATEWAYS)

    # 1.2: create initial events (case arrivals)
    spawn_cases(engine, NUM_CASES, START_ACTIVITY, SIM_START_TIME, LAMBDA_RATE)

    # 1.1: run core DES loop (uses 1.3 duration function)
    engine.run(duration_function)

    # export
    engine.export_csv("sim.csv")
    engine.export_xes("sim.xes")
