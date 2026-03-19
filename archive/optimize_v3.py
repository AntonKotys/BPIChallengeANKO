"""
Final optimization pass: aggressive tuning for 1000-case performance.

Key insight: at high load, coordination overhead dominates.
Strategy: minimize waiting/batching overhead while keeping the smart assignment.
  - K-Batch: K=2, max_wait=60s (minimal batching, still smarter than greedy)
  - SVFA: use optimized weights but with very high threshold (no postponement)
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_engine_core_final_version import (
    run_simulation,
    train_predictor_from_csv,
)

OUTPUT_DIR = Path("sim_outputs") / "scaling_experiment_outputs_v3"
LOGS_DIR = OUTPUT_DIR / "logs"


def compute_metrics(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    for col in ["time:timestamp", "planned_start", "actual_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    case_times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    cycle_times_sec = (case_times["max"] - case_times["min"]).dt.total_seconds()
    avg_cycle_time_days = cycle_times_sec.mean() / 86400.0

    act = df[df["concept:name"] != "END"].copy()
    act["delay_seconds"] = pd.to_numeric(act["delay_seconds"], errors="coerce").fillna(0.0)
    avg_activity_delay_hours = act["delay_seconds"].mean() / 3600.0

    resource_act = act.copy()
    if not resource_act.empty:
        resource_act["busy_seconds"] = (
            resource_act["time:timestamp"] - resource_act["actual_start"]
        ).dt.total_seconds()
        sim_start = resource_act["actual_start"].min()
        sim_end = resource_act["time:timestamp"].max()
        horizon_sec = (sim_end - sim_start).total_seconds()
        busy_per_resource = resource_act.groupby("org:resource")["busy_seconds"].sum()
        occupation = busy_per_resource / horizon_sec
        avg_resource_occupation_pct = occupation.mean() * 100.0
        occ = occupation.values
        den = len(occ) * np.sum(occ ** 2)
        resource_fairness_jain = ((occ.sum() ** 2) / den) if den > 0 else np.nan
        delayed_share_pct = (resource_act["delay_seconds"] > 0).mean() * 100.0
    else:
        avg_resource_occupation_pct = np.nan
        resource_fairness_jain = np.nan
        delayed_share_pct = np.nan

    return {
        "avg_cycle_time_days": avg_cycle_time_days,
        "avg_activity_delay_hours": avg_activity_delay_hours,
        "avg_resource_occupation_pct": avg_resource_occupation_pct,
        "resource_fairness_jain": resource_fairness_jain,
        "delayed_share_pct": delayed_share_pct,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training predictor...")
    predictor = train_predictor_from_csv("bpi2017.csv", mode="basic", context_k=2)

    with open("svfa_weights_optimized.json") as f:
        svfa_data = json.load(f)
    optimized_weights = svfa_data["weights"]

    CASE_COUNTS = [100, 500, 1000]

    configs = {
        "random": {},
        "round_robin": {},
        "earliest_available": {},
        "k_batch_k2_w60": {
            "allocation_strategy": "k_batch",
            "batch_size_k": 2,
            "batch_max_wait_seconds": 60.0,
        },
        "k_batch_k3_w120": {
            "allocation_strategy": "k_batch",
            "batch_size_k": 3,
            "batch_max_wait_seconds": 120.0,
        },
        "k_batch_k2_w30": {
            "allocation_strategy": "k_batch",
            "batch_size_k": 2,
            "batch_max_wait_seconds": 30.0,
        },
        "svfa_optimized": {
            "allocation_strategy": "svfa",
            "svfa_weights": optimized_weights,
        },
        "svfa_no_postpone": {
            "allocation_strategy": "svfa",
            "svfa_weights": optimized_weights[:6] + [1e12],
        },
    }

    results = []
    for name, cfg in configs.items():
        strategy = cfg.pop("allocation_strategy", name)
        for n_cases in CASE_COUNTS:
            np.random.seed(42)
            random.seed(42)

            print(f"\n>>> {name}, cases={n_cases}")
            engine_prefix = Path("scaling_experiment_outputs_v3") / "logs" / f"{name}_{n_cases}"
            out_prefix = LOGS_DIR / f"{name}_{n_cases}"

            kwargs = {
                "predictor": predictor,
                "n_cases": n_cases,
                "output_prefix": str(engine_prefix),
                "allocation_strategy": strategy,
            }
            kwargs.update(cfg)
            run_simulation(**kwargs)

            csv_path = f"{out_prefix}_advanced_roles.csv"
            m = compute_metrics(csv_path)
            m["strategy"] = name
            m["cases"] = n_cases
            results.append(m)
            print(f"    -> CT={m['avg_cycle_time_days']:.2f}d, "
                  f"delay={m['avg_activity_delay_hours']:.2f}h, "
                  f"fairness={m['resource_fairness_jain']:.3f}")

        cfg["allocation_strategy"] = strategy

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(OUTPUT_DIR / "results_v3.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS V3")
    print("=" * 70)

    for n_cases in CASE_COUNTS:
        print(f"\n--- {n_cases} cases ---")
        sub = raw_df[raw_df["cases"] == n_cases].sort_values("avg_cycle_time_days")
        print(sub[["strategy", "avg_cycle_time_days", "avg_activity_delay_hours",
                    "resource_fairness_jain", "delayed_share_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
