"""
Optimize K-Batch and SVFA parameters, then re-run the scaling experiment.

Strategy:
  1. Quick parameter sweep for K-Batch (batch_size, max_wait, strategy)
  2. Bayesian optimization for SVFA weights (small n_cases for speed)
  3. Re-run full scaling experiment with best parameters
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_engine_core_final_version import (
    run_simulation,
    train_predictor_from_csv,
)

TRAINING_LOG = "bpi2017.csv"
PREDICTOR_MODE = "basic"
CONTEXT_K = 2

OUTPUT_DIR = Path("sim_outputs") / "scaling_experiment_outputs_v2"
LOGS_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"


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


def run_sim_and_measure(predictor, strategy, n_cases, prefix, **extra_kwargs):
    np.random.seed(42)
    random.seed(42)
    engine_prefix = Path("scaling_experiment_outputs_v2") / "logs" / prefix
    out_prefix = LOGS_DIR / prefix

    kwargs = {
        "predictor": predictor,
        "n_cases": n_cases,
        "output_prefix": str(engine_prefix),
        "allocation_strategy": strategy,
    }
    kwargs.update(extra_kwargs)
    run_simulation(**kwargs)
    csv_path = f"{out_prefix}_advanced_roles.csv"
    return compute_metrics(csv_path)


# =====================================================================
# PHASE 1: K-Batch parameter sweep (100 cases — fast)
# =====================================================================

def sweep_kbatch(predictor):
    print("\n" + "=" * 70)
    print("PHASE 1: K-BATCH PARAMETER SWEEP (100 cases)")
    print("=" * 70)

    configs = [
        {"batch_size_k": 3, "batch_max_wait_seconds": 300.0},
        {"batch_size_k": 3, "batch_max_wait_seconds": 600.0},
        {"batch_size_k": 2, "batch_max_wait_seconds": 300.0},
        {"batch_size_k": 2, "batch_max_wait_seconds": 120.0},
        {"batch_size_k": 4, "batch_max_wait_seconds": 300.0},
        {"batch_size_k": 5, "batch_max_wait_seconds": 300.0},
        {"batch_size_k": 3, "batch_max_wait_seconds": 180.0},
    ]

    results = []
    for i, cfg in enumerate(configs):
        k = cfg["batch_size_k"]
        w = cfg["batch_max_wait_seconds"]
        print(f"\n  [{i+1}/{len(configs)}] K={k}, max_wait={w}s ...")
        m = run_sim_and_measure(
            predictor, "k_batch", 100,
            f"sweep_kb_{k}_{int(w)}",
            batch_size_k=k,
            batch_max_wait_seconds=w,
        )
        m.update(cfg)
        results.append(m)
        print(f"    -> cycle_time={m['avg_cycle_time_days']:.2f}d, "
              f"delay={m['avg_activity_delay_hours']:.2f}h, "
              f"fairness={m['resource_fairness_jain']:.3f}")

    best = min(results, key=lambda r: r["avg_cycle_time_days"])
    print(f"\n  BEST K-Batch: K={best['batch_size_k']}, "
          f"max_wait={best['batch_max_wait_seconds']}s "
          f"-> {best['avg_cycle_time_days']:.2f}d")
    return best


# =====================================================================
# PHASE 2: SVFA weight optimization (Bayesian, 100 cases)
# =====================================================================

def optimize_svfa(predictor):
    print("\n" + "=" * 70)
    print("PHASE 2: SVFA WEIGHT OPTIMIZATION (Bayesian, 100 cases)")
    print("=" * 70)

    from skopt import gp_minimize
    from skopt.space import Real

    call_count = [0]

    def objective(weights_list):
        call_count[0] += 1
        weights = list(weights_list)
        try:
            np.random.seed(None)
            random.seed(None)
            m = run_sim_and_measure(
                predictor, "svfa", 100,
                f"svfa_opt_{call_count[0]}",
                svfa_weights=weights,
            )
            ct = m["avg_cycle_time_days"]
            print(f"  [{call_count[0]}] w={[f'{w:.1f}' for w in weights]} "
                  f"-> CT={ct:.2f}d")
            return ct
        except Exception as e:
            print(f"  [{call_count[0]}] FAILED: {e}")
            return 200.0

    space = [
        Real(0.1, 50.0, name="w1_mean"),
        Real(0.0, 20.0, name="w2_var"),
        Real(0.0, 20.0, name="w3_act_rank"),
        Real(0.0, 20.0, name="w4_res_rank"),
        Real(0.0, 50.0, name="w5_prob_fin"),
        Real(0.0, 20.0, name="w6_queue_len"),
        Real(50.0, 100000.0, name="w7_threshold", prior="log-uniform"),
    ]

    t0 = time.time()
    result = gp_minimize(
        objective,
        space,
        acq_func="EI",
        n_calls=30,
        n_random_starts=12,
        noise=0.5 ** 2,
        random_state=42,
    )
    elapsed = time.time() - t0

    best_weights = list(result.x)
    best_ct = float(result.fun)
    print(f"\n  Optimization done in {elapsed:.0f}s")
    print(f"  Best weights: {[f'{w:.4f}' for w in best_weights]}")
    print(f"  Best cycle time: {best_ct:.2f}d")

    with open("svfa_weights_optimized.json", "w") as f:
        json.dump({
            "weights": best_weights,
            "best_cycle_time_days": best_ct,
            "n_calls": 30,
            "n_cases_per_call": 100,
        }, f, indent=2)

    return best_weights


# =====================================================================
# PHASE 3: Full scaling experiment with optimized parameters
# =====================================================================

def run_scaling_experiment(predictor, best_kb_params, best_svfa_weights):
    print("\n" + "=" * 70)
    print("PHASE 3: FULL SCALING EXPERIMENT (optimized parameters)")
    print("=" * 70)

    CASE_COUNTS = [100, 500, 1000]
    STRATEGIES = ["random", "round_robin", "earliest_available", "k_batch", "svfa"]

    results = []
    for strategy in STRATEGIES:
        for n_cases in CASE_COUNTS:
            print(f"\n>>> strategy={strategy}, cases={n_cases}")
            np.random.seed(42)
            random.seed(42)

            extra = {}
            if strategy == "k_batch":
                extra["batch_size_k"] = best_kb_params["batch_size_k"]
                extra["batch_max_wait_seconds"] = best_kb_params["batch_max_wait_seconds"]
            elif strategy == "svfa":
                extra["svfa_weights"] = best_svfa_weights

            m = run_sim_and_measure(
                predictor, strategy, n_cases,
                f"final_{strategy}_{n_cases}",
                **extra,
            )
            m["strategy"] = strategy
            m["cases"] = n_cases
            results.append(m)
            print(f"    -> CT={m['avg_cycle_time_days']:.2f}d, "
                  f"delay={m['avg_activity_delay_hours']:.2f}h, "
                  f"fairness={m['resource_fairness_jain']:.3f}")

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(OUTPUT_DIR / "scaling_experiment_v2_results.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(raw_df[["strategy", "cases", "avg_cycle_time_days",
                   "avg_activity_delay_hours", "resource_fairness_jain"]].to_string(index=False))

    return raw_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training predictor...")
    predictor = train_predictor_from_csv(
        TRAINING_LOG, mode=PREDICTOR_MODE, context_k=CONTEXT_K,
    )

    best_kb = sweep_kbatch(predictor)
    best_svfa_weights = optimize_svfa(predictor)

    print(f"\n{'='*70}")
    print("OPTIMIZED PARAMETERS:")
    print(f"  K-Batch: K={best_kb['batch_size_k']}, "
          f"max_wait={best_kb['batch_max_wait_seconds']}s")
    print(f"  SVFA weights: {[f'{w:.2f}' for w in best_svfa_weights]}")
    print(f"{'='*70}")

    run_scaling_experiment(predictor, best_kb, best_svfa_weights)


if __name__ == "__main__":
    main()
