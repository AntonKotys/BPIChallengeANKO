import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

from simulation_engine_core_final_version_bruteforce_firing import (
    run_simulation,
    train_predictor_from_csv,
)

SYSTEM_RESOURCES = {"SYSTEM_W"}
N_CASES = 100
ALLOCATION_STRATEGY = "earliest_available"
SEED = 42

OUTPUT_DIR = Path("sim_outputs") / "firing_experiment"
LOGS_DIR = OUTPUT_DIR / "logs"

TRAINING_LOG = "bpi2017.csv"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)


def is_human_resource(resource) -> bool:
    if pd.isna(resource):
        return False
    return str(resource) not in SYSTEM_RESOURCES


def compute_avg_cycle_time_days(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="mixed", errors="coerce")

    case_times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    cycle_times_sec = (case_times["max"] - case_times["min"]).dt.total_seconds()
    return cycle_times_sec.mean() / 86400.0


def compute_busy_time_per_human_resource(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)

    for col in ["time:timestamp", "actual_start"]:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    act = df[df["concept:name"] != "END"].copy()
    human_act = act[act["org:resource"].apply(is_human_resource)].copy()

    if human_act.empty:
        return pd.Series(dtype=float)

    human_act["busy_seconds"] = (
        human_act["time:timestamp"] - human_act["actual_start"]
    ).dt.total_seconds()

    return human_act.groupby("org:resource")["busy_seconds"].sum().sort_values()


def run_and_get_csv(
    predictor,
    output_prefix: str,
    excluded_resources=None,
) -> str:
    set_seed(SEED)

    run_simulation(
        predictor=predictor,
        n_cases=N_CASES,
        output_prefix=output_prefix,
        allocation_strategy=ALLOCATION_STRATEGY,
        excluded_resources=excluded_resources,
    )

    base = Path("sim_outputs") / f"{output_prefix}"
    csv_default = f"{base}.csv"
    csv_adv = f"{base}_advanced_roles.csv"
    csv_basic = f"{base}_basic_roles.csv"

    if Path(csv_adv).exists():
        return csv_adv
    if Path(csv_basic).exists():
        return csv_basic
    if Path(csv_default).exists():
        return csv_default

    raise FileNotFoundError(
        f"Could not find output CSV for prefix {output_prefix}. "
        f"Tried: {csv_adv}, {csv_basic}, {csv_default}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING PREDICTOR")
    print("=" * 70)

    predictor = train_predictor_from_csv(
        TRAINING_LOG,
        mode="basic",
        context_k=2,
    )

    # 1) Baseline run
    print("\n" + "=" * 70)
    print("BASELINE RUN")
    print("=" * 70)

    baseline_prefix = "firing_experiment/logs/baseline_earliest_available_100"
    baseline_csv = run_and_get_csv(
        predictor=predictor,
        output_prefix=baseline_prefix,
        excluded_resources=None,
    )

    baseline_cycle = compute_avg_cycle_time_days(baseline_csv)
    busy_times = compute_busy_time_per_human_resource(baseline_csv)

    print(f"Baseline CSV: {baseline_csv}")
    print(f"Baseline average cycle time: {baseline_cycle:.4f} days")

    if busy_times.empty:
        print("No human resources found in baseline log.")
        return

    print("\nBusy time per human resource (ascending):")
    print(busy_times.to_string())

    heuristic_pair = tuple(busy_times.index[:2])
    print(f"\nHeuristic pair (two lowest busy-time employees): {heuristic_pair}")

    # -------------------------------------------------
    # 2) Brute-force over all pairs
    # -------------------------------------------------
    print("\n" + "=" * 70)
    print("BRUTE-FORCE SEARCH OVER ALL EMPLOYEE PAIRS")
    print("=" * 70)

    human_resources = list(busy_times.index)
    pairs = list(itertools.combinations(human_resources, 2))

    results = []

    for idx, pair in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] Testing pair {pair}")

        pair_name = f"{pair[0]}__{pair[1]}".replace("/", "_").replace(" ", "_")
        out_prefix = f"firing_experiment/logs/fire_{pair_name}"

        csv_path = run_and_get_csv(
            predictor=predictor,
            output_prefix=out_prefix,
            excluded_resources=set(pair),
        )

        # detect collapsed simulations
        df = pd.read_csv(csv_path)
        trace_lengths = df.groupby("case:concept:name").size()
        avg_trace_length = trace_lengths.mean()

        if avg_trace_length < 10:
            print(f"Skipping pair {pair} because process collapsed (avg trace length = {avg_trace_length:.2f})")
            continue

        # only compute metrics if simulation is valid
        cycle = compute_avg_cycle_time_days(csv_path)

        results.append({
            "resource_1": pair[0],
            "resource_2": pair[1],
            "avg_cycle_time_days": cycle,
            "delta_vs_baseline_days": cycle - baseline_cycle,
            "is_heuristic_pair": set(pair) == set(heuristic_pair),
        })

    results_df = pd.DataFrame(results).sort_values(
        ["avg_cycle_time_days", "resource_1", "resource_2"]
    )

    results_path = OUTPUT_DIR / "bruteforce_pair_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0]
    best_pair = (best_row["resource_1"], best_row["resource_2"])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best pair by brute force: {best_pair}")
    print(f"Best pair cycle time: {best_row['avg_cycle_time_days']:.4f} days")
    print(f"Change vs baseline: {best_row['delta_vs_baseline_days']:+.4f} days")

    heuristic_row = results_df[results_df["is_heuristic_pair"]].iloc[0]
    print(f"\nHeuristic pair: {heuristic_pair}")
    print(f"Heuristic pair cycle time: {heuristic_row['avg_cycle_time_days']:.4f} days")
    print(f"Change vs baseline: {heuristic_row['delta_vs_baseline_days']:+.4f} days")

    print("\nTop 10 best pairs:")
    print(results_df.head(10).to_string(index=False))

    if set(best_pair) == set(heuristic_pair):
        print("\nConclusion: The brute-force optimum matches the two lowest busy-time employees.")
    else:
        print("\nConclusion: The brute-force optimum does NOT exactly match the two lowest busy-time employees.")

    print(f"\nFull brute-force results saved to: {results_path}")


if __name__ == "__main__":
    main()