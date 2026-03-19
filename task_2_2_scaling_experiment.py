
from pathlib import Path

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation_engine_core_final_version import (
    run_simulation,
    train_predictor_from_csv,
)

CASE_COUNTS = [100, 500, 1000]

STRATEGIES = [
    "random",
    "round_robin",
    "earliest_available",
    "k_batch",
    "svfa",
]

# Use >1 if you want smoother curves
N_REPEATS = 1

OUTPUT_DIR = Path("sim_outputs") / "scaling_experiment_outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Fixed settings from current experiments
PREDICTOR_MODE = "basic"
CONTEXT_K = 2
BATCH_SIZE_K = 5
BATCH_MAX_WAIT_SECONDS = 3600.0

TRAINING_LOG = "bpi2017.csv"



# METRIC COMPUTATION

def compute_metrics(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    for col in ["time:timestamp", "planned_start", "actual_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # 1) Average cycle time
    case_times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    cycle_times_sec = (case_times["max"] - case_times["min"]).dt.total_seconds()
    avg_cycle_time_days = cycle_times_sec.mean() / 86400.0

    # 2) Average activity delay
    act = df[df["concept:name"] != "END"].copy()
    act["delay_seconds"] = pd.to_numeric(act["delay_seconds"], errors="coerce").fillna(0.0)
    avg_activity_delay_hours = act["delay_seconds"].mean() / 3600.0

    # 3) Resource-based metrics (ALL resources, no filtering)
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
        resource_fairness_jain = (
            ((occ.sum() ** 2) / den) if den > 0 else np.nan
        )

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


# EXPERIMENT

def run_one_experiment(predictor, strategy: str, n_cases: int, repeat_idx: int) -> dict:
    # reset RNG so each experiment starts from same seed
    np.random.seed(42)
    random.seed(42)
    # path where simulation engine should save logs (relative!)
    engine_prefix = Path("scaling_experiment_outputs") / "logs" / f"sim_{strategy}_{n_cases}_run{repeat_idx}"

    # actual location for reading results later
    out_prefix = LOGS_DIR / f"sim_{strategy}_{n_cases}_run{repeat_idx}"

    kwargs = {
        "predictor": predictor,
        "n_cases": n_cases,
        "output_prefix": str(engine_prefix),  # important
        "allocation_strategy": strategy,
    }

    # Strategy-specific settings
    if strategy == "k_batch":
        kwargs["batch_size_k"] = BATCH_SIZE_K
        kwargs["batch_max_wait_seconds"] = BATCH_MAX_WAIT_SECONDS

    run_simulation(**kwargs)

    csv_path = f"{out_prefix}_advanced_roles.csv"
    metrics = compute_metrics(csv_path)
    metrics["strategy"] = strategy
    metrics["cases"] = n_cases
    metrics["repeat"] = repeat_idx
    return metrics


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "avg_cycle_time_days",
        "avg_activity_delay_hours",
        "avg_resource_occupation_pct",
        "resource_fairness_jain",
        "delayed_share_pct",
    ]

    agg_df = (
        results_df.groupby(["strategy", "cases"], as_index=False)[metric_cols]
        .mean()
    )
    return agg_df

# PLOTTING

def plot_metric(df: pd.DataFrame, metric_col: str, y_label: str, output_file: Path):
    plt.figure(figsize=(9, 5))

    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy].sort_values("cases")
        plt.plot(sub["cases"], sub[metric_col], marker="o", label=strategy)

    plt.xlabel("Number of simulated cases")
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs number of simulated cases")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def save_all_plots(df: pd.DataFrame):
    plot_metric(
        df,
        "avg_cycle_time_days",
        "Average cycle time (days)",
        PLOTS_DIR / "avg_cycle_time_vs_cases.png",
    )

    plot_metric(
        df,
        "avg_activity_delay_hours",
        "Average activity delay (hours)",
        PLOTS_DIR / "avg_activity_delay_vs_cases.png",
    )

    plot_metric(
        df,
        "avg_resource_occupation_pct",
        "Average resource occupation (%)",
        PLOTS_DIR / "avg_resource_occupation_vs_cases.png",
    )

    plot_metric(
        df,
        "resource_fairness_jain",
        "Resource fairness (Jain index)",
        PLOTS_DIR / "resource_fairness_vs_cases.png",
    )

    plot_metric(
        df,
        "delayed_share_pct",
        "Share of delayed activities (%)",
        PLOTS_DIR / "delayed_share_vs_cases.png",
    )


# MAIN

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING PREDICTOR")
    print("=" * 70)

    predictor = train_predictor_from_csv(
        TRAINING_LOG,
        mode=PREDICTOR_MODE,
        context_k=CONTEXT_K,
    )

    results = []

    print("\n" + "=" * 70)
    print("RUNNING SCALING EXPERIMENT")
    print("=" * 70)

    for strategy in STRATEGIES:
        for n_cases in CASE_COUNTS:
            for repeat_idx in range(1, N_REPEATS + 1):
                print(f"\n>>> strategy={strategy}, cases={n_cases}, repeat={repeat_idx}")
                row = run_one_experiment(
                    predictor=predictor,
                    strategy=strategy,
                    n_cases=n_cases,
                    repeat_idx=repeat_idx,
                )
                results.append(row)

    raw_df = pd.DataFrame(results)
    raw_path = OUTPUT_DIR / "scaling_experiment_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)

    agg_df = aggregate_results(raw_df)
    agg_path = OUTPUT_DIR / "scaling_experiment_aggregated_results.csv"
    agg_df.to_csv(agg_path, index=False)

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    print(agg_df.sort_values(["strategy", "cases"]).to_string(index=False))

    save_all_plots(agg_df)

    print("\nDone.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Aggregated results saved to: {agg_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()