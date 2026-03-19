import pandas as pd
import numpy as np
from pathlib import Path

FILES = [
    "sim_outputs/sim_predicted_random_advanced_roles.csv",
    "sim_outputs/sim_predicted_round_robin_advanced_roles.csv",
    "sim_outputs/sim_predicted_earliest_available_advanced_roles.csv",
    "sim_outputs/sim_predicted_k_batch_advanced_roles.csv",
    "sim_outputs/sim_predicted_svfa_advanced_roles.csv",

    "sim_outputs/sim_predicted_random_basic_roles.csv",
    "sim_outputs/sim_predicted_round_robin_basic_roles.csv",
    "sim_outputs/sim_predicted_earliest_available_basic_roles.csv",
    "sim_outputs/sim_predicted_k_batch_basic_roles.csv",
    "sim_outputs/sim_predicted_svfa_basic_roles.csv",
]


def compute_metrics(path: str):
    df = pd.read_csv(path)

    for col in ["time:timestamp", "planned_start", "actual_start"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")

    # 1) Average case duration
    case_times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
    cycle_times_sec = (case_times["max"] - case_times["min"]).dt.total_seconds()
    avg_cycle_time_sec = cycle_times_sec.mean()

    # 2) Average activity delay
    act = df[df["concept:name"] != "END"].copy()
    act["delay_seconds"] = pd.to_numeric(act["delay_seconds"], errors="coerce").fillna(0.0)
    avg_delay_sec = act["delay_seconds"].mean()

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
        avg_resource_occupation = occupation.mean()

        occ = occupation.values
        den = len(occ) * np.sum(occ ** 2)
        fairness_jain = ((occ.sum() ** 2) / den) if den > 0 else np.nan

        delayed_share_resources = (resource_act["delay_seconds"] > 0).mean()
    else:
        avg_resource_occupation = np.nan
        fairness_jain = np.nan
        delayed_share_resources = np.nan

    return {
        "strategy": Path(path).stem.replace("sim_predicted_", ""),
        "avg_cycle_time_days": avg_cycle_time_sec / 86400,
        "avg_activity_delay_hours": avg_delay_sec / 3600,
        "avg_resource_occupation_pct": avg_resource_occupation * 100 if pd.notna(avg_resource_occupation) else np.nan,
        "resource_fairness_jain": fairness_jain,
        "delayed_share_pct": delayed_share_resources * 100 if pd.notna(delayed_share_resources) else np.nan,
    }


existing_files = []
for f in FILES:
    if Path(f).exists():
        existing_files.append(f)
    else:
        print(f"Missing file: {f}")

results = pd.DataFrame([compute_metrics(f) for f in existing_files])
print(results.sort_values("strategy").to_string(index=False))