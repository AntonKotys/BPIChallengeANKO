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

SYSTEM_RESOURCES = {"SYSTEM_W"}

def is_human_resource(resource) -> bool:
    if pd.isna(resource):
        return False
    return str(resource) not in SYSTEM_RESOURCES


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
    # use all real activities except END
    act = df[df["concept:name"] != "END"].copy()
    act["delay_seconds"] = pd.to_numeric(act["delay_seconds"], errors="coerce").fillna(0.0)
    avg_delay_sec = act["delay_seconds"].mean()

    # 3) Human-resource-only metrics
    human_act = act[act["org:resource"].apply(is_human_resource)].copy()

    # busy time per activity = completion - actual_start
    human_act["busy_seconds"] = (
        human_act["time:timestamp"] - human_act["actual_start"]
    ).dt.total_seconds()

    # simulation horizon for humans
    if not human_act.empty:
        sim_start = human_act["actual_start"].min()
        sim_end = human_act["time:timestamp"].max()
        horizon_sec = (sim_end - sim_start).total_seconds()

        busy_per_resource = human_act.groupby("org:resource")["busy_seconds"].sum()
        occupation = busy_per_resource / horizon_sec
        avg_resource_occupation = occupation.mean()

        occ = occupation.values
        fairness_jain = (occ.sum() ** 2) / (len(occ) * np.sum(occ ** 2))
    else:
        avg_resource_occupation = np.nan
        fairness_jain = np.nan

    # optional: human-only delayed share
    if not human_act.empty:
        delayed_share_human = (human_act["delay_seconds"] > 0).mean()
    else:
        delayed_share_human = np.nan

    return {
        "strategy": Path(path).stem.replace("sim_predicted_", ""),
        "avg_cycle_time_days": avg_cycle_time_sec / 86400,
        "avg_activity_delay_hours": avg_delay_sec / 3600,
        "avg_human_resource_occupation_pct": avg_resource_occupation * 100 if pd.notna(avg_resource_occupation) else np.nan,
        "human_resource_fairness_jain": fairness_jain,
        "human_delayed_share_pct": delayed_share_human * 100 if pd.notna(delayed_share_human) else np.nan,
    }


results = pd.DataFrame([compute_metrics(f) for f in FILES])
print(results.sort_values("strategy").to_string(index=False))