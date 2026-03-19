"""Quick evaluation: compare old GradientBoosting vs new RandomForest on sampled data."""
import numpy as np
import pandas as pd
from collections import defaultdict
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from simulation_engine_core_final_version import PROCESS_MODEL, GATEWAYS
from task_1_4_next_activity import ExpertActivityPredictor

ACTIVITY_MAPPING = {
    "A_Create Application": "A_Create Application",
    "A_Submitted": "A_Submitted",
    "W_Handle leads": "W_Handle leads",
    "O_Returned": "O_Returned",
    "O_Sent (mail and online)": "O_Sent (mail and online)",
    "O_Sent (online only)": "O_Sent (mail and online)",
    "A_Accepted": "A_Accepted",
    "A_Incomplete": "A_Incomplete",
    "A_Validating": "A_Validating",
    "W_Complete application": "W_Complete application & A_Concept",
    "A_Concept": "W_Complete application & A_Concept",
    "O_Create Offer": "O_Create Offer & O_Created",
    "O_Created": "O_Create Offer & O_Created",
    "W_Call after offers": "W_Call after offers & A_Complete",
    "A_Complete": "W_Call after offers & A_Complete",
    "W_Validate application": "W_Validate application & A_Validating",
    "A_Cancelled": "A_Cancelled & O_Cancelled",
    "O_Cancelled": "A_Cancelled & O_Cancelled",
    "A_Denied": "END",
    "O_Refused": "END",
    "O_Accepted": "END",
    "W_Call incomplete files": "W_Validate application & A_Validating",
    "A_Pending": "A_Pending",
    "W_Assess potential fraud": "W_Assess potential fraud",
    "W_Personal Loan collection": "W_Personal Loan collection",
    "W_Shortened completion ": "W_Shortened completion",
}

print("Loading CSV (sampling 5000 cases for speed)...")
t0 = time.time()
df = pd.read_csv("bpi2017.csv")
for old, new in {"case:concept:name": "case_id", "concept:name": "activity", "time:timestamp": "timestamp"}.items():
    if old in df.columns and new not in df.columns:
        df[new] = df[old]
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

# Sample 5000 cases
np.random.seed(42)
all_cases = df["case_id"].unique()
sample_cases = np.random.choice(all_cases, size=min(5000, len(all_cases)), replace=False)
df = df[df["case_id"].isin(set(sample_cases))].reset_index(drop=True)

# Apply mapping + dedup
df["activity"] = df["activity"].map(lambda a: ACTIVITY_MAPPING.get(a, a))
prev_case, prev_act, keep = None, None, []
for _, row in df.iterrows():
    if row["case_id"] != prev_case or row["activity"] != prev_act:
        keep.append(True)
    else:
        keep.append(False)
    prev_case, prev_act = row["case_id"], row["activity"]
df = df[keep].reset_index(drop=True)
print(f"Loaded in {time.time()-t0:.0f}s. Cases: {df['case_id'].nunique()}, Events: {len(df)}")

xor_acts = {a: s for a, s in PROCESS_MODEL.items() if len(s) > 1}

# Split
cases = df["case_id"].unique()
np.random.seed(42)
np.random.shuffle(cases)
split = int(len(cases) * 0.8)
train_cases = set(cases[:split])
test_cases = set(cases[split:])
train_df = df[df["case_id"].isin(train_cases)]
test_df = df[df["case_id"].isin(test_cases)]

def eval_accuracy(predictor, test_df, xor_acts, use_predict=False):
    correct, total = 0, 0
    per_gw = defaultdict(lambda: {"correct": 0, "total": 0})
    for cid, grp in test_df.groupby("case_id", sort=False):
        acts = grp["activity"].tolist()
        timestamps = grp["timestamp"].tolist()
        for i in range(len(acts) - 1):
            curr, nxt = acts[i], acts[i+1]
            if curr in xor_acts and nxt in xor_acts[curr]:
                prefix = acts[:i+1]
                if use_predict:
                    pred = predictor.predict_next_activity(
                        prefix,
                        current_timestamp=timestamps[i],
                        case_start_timestamp=timestamps[0],
                        enabled_next=xor_acts[curr],
                        current_activity=curr,
                    )
                else:
                    pred = predictor.sample_next_activity(prefix, enabled_next=xor_acts[curr], current_activity=curr)
                total += 1
                per_gw[curr]["total"] += 1
                if pred == nxt:
                    correct += 1
                    per_gw[curr]["correct"] += 1
    return correct / total if total > 0 else 0, total, correct, dict(per_gw)

# --- BASIC ---
print("\n--- BASIC (k-gram) ---")
t0 = time.time()
p_basic = ExpertActivityPredictor(mode="basic", basic_context_k=2, process_model=PROCESS_MODEL, gateways=GATEWAYS)
p_basic.fit(train_df)
acc, tot, cor, gw = eval_accuracy(p_basic, test_df, xor_acts)
print(f"Accuracy: {acc:.1%} ({cor}/{tot}) [{time.time()-t0:.0f}s]")
for g, s in sorted(gw.items()):
    print(f"  {g}: {s['correct']/s['total']:.1%} ({s['correct']}/{s['total']})" if s['total'] > 0 else f"  {g}: no data")

# --- ADVANCED (GradientBoosting - old) ---
print("\n--- ADVANCED (GradientBoosting) ---")
t0 = time.time()
p_adv = ExpertActivityPredictor(mode="advanced", basic_context_k=2, process_model=PROCESS_MODEL, gateways=GATEWAYS)
p_adv.fit(train_df)
acc_adv, tot_adv, cor_adv, gw_adv = eval_accuracy(p_adv, test_df, xor_acts, use_predict=True)
print(f"Accuracy: {acc_adv:.1%} ({cor_adv}/{tot_adv}) [{time.time()-t0:.0f}s]")
for g, s in sorted(gw_adv.items()):
    print(f"  {g}: {s['correct']/s['total']:.1%} ({s['correct']}/{s['total']})" if s['total'] > 0 else f"  {g}: no data")

# --- ADVANCED ENRICHED (RandomForest - new) ---
print("\n--- ADVANCED ENRICHED (RandomForest - NEW) ---")
t0 = time.time()
p_rf = ExpertActivityPredictor(mode="advanced_enriched", basic_context_k=2, process_model=PROCESS_MODEL, gateways=GATEWAYS)
p_rf.fit(train_df)
acc_rf, tot_rf, cor_rf, gw_rf = eval_accuracy(p_rf, test_df, xor_acts, use_predict=True)
print(f"Accuracy: {acc_rf:.1%} ({cor_rf}/{tot_rf}) [{time.time()-t0:.0f}s]")
for g, s in sorted(gw_rf.items()):
    print(f"  {g}: {s['correct']/s['total']:.1%} ({s['correct']}/{s['total']})" if s['total'] > 0 else f"  {g}: no data")

print("\n--- SUMMARY ---")
print(f"{'Model':<35} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
print("-" * 65)
print(f"{'Basic (k-gram)':<35} {acc:.1%}{cor:>10}/{tot}")
print(f"{'Advanced (GradientBoosting)':<35} {acc_adv:.1%}{cor_adv:>10}/{tot_adv}")
print(f"{'Advanced Enriched (RandomForest)':<35} {acc_rf:.1%}{cor_rf:>10}/{tot_rf}")
