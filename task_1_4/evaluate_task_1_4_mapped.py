"""
Task 1.4 Evaluation with Activity Mapping Applied to Event Log.

This script applies the ACTIVITY_MAPPING to the raw event log BEFORE training
and evaluation, so that the event log activities match the PROCESS_MODEL.
This enables evaluation on ALL 7 XOR/OR gateways, not just the 2 that
happen to have unmapped names.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation_engine_core_final_version import PROCESS_MODEL, GATEWAYS
from task_1_4.task_1_4_next_activity import ExpertActivityPredictor

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


def load_and_map_log():
    """Load event log and apply activity mapping + deduplication of consecutive same activities."""
    df = pd.read_csv("bpi2017.csv")
    rename = {
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    df["activity_raw"] = df["activity"]
    df["activity"] = df["activity"].map(lambda a: ACTIVITY_MAPPING.get(a, a))

    # After mapping, consecutive duplicate activities within a case should be collapsed
    # e.g. W_Complete application -> "W_Complete application & A_Concept"
    #      A_Concept             -> "W_Complete application & A_Concept"
    # These two consecutive rows now have the same mapped name; keep only the first occurrence.
    prev_case = None
    prev_act = None
    keep = []
    for idx, row in df.iterrows():
        if row["case_id"] != prev_case or row["activity"] != prev_act:
            keep.append(True)
        else:
            keep.append(False)
        prev_case = row["case_id"]
        prev_act = row["activity"]

    df_dedup = df[keep].reset_index(drop=True)
    return df_dedup


def get_xor_activities():
    xor_acts = {}
    for act, successors in PROCESS_MODEL.items():
        if len(successors) > 1:
            xor_acts[act] = successors
    return xor_acts


def compute_historical_xor_distribution(df, xor_activities):
    hist_dist = {}
    for case_id, group in df.groupby("case_id", sort=False):
        activities = group["activity"].tolist()
        for i in range(len(activities) - 1):
            curr = activities[i]
            nxt = activities[i + 1]
            if curr in xor_activities:
                valid = xor_activities[curr]
                if nxt in valid:
                    if curr not in hist_dist:
                        hist_dist[curr] = defaultdict(int)
                    hist_dist[curr][nxt] += 1
    hist_probs = {}
    for act, counts in hist_dist.items():
        total = sum(counts.values())
        hist_probs[act] = {k: v / total for k, v in counts.items()}
    return hist_probs


def compute_random_xor_distribution(xor_activities):
    return {
        act: {s: 1.0 / len(succs) for s in succs}
        for act, succs in xor_activities.items()
    }


def avg_xor_difference(dist_a, dist_b, xor_activities):
    diffs = []
    for act, succs in xor_activities.items():
        if act not in dist_a or act not in dist_b:
            continue
        da = dist_a[act]
        db = dist_b[act]
        for s in succs:
            diff = abs(da.get(s, 0.0) - db.get(s, 0.0))
            diffs.append(diff)
    return np.mean(diffs) * 100 if diffs else 0.0


def evaluate_predictor_accuracy(predictor, df, xor_activities, use_enriched=False):
    cases = df["case_id"].unique()
    np.random.seed(42)
    np.random.shuffle(cases)
    split = int(len(cases) * 0.8)
    test_cases = set(cases[split:])
    test_df = df[df["case_id"].isin(test_cases)]

    offer_activities = {"O_Create Offer", "O_Create Offer & O_Created", "O_Created"}

    correct = 0
    total = 0
    per_gateway = defaultdict(lambda: {"correct": 0, "total": 0})

    for case_id, group in test_df.groupby("case_id", sort=False):
        activities = group["activity"].tolist()
        timestamps = group["timestamp"].tolist()

        case_attrs = {}
        if use_enriched:
            case_attrs = {
                "case:RequestedAmount": group["case:RequestedAmount"].iloc[0] if "case:RequestedAmount" in group.columns else 0,
                "case:LoanGoal": group["case:LoanGoal"].iloc[0] if "case:LoanGoal" in group.columns else "Unknown",
                "case:ApplicationType": group["case:ApplicationType"].iloc[0] if "case:ApplicationType" in group.columns else "Unknown",
            }
            credit_scores = group["CreditScore"].values if "CreditScore" in group.columns else np.zeros(len(group))
            offered_amounts = group["OfferedAmount"].values if "OfferedAmount" in group.columns else np.zeros(len(group))

        last_credit = 0.0
        last_offered = 0.0

        for i in range(len(activities) - 1):
            curr = activities[i]
            nxt = activities[i + 1]

            if use_enriched:
                cs = credit_scores[i]
                if not np.isnan(cs) and cs > 0:
                    last_credit = cs
                oa = offered_amounts[i]
                if not np.isnan(oa) and oa > 0:
                    last_offered = oa

            if curr in xor_activities:
                valid = xor_activities[curr]
                if nxt in valid:
                    prefix = activities[:i + 1]

                    if use_enriched:
                        case_attrs["last_credit_score"] = last_credit
                        case_attrs["last_offered_amount"] = last_offered
                        predicted = predictor.predict_next_activity(
                            prefix,
                            current_timestamp=timestamps[i],
                            case_start_timestamp=timestamps[0],
                            enabled_next=valid,
                            current_activity=curr,
                            case_attributes=case_attrs,
                        )
                    else:
                        predicted = predictor.sample_next_activity(
                            prefix, enabled_next=valid, current_activity=curr
                        )

                    total += 1
                    per_gateway[curr]["total"] += 1
                    if predicted == nxt:
                        correct += 1
                        per_gateway[curr]["correct"] += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total, correct, dict(per_gateway)


def evaluate_distribution_quality(predictor, df, xor_activities, hist_probs):
    pred_dist = {}
    for act, succs in xor_activities.items():
        dist = predictor.get_next_activity_distribution(
            [act], enabled_next=succs, current_activity=act
        )
        if dist:
            pred_dist[act] = dist
    return pred_dist


def main():
    print("=" * 70)
    print("TASK 1.4 EVALUATION — WITH ACTIVITY MAPPING APPLIED")
    print("=" * 70)

    print("\n[1] Loading event log with activity mapping...")
    df = load_and_map_log()
    print(f"    Cases: {df['case_id'].nunique()}, Events (after dedup): {len(df)}")

    unique_acts = set(df["activity"].unique())
    model_acts = set(PROCESS_MODEL.keys()) | {"END"}
    print(f"    Unique activities in mapped log: {len(unique_acts)}")
    print(f"    Activities in PROCESS_MODEL: {len(model_acts)}")
    print(f"    Overlap: {len(unique_acts & model_acts)}")
    extra = unique_acts - model_acts
    if extra:
        print(f"    Extra activities not in model: {extra}")

    xor_activities = get_xor_activities()
    print(f"\n[2] XOR/OR gateways found: {len(xor_activities)}")
    for act, succs in xor_activities.items():
        print(f"    {act} -> {succs}")

    print("\n[3] Computing historical XOR distribution (from mapped log)...")
    hist_probs = compute_historical_xor_distribution(df, xor_activities)
    gateways_with_data = 0
    gateways_without_data = []
    for act, succs in xor_activities.items():
        if act in hist_probs:
            gateways_with_data += 1
            print(f"    {act}: {hist_probs[act]}")
        else:
            gateways_without_data.append(act)
            print(f"    {act}: NO DATA")

    print(f"\n    Gateways with historical data: {gateways_with_data}/{len(xor_activities)}")
    if gateways_without_data:
        print(f"    Gateways without data: {gateways_without_data}")

    random_dist = compute_random_xor_distribution(xor_activities)

    # =============================================
    # BASIC MODE
    # =============================================
    print("\n" + "=" * 70)
    print("BASIC MODE (k-gram, process-model-aware) — MAPPED LOG")
    print("=" * 70)

    predictor_basic = ExpertActivityPredictor(
        mode="basic",
        basic_context_k=2,
        process_model=PROCESS_MODEL,
        gateways=GATEWAYS,
    )
    predictor_basic.fit(df)

    pred_dist_basic = evaluate_distribution_quality(
        predictor_basic, df, xor_activities, hist_probs
    )

    print("\n[Basic] Predicted distributions at XOR gateways:")
    for act, probs in pred_dist_basic.items():
        print(f"    {act}: {probs}")

    diff_random = avg_xor_difference(random_dist, hist_probs, xor_activities)
    diff_basic = avg_xor_difference(pred_dist_basic, hist_probs, xor_activities)
    print(f"\n[Basic] Avg XOR difference (random vs historical): {diff_random:.1f}%")
    print(f"[Basic] Avg XOR difference (predictor vs historical): {diff_basic:.1f}%")
    print(f"[Basic] Improvement: {diff_random - diff_basic:.1f} pp")

    print("\n[Basic] Evaluating prediction accuracy on holdout set...")
    acc_basic, total_basic, correct_basic, gw_basic = evaluate_predictor_accuracy(
        predictor_basic, df, xor_activities
    )
    print(f"[Basic] Accuracy: {acc_basic:.1%} ({correct_basic}/{total_basic})")
    print("[Basic] Per-gateway accuracy:")
    for gw, stats in sorted(gw_basic.items()):
        gw_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {gw}: {gw_acc:.1%} ({stats['correct']}/{stats['total']})")

    # =============================================
    # ADVANCED MODE (ML)
    # =============================================
    print("\n" + "=" * 70)
    print("ADVANCED MODE (GradientBoosting) — MAPPED LOG")
    print("=" * 70)

    predictor_adv = ExpertActivityPredictor(
        mode="advanced",
        basic_context_k=2,
        process_model=PROCESS_MODEL,
        gateways=GATEWAYS,
    )
    predictor_adv.fit(df)

    pred_dist_adv = evaluate_distribution_quality(
        predictor_adv, df, xor_activities, hist_probs
    )

    print("\n[Advanced] Predicted distributions at XOR gateways:")
    for act, probs in pred_dist_adv.items():
        print(f"    {act}: {probs}")

    diff_adv = avg_xor_difference(pred_dist_adv, hist_probs, xor_activities)
    print(f"\n[Advanced] Avg XOR difference (predictor vs historical): {diff_adv:.1f}%")

    print("\n[Advanced] Evaluating prediction accuracy on holdout set...")
    acc_adv, total_adv, correct_adv, gw_adv = evaluate_predictor_accuracy(
        predictor_adv, df, xor_activities
    )
    print(f"[Advanced] Accuracy: {acc_adv:.1%} ({correct_adv}/{total_adv})")
    print("[Advanced] Per-gateway accuracy:")
    for gw, stats in sorted(gw_adv.items()):
        gw_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {gw}: {gw_acc:.1%} ({stats['correct']}/{stats['total']})")

    # =============================================
    # ADVANCED ENRICHED MODE (ML with case-level features)
    # =============================================
    print("\n" + "=" * 70)
    print("ADVANCED ENRICHED MODE (GradientBoosting + case attributes) — MAPPED LOG")
    print("=" * 70)

    predictor_enriched = ExpertActivityPredictor(
        mode="advanced_enriched",
        basic_context_k=2,
        process_model=PROCESS_MODEL,
        gateways=GATEWAYS,
    )
    predictor_enriched.fit(df)

    pred_dist_enriched = evaluate_distribution_quality(
        predictor_enriched, df, xor_activities, hist_probs
    )

    print("\n[Enriched] Predicted distributions at XOR gateways:")
    for act, probs in pred_dist_enriched.items():
        print(f"    {act}: {probs}")

    diff_enriched = avg_xor_difference(pred_dist_enriched, hist_probs, xor_activities)
    print(f"\n[Enriched] Avg XOR difference (predictor vs historical): {diff_enriched:.1f}%")

    print("\n[Enriched] Evaluating prediction accuracy on holdout set...")
    acc_enriched, total_enriched, correct_enriched, gw_enriched = evaluate_predictor_accuracy(
        predictor_enriched, df, xor_activities, use_enriched=True
    )
    print(f"[Enriched] Accuracy: {acc_enriched:.1%} ({correct_enriched}/{total_enriched})")
    print("[Enriched] Per-gateway accuracy:")
    for gw, stats in sorted(gw_enriched.items()):
        gw_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {gw}: {gw_acc:.1%} ({stats['correct']}/{stats['total']})")

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 70)
    print("SUMMARY — MAPPED LOG EXPERIMENT")
    print("=" * 70)
    print(f"\n{'Metric':<50} {'Random':>10} {'Basic':>10} {'Advanced':>10} {'Enriched':>10}")
    print("-" * 90)
    print(f"{'Avg XOR diff vs historical (%)':<50} {diff_random:>10.1f} {diff_basic:>10.1f} {diff_adv:>10.1f} {diff_enriched:>10.1f}")
    print(f"{'Prediction accuracy on holdout (%)':<50} {'~50%':>10} {acc_basic*100:>10.1f} {acc_adv*100:>10.1f} {acc_enriched*100:>10.1f}")
    print(f"{'Total XOR predictions evaluated':<50} {'':>10} {total_basic:>10} {total_adv:>10} {total_enriched:>10}")
    print(f"{'Gateways with historical data':<50} {'':>10} {gateways_with_data:>10} {gateways_with_data:>10} {gateways_with_data:>10}")

    print("\n[Comparison with unmapped log]")
    print("  Without mapping: 2/7 gateways covered, ~8,242 XOR decisions")
    print(f"  With mapping:    {gateways_with_data}/{len(xor_activities)} gateways covered, {total_basic} XOR decisions")


if __name__ == "__main__":
    main()
