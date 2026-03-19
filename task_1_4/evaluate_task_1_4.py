"""
Comprehensive evaluation of Task 1.4: Next Activity Prediction.
Computes all metrics from scratch for both basic and advanced modes.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation_engine_core_final_version import PROCESS_MODEL, GATEWAYS
from task_1_4.task_1_4_next_activity import ExpertActivityPredictor


def load_log():
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
    return df


def get_xor_activities():
    """Activities that have XOR gateways (multiple successors in process model)."""
    xor_acts = {}
    for act, successors in PROCESS_MODEL.items():
        if len(successors) > 1:
            xor_acts[act] = successors
    return xor_acts


def compute_historical_xor_distribution(df, xor_activities):
    """Compute actual branching distribution from historical data at XOR points."""
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
    """Uniform distribution at each XOR gateway."""
    return {
        act: {s: 1.0 / len(succs) for s in succs}
        for act, succs in xor_activities.items()
    }


def kl_divergence(p, q, activities):
    """KL divergence D(p || q) with smoothing."""
    eps = 1e-10
    kl = 0.0
    for a in activities:
        pi = p.get(a, eps)
        qi = q.get(a, eps)
        kl += pi * np.log(pi / qi)
    return kl


def avg_xor_difference(dist_a, dist_b, xor_activities):
    """
    Average absolute difference between two distributions at XOR gateways.
    This is the "Average XOR difference" metric from the report.
    """
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


def evaluate_predictor_accuracy(predictor, df, xor_activities):
    """
    Evaluate predictor accuracy at XOR gateways using holdout traces.
    Returns: accuracy, total_predictions, correct_predictions, per-gateway stats.
    """
    cases = df["case_id"].unique()
    np.random.seed(42)
    np.random.shuffle(cases)
    split = int(len(cases) * 0.8)
    test_cases = set(cases[split:])

    test_df = df[df["case_id"].isin(test_cases)]

    correct = 0
    total = 0
    per_gateway = defaultdict(lambda: {"correct": 0, "total": 0})

    for case_id, group in test_df.groupby("case_id", sort=False):
        activities = group["activity"].tolist()
        for i in range(len(activities) - 1):
            curr = activities[i]
            nxt = activities[i + 1]
            if curr in xor_activities:
                valid = xor_activities[curr]
                if nxt in valid:
                    prefix = activities[:i + 1]
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
    """
    Evaluate how well the predictor's distribution matches historical distribution.
    Uses the predictor's get_next_activity_distribution at each XOR gateway.
    """
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
    print("TASK 1.4 EVALUATION: Next Activity Prediction")
    print("=" * 70)

    print("\n[1] Loading event log...")
    df = load_log()
    print(f"    Cases: {df['case_id'].nunique()}, Events: {len(df)}")

    xor_activities = get_xor_activities()
    print(f"\n[2] XOR gateways found: {len(xor_activities)}")
    for act, succs in xor_activities.items():
        print(f"    {act} -> {succs}")

    print("\n[3] Computing historical XOR distribution...")
    hist_probs = compute_historical_xor_distribution(df, xor_activities)
    for act, probs in hist_probs.items():
        print(f"    {act}: {probs}")

    random_dist = compute_random_xor_distribution(xor_activities)

    # =============================================
    # BASIC MODE
    # =============================================
    print("\n" + "=" * 70)
    print("BASIC MODE (k-gram, process-model-aware)")
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
    print(f"[Basic] Improvement: {diff_random - diff_basic:.1f} percentage points")

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
    print("ADVANCED MODE (GradientBoosting)")
    print("=" * 70)

    predictor_adv = ExpertActivityPredictor(
        mode="advanced",
        basic_context_k=2,
        process_model=PROCESS_MODEL,
        gateways=GATEWAYS,
    )
    predictor_adv.fit(df)

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
    # TOKEN REPLAY (Advanced Task)
    # =============================================
    print("\n" + "=" * 70)
    print("TOKEN REPLAY MODE")
    print("=" * 70)

    bpmn_path = "bpianko9.0.bpmn"
    if os.path.exists(bpmn_path):
        predictor_replay = ExpertActivityPredictor(
            mode="basic",
            basic_context_k=2,
            process_model=PROCESS_MODEL,
            gateways=GATEWAYS,
            use_token_replay=True,
        )
        predictor_replay.fit(df, bpmn_path=bpmn_path)

        pred_dist_replay = evaluate_distribution_quality(
            predictor_replay, df, xor_activities, hist_probs
        )

        print("\n[Replay] Predicted distributions at XOR gateways:")
        for act, probs in pred_dist_replay.items():
            print(f"    {act}: {probs}")

        diff_replay = avg_xor_difference(pred_dist_replay, hist_probs, xor_activities)
        print(f"\n[Replay] Avg XOR difference (predictor vs historical): {diff_replay:.1f}%")

        print("\n[Replay] Evaluating prediction accuracy on holdout set...")
        acc_replay, total_replay, correct_replay, gw_replay = evaluate_predictor_accuracy(
            predictor_replay, df, xor_activities
        )
        print(f"[Replay] Accuracy: {acc_replay:.1%} ({correct_replay}/{total_replay})")
        print("[Replay] Per-gateway accuracy:")
        for gw, stats in sorted(gw_replay.items()):
            gw_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"    {gw}: {gw_acc:.1%} ({stats['correct']}/{stats['total']})")

        if hasattr(predictor_replay, 'replay_stats') and predictor_replay.replay_stats:
            rs = predictor_replay.replay_stats
            print(f"\n[Replay] Token replay stats:")
            print(f"    Total traces: {rs.get('total_traces', 'N/A')}")
            print(f"    Fit traces: {rs.get('fit_traces', 'N/A')}")
            print(f"    Decision instances: {rs.get('decision_instances', 'N/A')}")
            print(f"    Decision places: {rs.get('decision_places_found', 'N/A')}")
    else:
        print(f"    BPMN file not found: {bpmn_path}")
        print("    Skipping token replay evaluation.")

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<45} {'Random':>10} {'Basic':>10} {'Advanced':>10}")
    print("-" * 75)
    print(f"{'Avg XOR diff vs historical (%)':<45} {diff_random:>10.1f} {diff_basic:>10.1f} {'N/A':>10}")
    print(f"{'Prediction accuracy on holdout (%)':<45} {'~50%':>10} {acc_basic*100:>10.1f} {acc_adv*100:>10.1f}")
    print(f"{'Total XOR predictions evaluated':<45} {'':>10} {total_basic:>10} {total_adv:>10}")

    if os.path.exists(bpmn_path):
        print(f"\n{'Token Replay metrics:'}")
        print(f"{'Avg XOR diff vs historical (%)':<45} {'':>10} {diff_replay:>10.1f}")
        print(f"{'Prediction accuracy on holdout (%)':<45} {'':>10} {acc_replay*100:>10.1f}")


if __name__ == "__main__":
    main()
