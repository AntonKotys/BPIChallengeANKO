"""
Task 1.4 Performance Metrics V4 - Matched to Simulation V1.5

Uses the SAME activity mapping as the simulation engine to ensure
fair apple-to-apple comparison.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set


# ============================================================
# ACTIVITY MAPPING (same as simulation_engine_core_V1.5)
# ============================================================

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

# ============================================================
# PROCESS MODEL (same as simulation_engine_core_V1.6)
# ============================================================

PROCESS_MODEL = {
    "A_Create Application": ["A_Submitted", "W_Complete application & A_Concept"],
    "A_Submitted": ["W_Handle leads", "W_Complete application & A_Concept"],
    "W_Handle leads": ["W_Complete application & A_Concept"],
    "W_Complete application & A_Concept": ["A_Accepted"],
    "A_Accepted": ["O_Create Offer & O_Created"],
    "O_Create Offer & O_Created": ["O_Sent (mail and online)"],
    "O_Sent (mail and online)": ["W_Call after offers & A_Complete", "O_Create Offer & O_Created"],
    "W_Call after offers & A_Complete": ["W_Validate application & A_Validating", "O_Create Offer & O_Created", "A_Cancelled & O_Cancelled"],
    "W_Validate application & A_Validating": ["O_Returned", "A_Incomplete", "A_Validating"],
    "O_Returned": ["W_Validate application & A_Validating", "END"],
    "A_Incomplete": ["W_Validate application & A_Validating", "END"],
    "A_Validating": ["O_Returned", "W_Validate application & A_Validating", "END"],
    "A_Cancelled & O_Cancelled": ["END"]
}

GATEWAYS = {
    "A_Create Application": "xor",
    "A_Submitted": "xor",
    "O_Returned": "xor",
    "A_Incomplete": "xor",
    "A_Validating": "xor",
    "W_Validate application & A_Validating": "xor",
    "W_Call after offers & A_Complete": "or",
    "O_Sent (mail and online)": "or",
}


def map_activity(activity: str) -> str:
    return ACTIVITY_MAPPING.get(activity, activity)


def load_log(csv_path: str, apply_mapping: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    rename_map = {
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'time:timestamp': 'timestamp'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    if apply_mapping and 'activity' in df.columns:
        df['activity'] = df['activity'].apply(map_activity)

    return df


def extract_transitions(df: pd.DataFrame) -> Dict[str, Counter]:
    df = df.sort_values(['case_id', 'timestamp'])
    transitions = defaultdict(Counter)

    for case_id, group in df.groupby('case_id', sort=False):
        activities = group['activity'].tolist()
        for i in range(len(activities) - 1):
            transitions[activities[i]][activities[i + 1]] += 1

    return dict(transitions)


def check_validity(sim_csv: str, historical_csv: str, verbose: bool = True) -> Dict:
    """Check transition validity against process model and historical data."""

    df_sim = load_log(sim_csv, apply_mapping=False)
    df_hist = load_log(historical_csv, apply_mapping=True)

    sim_trans = extract_transitions(df_sim)
    hist_trans = extract_transitions(df_hist)

    # Historical transitions as set
    hist_set = set()
    for from_act, to_counts in hist_trans.items():
        for to_act in to_counts.keys():
            hist_set.add((from_act, to_act))

    # Model transitions as set
    model_set = set()
    for from_act, to_acts in PROCESS_MODEL.items():
        for to_act in to_acts:
            model_set.add((from_act, to_act))

    results = {
        "total_types": 0,
        "total_instances": 0,
        "valid_model": 0,
        "invalid_model": [],
        "invalid_instances": 0,
        "in_historical": 0,
        "not_in_historical": [],
    }

    for from_act, to_counts in sim_trans.items():
        for to_act, count in to_counts.items():
            results["total_types"] += 1
            results["total_instances"] += count

            # Check model validity
            if from_act == "END":
                results["valid_model"] += 1
            elif (from_act, to_act) in model_set or from_act not in PROCESS_MODEL:
                results["valid_model"] += 1
            else:
                results["invalid_model"].append({"from": from_act, "to": to_act, "count": count})
                results["invalid_instances"] += count

            # Check historical
            if (from_act, to_act) in hist_set:
                results["in_historical"] += 1
            else:
                results["not_in_historical"].append({"from": from_act, "to": to_act, "count": count})

    if verbose:
        print("\n" + "=" * 70)
        print("  TRANSITION VALIDITY")
        print("=" * 70)
        print(f"\nFile: {sim_csv}")
        print(f"Total: {results['total_types']} types, {results['total_instances']} instances")

        n_inv = len(results['invalid_model'])
        if n_inv == 0:
            print("\n✅ All transitions VALID per process model")
        else:
            print(f"\n❌ {n_inv} INVALID types ({results['invalid_instances']} instances)")
            for item in results['invalid_model'][:5]:
                print(f"   {item['from']} → {item['to']} ({item['count']}x)")

        n_unseen = len(results['not_in_historical'])
        if n_unseen == 0:
            print("✅ All transitions seen in historical data")
        else:
            print(f"⚠️  {n_unseen} not in historical (may be valid but rare)")

    return results


def compare_xor(sim_csv: str, historical_csv: str, verbose: bool = True) -> Dict:
    """Compare XOR branching probabilities."""

    df_sim = load_log(sim_csv, apply_mapping=False)
    df_hist = load_log(historical_csv, apply_mapping=True)

    sim_trans = extract_transitions(df_sim)
    hist_trans = extract_transitions(df_hist)

    xor_gateways = [k for k, v in GATEWAYS.items() if v == "xor"]

    results = {}

    for gw in xor_gateways:
        valid_succs = set(PROCESS_MODEL.get(gw, []))

        sim_counts = sim_trans.get(gw, Counter())
        hist_counts = hist_trans.get(gw, Counter())

        # Filter to valid successors
        sim_valid = {k: v for k, v in sim_counts.items() if k in valid_succs}
        hist_valid = {k: v for k, v in hist_counts.items() if k in valid_succs}

        sim_total = sum(sim_valid.values())
        hist_total = sum(hist_valid.values())

        gw_result = {
            "valid_successors": list(valid_succs),
            "sim_total": sim_total,
            "hist_total": hist_total,
            "successors": {},
            "total_diff": 0.0,
        }

        for succ in valid_succs:
            sim_pct = (sim_valid.get(succ, 0) / sim_total * 100) if sim_total > 0 else 0
            hist_pct = (hist_valid.get(succ, 0) / hist_total * 100) if hist_total > 0 else 0
            diff = sim_pct - hist_pct

            gw_result["successors"][succ] = {
                "sim_pct": sim_pct,
                "hist_pct": hist_pct,
                "diff": diff,
                "abs_diff": abs(diff),
            }
            gw_result["total_diff"] += abs(diff)

        results[gw] = gw_result

    if verbose:
        print("\n" + "=" * 70)
        print("  XOR BRANCHING COMPARISON")
        print("=" * 70)

        for gw, data in results.items():
            print(f"\n{'─' * 70}")
            print(f"  {gw}")
            print(f"  Successors: {data['valid_successors']}")

            if data['hist_total'] == 0:
                print(f"  ⚠️  No historical data for this gateway!")
                continue

            print(f"{'─' * 70}")
            print(f"  {'Successor':<40} {'Hist':>8} {'Sim':>8} {'Diff':>8}")

            for succ, stats in sorted(data['successors'].items(), key=lambda x: -x[1]['hist_pct']):
                marker = "✓" if stats['abs_diff'] < 5 else "△" if stats['abs_diff'] < 15 else "✗"
                print(f"  {marker} {succ:<38} {stats['hist_pct']:>7.1f}% {stats['sim_pct']:>7.1f}% {stats['diff']:>+7.1f}%")

            print(f"\n  Historical: {data['hist_total']}, Simulated: {data['sim_total']}")
            print(f"  Sum of absolute differences: {data['total_diff']:.1f}%")

    return results


def evaluate(sim_csv: str, historical_csv: str, label: str = "Simulation", verbose: bool = True) -> Dict:
    """Full evaluation of a simulation."""
    if verbose:
        print("\n" + "#" * 70)
        print(f"#  EVALUATING: {label}")
        print("#" * 70)

    validity = check_validity(sim_csv, historical_csv, verbose)
    xor = compare_xor(sim_csv, historical_csv, verbose)

    # Calculate summary
    gateways_with_data = [g for g, d in xor.items() if d['hist_total'] > 0]
    avg_diff = np.mean([xor[g]['total_diff'] for g in gateways_with_data]) if gateways_with_data else 0

    summary = {
        "label": label,
        "invalid_types": len(validity['invalid_model']),
        "invalid_instances": validity['invalid_instances'],
        "avg_xor_diff": avg_diff,
        "gateways_compared": len(gateways_with_data),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  Model Validity: {'✅ PASS' if summary['invalid_types'] == 0 else '❌ FAIL'}")
        print(f"    Invalid: {summary['invalid_types']} types, {summary['invalid_instances']} instances")
        print(f"  XOR Accuracy ({summary['gateways_compared']} gateways):")
        print(f"    Average diff: {summary['avg_xor_diff']:.1f}% (lower is better)")

    return summary


def compare_simulations(random_csv: str, trained_csv: str, historical_csv: str):
    """Compare RANDOM vs TRAINED predictor."""
    print("\n" + "█" * 70)
    print("█  BENCHMARK: RANDOM vs TRAINED PREDICTOR")
    print("█" * 70)

    # Show historical data analysis
    print("\n" + "=" * 70)
    print("  HISTORICAL DATA (after mapping)")
    print("=" * 70)

    df_hist = load_log(historical_csv, apply_mapping=True)
    hist_trans = extract_transitions(df_hist)

    for gw in ["A_Create Application", "A_Submitted", "O_Returned", "A_Incomplete", "A_Validating"]:
        if gw in hist_trans:
            valid_succs = set(PROCESS_MODEL.get(gw, []))
            counts = hist_trans[gw]
            total = sum(counts.values())
            valid_total = sum(c for s, c in counts.items() if s in valid_succs)

            print(f"\n  {gw}:")
            print(f"    Valid successors: {list(valid_succs)}")
            print(f"    Coverage: {valid_total}/{total} ({valid_total/total*100:.1f}%)")
            for succ, count in counts.most_common(5):
                pct = count / total * 100
                mark = "✓" if succ in valid_succs else "✗"
                print(f"      {mark} {succ}: {count} ({pct:.1f}%)")

    # Evaluate both
    random_res = evaluate(random_csv, historical_csv, "RANDOM", verbose=True)
    trained_res = evaluate(trained_csv, historical_csv, "TRAINED", verbose=True)

    # Final comparison
    print("\n" + "█" * 70)
    print("█  FINAL COMPARISON")
    print("█" * 70)

    print("\n  Metric                              RANDOM      TRAINED      Winner")
    print("  " + "-" * 65)

    r_inv = random_res['invalid_instances']
    t_inv = trained_res['invalid_instances']
    winner = "TIE" if r_inv == t_inv else ("TRAINED" if t_inv < r_inv else "RANDOM")
    print(f"  Invalid instances                   {r_inv:>6}      {t_inv:>6}       {winner}")

    r_xor = random_res['avg_xor_diff']
    t_xor = trained_res['avg_xor_diff']
    winner = "TIE" if abs(r_xor - t_xor) < 1 else ("TRAINED" if t_xor < r_xor else "RANDOM")
    print(f"  Avg XOR diff (%)                    {r_xor:>6.1f}      {t_xor:>6.1f}       {winner}")

    # Overall
    r_score = r_inv * 10 + r_xor
    t_score = t_inv * 10 + t_xor

    print("\n  " + "=" * 65)
    if t_score < r_score - 5:
        print("  🏆 WINNER: TRAINED Predictor")
    elif r_score < t_score - 5:
        print("  🏆 WINNER: RANDOM Branching")
    else:
        print("  🤝 TIE (within margin)")
    print(f"     Score: RANDOM={r_score:.1f}, TRAINED={t_score:.1f} (lower is better)")

    return {"random": random_res, "trained": trained_res}


if __name__ == "__main__":
    import os
    import sys

    hist = "bpi2017.csv"
    sim_r = "sim_random.csv"
    sim_t = "sim_predicted.csv"

    if not os.path.exists(hist):
        print(f"❌ Not found: {hist}")
        sys.exit(1)

    if os.path.exists(sim_r) and os.path.exists(sim_t):
        compare_simulations(sim_r, sim_t, hist)
    elif os.path.exists(sim_r):
        evaluate(sim_r, hist, "RANDOM")
    elif os.path.exists(sim_t):
        evaluate(sim_t, hist, "TRAINED")
    else:
        print("❌ No simulation files found!")
        sys.exit(1)

    print("\n✅ Evaluation complete!")