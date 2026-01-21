"""
Gateway Branch Comparison Report

Compares XOR and OR gateway branch distributions across:
1. Historical data (real event log)
2. Trained simulation (learned probabilities)
3. Random simulation (uniform random)

For each gateway, shows the percentage of times each branch was taken
and calculates deviation from historical data.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime

# ============================================================
# PROCESS MODEL AND GATEWAY DEFINITIONS
# ============================================================

PROCESS_MODEL = {
    "A_Create Application": ["A_Submitted", "W_Complete application & A_Concept"],
    "A_Submitted": ["W_Handle leads", "W_Complete application & A_Concept"],
    "W_Handle leads": ["W_Complete application & A_Concept"],
    "W_Complete application & A_Concept": ["A_Accepted"],
    "A_Accepted": ["O_Create Offer & O_Created"],
    "O_Create Offer & O_Created": ["O_Sent (mail and online)"],
    "O_Sent (mail and online)": ["W_Call after offers & A_Complete"],
    "W_Call after offers & A_Complete": [
        "W_Validate application & A_Validating",
        "O_Create Offer & O_Created",
        "A_Cancelled & O_Cancelled"
    ],
    "W_Validate application & A_Validating": ["O_Returned"],
    "O_Returned": ["A_Incomplete", "END"],
    "A_Incomplete": ["A_Validating", "END"],
    "A_Validating": ["A_Incomplete", "END"],
    "A_Cancelled & O_Cancelled": ["END"],
}

GATEWAYS = {
    "A_Create Application": "XOR",
    "A_Submitted": "XOR",
    "O_Returned": "XOR",
    "A_Incomplete": "XOR",
    "A_Validating": "XOR",
    "W_Call after offers & A_Complete": "OR",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names."""
    df = df.copy()
    rename_map = {
        'case:concept:name': 'case_id',
        'concept:name': 'activity',
        'time:timestamp': 'timestamp'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def count_gateway_transitions(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Count transitions at each gateway.

    Returns: Dict[gateway -> Dict[next_activity -> count]]
    """
    df = standardize_columns(df)
    if 'timestamp' in df.columns:
        df = df.sort_values(['case_id', 'timestamp'])

    counts = {gw: defaultdict(int) for gw in GATEWAYS}

    for case_id, group in df.groupby('case_id', sort=False):
        activities = group['activity'].tolist()

        for i in range(len(activities) - 1):
            curr = activities[i]
            nxt = activities[i + 1]

            if curr in GATEWAYS:
                valid = PROCESS_MODEL.get(curr, [])
                if nxt in valid:
                    counts[curr][nxt] += 1

    return {gw: dict(c) for gw, c in counts.items()}


def counts_to_percentages(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert counts to percentages."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {act: count / total * 100 for act, count in counts.items()}


def generate_comparison_report(
        historical_path: str,
        predicted_path: str,
        random_path: str,
        output_path: str = "gateway_comparison_report.txt"
) -> Dict:
    """
    Generate comparison report of gateway branching behavior.

    Returns: Dict with all comparison data
    """
    # Load data
    print("  Loading event logs...")
    df_hist = pd.read_csv(historical_path)
    df_pred = pd.read_csv(predicted_path)
    df_rand = pd.read_csv(random_path)

    # Standardize
    df_hist = standardize_columns(df_hist)
    df_pred = standardize_columns(df_pred)
    df_rand = standardize_columns(df_rand)

    # Count transitions
    print("  Counting transitions...")
    hist_counts = count_gateway_transitions(df_hist)
    pred_counts = count_gateway_transitions(df_pred)
    rand_counts = count_gateway_transitions(df_rand)

    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'case_counts': {
            'historical': df_hist['case_id'].nunique(),
            'predicted': df_pred['case_id'].nunique(),
            'random': df_rand['case_id'].nunique()
        },
        'gateways': {}
    }

    # Analyze each gateway
    for gateway, gw_type in GATEWAYS.items():
        outgoing = PROCESS_MODEL.get(gateway, [])
        if len(outgoing) < 2:
            continue

        hist_pct = counts_to_percentages(hist_counts.get(gateway, {}))
        pred_pct = counts_to_percentages(pred_counts.get(gateway, {}))
        rand_pct = counts_to_percentages(rand_counts.get(gateway, {}))

        # Calculate deviation from historical
        pred_dev = sum(abs(hist_pct.get(a, 0) - pred_pct.get(a, 0)) for a in outgoing)
        rand_dev = sum(abs(hist_pct.get(a, 0) - rand_pct.get(a, 0)) for a in outgoing)

        report['gateways'][gateway] = {
            'type': gw_type,
            'outgoing': outgoing,
            'historical': hist_pct,
            'predicted': pred_pct,
            'random': rand_pct,
            'hist_counts': hist_counts.get(gateway, {}),
            'pred_counts': pred_counts.get(gateway, {}),
            'rand_counts': rand_counts.get(gateway, {}),
            'pred_deviation': pred_dev,
            'rand_deviation': rand_dev
        }

    # Generate text report
    lines = []
    lines.append("=" * 80)
    lines.append("  GATEWAY BRANCH COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {report['timestamp']}")
    lines.append(f"\nCase counts:")
    lines.append(f"  Historical: {report['case_counts']['historical']}")
    lines.append(f"  Predicted:  {report['case_counts']['predicted']}")
    lines.append(f"  Random:     {report['case_counts']['random']}")

    # Summary
    total_pred_dev = sum(g['pred_deviation'] for g in report['gateways'].values())
    total_rand_dev = sum(g['rand_deviation'] for g in report['gateways'].values())

    lines.append("\n" + "-" * 80)
    lines.append("  EXECUTIVE SUMMARY")
    lines.append("-" * 80)
    lines.append(f"\nTotal deviation from historical:")
    lines.append(f"  Trained Predictor: {total_pred_dev:.1f}%")
    lines.append(f"  Random Simulation: {total_rand_dev:.1f}%")

    if total_rand_dev > 0:
        improvement = ((total_rand_dev - total_pred_dev) / total_rand_dev) * 100
        if improvement > 0:
            lines.append(f"\n  ✓ Predictor is {improvement:.1f}% closer to historical than random")
        else:
            lines.append(f"\n  ⚠ Random performed {-improvement:.1f}% better (unusual)")

    # Detailed analysis
    lines.append("\n" + "=" * 80)
    lines.append("  DETAILED GATEWAY ANALYSIS")
    lines.append("=" * 80)

    for gateway, data in report['gateways'].items():
        lines.append(f"\n{'─' * 80}")
        lines.append(f"  {gateway}")
        lines.append(f"  Type: {data['type']} Gateway")
        lines.append(f"{'─' * 80}")

        # Table header
        lines.append(f"\n  {'Branch':<42} {'Hist':>8} {'Train':>8} {'Random':>8}")
        lines.append(f"  {'-' * 66}")

        for act in data['outgoing']:
            h = data['historical'].get(act, 0)
            p = data['predicted'].get(act, 0)
            r = data['random'].get(act, 0)

            # Mark if trained is closer
            marker = " ✓" if abs(h - p) < abs(h - r) else ""

            act_short = act[:40] + ".." if len(act) > 42 else act
            lines.append(f"  {act_short:<42} {h:>7.1f}% {p:>7.1f}% {r:>7.1f}%{marker}")

        lines.append(f"\n  Deviation from historical:")
        lines.append(f"    Trained: {data['pred_deviation']:.1f}%")
        lines.append(f"    Random:  {data['rand_deviation']:.1f}%")

        if data['pred_deviation'] < data['rand_deviation']:
            lines.append(f"    → Trained is better by {data['rand_deviation'] - data['pred_deviation']:.1f}%")

    # Conclusion
    lines.append(f"\n{'=' * 80}")
    lines.append("  CONCLUSION")
    lines.append("=" * 80)

    if total_pred_dev < total_rand_dev:
        lines.append("\n  ✓ The trained predictor produces branching distributions closer to")
        lines.append("    historical data than random simulation.")
    else:
        lines.append("\n  ⚠ Random simulation performed comparably or better.")
        lines.append("    Consider: more training data, different k-gram, or data variance.")

    lines.append("\n  Key: ✓ = Trained simulation closer to historical than random")
    lines.append("")

    # Write report
    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"  📊 Report saved: {output_path}")

    return report


def print_report(report: Dict):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("  GATEWAY COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n  Cases: Hist={report['case_counts']['historical']}, "
          f"Pred={report['case_counts']['predicted']}, "
          f"Rand={report['case_counts']['random']}")

    print(f"\n  {'Gateway':<42} {'Train Dev':>12} {'Rand Dev':>12}")
    print(f"  {'-' * 66}")

    for gateway, data in report['gateways'].items():
        gw_short = gateway[:40] + ".." if len(gateway) > 42 else gateway
        better = "✓" if data['pred_deviation'] < data['rand_deviation'] else ""
        print(f"  {gw_short:<42} {data['pred_deviation']:>11.1f}% {data['rand_deviation']:>11.1f}% {better}")

    total_pred = sum(g['pred_deviation'] for g in report['gateways'].values())
    total_rand = sum(g['rand_deviation'] for g in report['gateways'].values())

    print(f"  {'-' * 66}")
    print(f"  {'TOTAL':<42} {total_pred:>11.1f}% {total_rand:>11.1f}%")

    if total_rand > 0:
        improvement = ((total_rand - total_pred) / total_rand) * 100
        print(f"\n  Trained predictor is {improvement:.1f}% closer to historical")


if __name__ == "__main__":
    import sys

    hist = sys.argv[1] if len(sys.argv) > 1 else "bpi2017.csv"
    pred = sys.argv[2] if len(sys.argv) > 2 else "sim_predicted.csv"
    rand = sys.argv[3] if len(sys.argv) > 3 else "sim_random.csv"
    out = sys.argv[4] if len(sys.argv) > 4 else "gateway_comparison_report.txt"

    report = generate_comparison_report(hist, pred, rand, out)
    print_report(report)