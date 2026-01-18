"""
DIAGNOSTIC SCRIPT: Activity Name Mismatch Analysis

Run this to see exactly what activities exist in your historical data
and how they should map to your process model.

Usage: python diagnose_activities.py
"""

import pandas as pd
from collections import Counter, defaultdict


def analyze_bpi_data(csv_path: str = "bpi2017.csv"):
    """Analyze the BPI Challenge 2017 data to understand activity names."""

    print("=" * 70)
    print("  BPI CHALLENGE 2017 DATA ANALYSIS")
    print("=" * 70)

    # Load data
    df = pd.read_csv(csv_path)

    # Standardize columns
    if 'concept:name' in df.columns:
        df['activity'] = df['concept:name']
    if 'case:concept:name' in df.columns:
        df['case_id'] = df['case:concept:name']
    if 'time:timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time:timestamp'], format="mixed", utc=True, errors="raise")

    df = df.sort_values(['case_id', 'timestamp'])

    # 1. Show ALL unique activities
    print("\n" + "-" * 70)
    print("  ALL UNIQUE ACTIVITIES IN HISTORICAL DATA")
    print("-" * 70)

    activity_counts = df['activity'].value_counts()
    for act, count in activity_counts.items():
        print(f"  {act}: {count:,}")

    # 2. Extract transitions
    print("\n" + "-" * 70)
    print("  TRANSITION ANALYSIS FOR XOR GATEWAYS")
    print("-" * 70)

    transitions = defaultdict(Counter)
    for case_id, group in df.groupby('case_id', sort=False):
        acts = group['activity'].tolist()
        for i in range(len(acts) - 1):
            transitions[acts[i]][acts[i + 1]] += 1

    # Check each XOR gateway in the process model
    xor_gateways = {
        "A_Create Application": ["A_Submitted", "W_Complete application & A_Concept"],
        "A_Submitted": ["W_Handle leads", "W_Complete application & A_Concept"],
        "O_Returned": ["A_Incomplete", "END"],
        "A_Incomplete": ["A_Validating", "END"],
        "A_Validating": ["A_Incomplete", "END"],
    }

    for gateway, model_successors in xor_gateways.items():
        print(f"\n  === {gateway} ===")
        print(f"  Process Model expects: {model_successors}")

        if gateway in transitions:
            actual = transitions[gateway]
            total = sum(actual.values())
            print(f"  Historical data shows (top 10):")

            for succ, count in actual.most_common(10):
                pct = count / total * 100
                in_model = "✓" if succ in model_successors else "✗"
                print(f"    {in_model} {succ}: {count:,} ({pct:.1f}%)")

            # Check how much of historical matches model
            matching = sum(actual.get(s, 0) for s in model_successors)
            match_pct = matching / total * 100 if total > 0 else 0
            print(f"  Coverage: {match_pct:.1f}% of historical transitions match process model")
        else:
            # Check for similar activity names
            similar = [a for a in activity_counts.index if gateway.lower().replace("_", " ") in a.lower()]
            print(f"  ⚠️  NOT FOUND in historical data!")
            if similar:
                print(f"  Similar activities: {similar}")

    # 3. Suggest mapping
    print("\n" + "=" * 70)
    print("  SUGGESTED FIX")
    print("=" * 70)
    print("""
    The issue is that your PROCESS MODEL uses activity names that don't
    match what's in the BPI Challenge 2017 historical data.

    You have two options:

    OPTION A: Update the process model to use the actual historical activity names

    OPTION B: Create a mapping from historical names to your model names
              For example: "W_Validate application" → "W_Validate application & A_Validating"

    Based on the analysis above, look at which historical activities should
    map to which process model activities.

    CRITICAL: For XOR gateways like O_Returned, the historical data shows it
    goes to "W_Validate application" 91% of the time, but your process model
    says it should go to "A_Incomplete" or "END". This is a fundamental mismatch!
    """)


if __name__ == "__main__":
    analyze_bpi_data("bpi2017.csv")