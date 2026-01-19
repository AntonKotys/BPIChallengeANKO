"""
Task 1.4 Verification Script

Verifies that the predictor:
1. Uses TRACE HISTORY (k-gram context) to influence predictions
2. ONLY returns valid activities from enabled_next
3. Simulation generates valid traces (handling OR gateway parallelism)
"""

import pandas as pd
from collections import Counter, defaultdict

from task_1_4_next_activity import ExpertActivityPredictor

# PROCESS_MODEL for validation
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

# Activities that can run in parallel (from OR gateway)
PARALLEL_ACTIVITIES = {
    "W_Validate application & A_Validating", "O_Returned", "A_Incomplete", "A_Validating",
    "O_Create Offer & O_Created", "O_Sent (mail and online)", "W_Call after offers & A_Complete",
    "A_Cancelled & O_Cancelled"
}


def test_trace_history_influence():
    """Test that different trace histories produce different predictions."""
    print("\n" + "=" * 70)
    print("TEST 1: Trace History Influence")
    print("=" * 70)

    # Create training data with pattern:
    # Pattern 1: A_Create Application alone → A_Submitted (80%)
    # Pattern 2: X → A_Create Application → W_Complete... (80%)

    training_data = []

    # Pattern 1a: A_Create Application → A_Submitted (80 cases)
    for i in range(80):
        training_data.extend([
            {"case_id": f"p1a_{i}", "activity": "A_Create Application", "timestamp": "2016-01-01 10:00:00"},
            {"case_id": f"p1a_{i}", "activity": "A_Submitted", "timestamp": "2016-01-01 10:01:00"},
        ])

    # Pattern 1b: A_Create Application → W_Complete... (20 cases)
    for i in range(20):
        training_data.extend([
            {"case_id": f"p1b_{i}", "activity": "A_Create Application", "timestamp": "2016-01-01 10:00:00"},
            {"case_id": f"p1b_{i}", "activity": "W_Complete application & A_Concept", "timestamp": "2016-01-01 10:01:00"},
        ])

    # Pattern 2a: X → A_Create Application → W_Complete... (80 cases)
    for i in range(80):
        training_data.extend([
            {"case_id": f"p2a_{i}", "activity": "X", "timestamp": "2016-01-01 10:00:00"},
            {"case_id": f"p2a_{i}", "activity": "A_Create Application", "timestamp": "2016-01-01 10:01:00"},
            {"case_id": f"p2a_{i}", "activity": "W_Complete application & A_Concept", "timestamp": "2016-01-01 10:02:00"},
        ])

    # Pattern 2b: X → A_Create Application → A_Submitted (20 cases)
    for i in range(20):
        training_data.extend([
            {"case_id": f"p2b_{i}", "activity": "X", "timestamp": "2016-01-01 10:00:00"},
            {"case_id": f"p2b_{i}", "activity": "A_Create Application", "timestamp": "2016-01-01 10:01:00"},
            {"case_id": f"p2b_{i}", "activity": "A_Submitted", "timestamp": "2016-01-01 10:02:00"},
        ])

    df = pd.DataFrame(training_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    predictor = ExpertActivityPredictor(mode="basic", basic_context_k=2)
    predictor.fit(df)

    enabled = PROCESS_MODEL["A_Create Application"]

    # Test trace 1: Just A_Create Application
    trace1 = ["A_Create Application"]
    dist1 = predictor.get_next_activity_distribution(trace1, enabled_next=enabled)

    print(f"\nTrace 1: {trace1}")
    print(f"Distribution:")
    for act, prob in sorted(dist1.items(), key=lambda x: -x[1]):
        print(f"  {act}: {prob:.1%}")

    # Test trace 2: X → A_Create Application
    trace2 = ["X", "A_Create Application"]
    dist2 = predictor.get_next_activity_distribution(trace2, enabled_next=enabled)

    print(f"\nTrace 2: {trace2}")
    print(f"Distribution:")
    for act, prob in sorted(dist2.items(), key=lambda x: -x[1]):
        print(f"  {act}: {prob:.1%}")

    # Verify different traces produce different results
    prob1 = dist1.get("A_Submitted", 0)
    prob2 = dist2.get("A_Submitted", 0)
    diff = abs(prob1 - prob2)

    print("\n✅ VERIFICATION:")
    if diff > 0.2:
        print(f"  ✓ Different traces produce different predictions!")
        print(f"  ✓ Difference: {diff:.1%}")
        return True
    else:
        print(f"  ✗ Predictions too similar (diff={diff:.1%})")
        return False


def test_enabled_next_constraint():
    """Test that predictor only returns activities from enabled_next."""
    print("\n" + "=" * 70)
    print("TEST 2: enabled_next Constraint")
    print("=" * 70)

    # Train on A→B→C→D→E transitions
    training_data = []
    for i in range(100):
        for j, act in enumerate(["A", "B", "C", "D", "E"]):
            training_data.append({
                "case_id": f"case_{i}",
                "activity": act,
                "timestamp": pd.Timestamp("2016-01-01") + pd.Timedelta(minutes=i*5 + j)
            })

    df = pd.DataFrame(training_data)

    predictor = ExpertActivityPredictor(mode="basic", basic_context_k=2)
    predictor.fit(df)

    # Test: only allow B and C
    enabled = ["B", "C"]
    trace = ["A"]

    print(f"\nTrained on: A, B, C, D, E")
    print(f"Trace: {trace}")
    print(f"enabled_next: {enabled}")

    # Get distribution
    dist = predictor.get_next_activity_distribution(trace, enabled_next=enabled)
    print(f"\nDistribution (should ONLY have B, C):")
    for act, prob in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {act}: {prob:.1%}")

    # Sample many times
    samples = Counter()
    for _ in range(1000):
        s = predictor.sample_next_activity(trace, enabled_next=enabled)
        samples[s] += 1

    print(f"\nSampled 1000 times:")
    for act, count in samples.most_common():
        print(f"  {act}: {count} ({count/10:.1f}%)")

    # Verify
    invalid = set(samples.keys()) - set(enabled)
    print("\n✅ VERIFICATION:")
    if not invalid:
        print("  ✓ All samples from enabled_next!")
        return True
    else:
        print(f"  ✗ Invalid activities sampled: {invalid}")
        return False


def test_simulation_traces():
    """Verify simulation output has valid transitions (handling parallel branches)."""
    print("\n" + "=" * 70)
    print("TEST 3: Simulation Output Validation")
    print("=" * 70)

    try:
        df = pd.read_csv("sim_predicted.csv")
    except FileNotFoundError:
        print("\n⚠️  sim_predicted.csv not found. Run simulation first.")
        return None

    # Standardize columns
    if 'concept:name' in df.columns:
        df = df.rename(columns={'concept:name': 'activity', 'case:concept:name': 'case_id'})

    has_branch = 'branch_id' in df.columns

    print(f"\nAnalyzing {df['case_id'].nunique()} cases...")
    print(f"Branch tracking: {'Yes' if has_branch else 'No'}")

    invalid = []
    parallel = []
    valid = 0

    for case_id, group in df.groupby('case_id'):
        if 'timestamp' in group.columns or 'time:timestamp' in group.columns:
            ts_col = 'timestamp' if 'timestamp' in group.columns else 'time:timestamp'
            group = group.sort_values(ts_col)

        activities = group['activity'].tolist()
        branches = group['branch_id'].tolist() if has_branch else [None] * len(activities)

        for i in range(len(activities) - 1):
            curr = activities[i]
            nxt = activities[i + 1]

            if nxt == "END":
                valid += 1
                continue

            valid_next = PROCESS_MODEL.get(curr, [])

            if nxt in valid_next:
                valid += 1
                continue

            # Check if parallel (different branches or both in parallel set)
            if has_branch and branches[i] != branches[i + 1]:
                parallel.append((case_id, curr, nxt))
                continue

            if curr in PARALLEL_ACTIVITIES and nxt in PARALLEL_ACTIVITIES:
                parallel.append((case_id, curr, nxt))
                continue

            # Flag as invalid only if curr has defined outgoing
            if valid_next:
                invalid.append({'case': case_id, 'from': curr, 'to': nxt, 'valid': valid_next})

    print(f"\n  Valid sequential: {valid}")
    print(f"  Parallel interleave: {len(parallel)}")

    print("\n✅ VERIFICATION:")
    if invalid:
        print(f"  ✗ Found {len(invalid)} invalid transitions:")
        for inv in invalid[:5]:
            print(f"    {inv['from']} → {inv['to']} (valid: {inv['valid'][:2]}...)")
        return False
    else:
        print("  ✓ All transitions valid!")
        print("  ✓ Parallel branches handled correctly!")
        return True


def main():
    """Run all verification tests."""
    print("\n" + "#" * 70)
    print("#  TASK 1.4 VERIFICATION")
    print("#" * 70)

    results = [
        ("Trace History Influence", test_trace_history_influence()),
        ("enabled_next Constraint", test_enabled_next_constraint()),
        ("Simulation Traces Valid", test_simulation_traces()),
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        if passed is None:
            status = "⚠️  SKIPPED"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
            all_passed = False
        print(f"  {name}: {status}")

    if all_passed:
        print("\n🎉 Task 1.4 verification complete!")
    else:
        print("\n⚠️  Some tests failed.")


if __name__ == "__main__":
    main()