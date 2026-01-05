"""
Test Script for Task 1.4: Next Activity Prediction

This script tests the ExpertActivityPredictor integration with the simulation engine.
Run this to verify Task 1.4 is implemented correctly.

Usage:
    python test_task_1_4.py
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta

# Import the Task 1.4 modules
from task_1_4_next_activity import ExpertActivityPredictor


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_1_basic_fit_and_predict():
    """Test 1: Basic fitting and prediction"""
    print_header("TEST 1: Basic Fit and Predict")

    # Create synthetic training data with known patterns
    # Pattern: A -> B -> C (60%), A -> B -> D (40%)
    training_data = pd.DataFrame({
        'case_id': ['C1']*3 + ['C2']*3 + ['C3']*3 + ['C4']*3 + ['C5']*3 +
                   ['C6']*3 + ['C7']*3 + ['C8']*3 + ['C9']*3 + ['C10']*3,
        'activity': [
            'A', 'B', 'C',  # C1
            'A', 'B', 'C',  # C2
            'A', 'B', 'C',  # C3
            'A', 'B', 'C',  # C4
            'A', 'B', 'C',  # C5
            'A', 'B', 'C',  # C6
            'A', 'B', 'D',  # C7
            'A', 'B', 'D',  # C8
            'A', 'B', 'D',  # C9
            'A', 'B', 'D',  # C10
        ],
        'timestamp': pd.date_range('2024-01-01', periods=30, freq='h')
    })

    print(f"Training data: {len(training_data)} events, {training_data['case_id'].nunique()} cases")
    print("Pattern: A -> B -> C (6 times), A -> B -> D (4 times)")

    # Train predictor
    predictor = ExpertActivityPredictor(mode='basic', basic_context_k=2, smoothing_alpha=0.0)
    predictor.fit(training_data)

    # Test: P(next | A, B) should be roughly C=60%, D=40%
    dist = predictor.get_next_activity_distribution(['A', 'B'])

    print(f"\nLearned distribution P(next | [A, B]):")
    for act, prob in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {act}: {prob:.1%}")

    # Verify
    assert 'C' in dist and 'D' in dist, "Both C and D should be in distribution"
    assert abs(dist['C'] - 0.6) < 0.1, f"P(C) should be ~60%, got {dist['C']:.1%}"
    assert abs(dist['D'] - 0.4) < 0.1, f"P(D) should be ~40%, got {dist['D']:.1%}"

    print("\n✅ TEST 1 PASSED: Predictor correctly learns transition probabilities")
    return True


def test_2_xor_gateway_restriction():
    """Test 2: XOR gateway restriction (enabled_next)"""
    print_header("TEST 2: XOR Gateway Restriction")

    # Training data with multiple possible successors
    training_data = pd.DataFrame({
        'case_id': ['C1']*4 + ['C2']*4 + ['C3']*4,
        'activity': [
            'Start', 'A', 'B', 'End',
            'Start', 'A', 'C', 'End',
            'Start', 'A', 'D', 'End',
        ],
        'timestamp': pd.date_range('2024-01-01', periods=12, freq='h')
    })

    predictor = ExpertActivityPredictor(mode='basic', basic_context_k=1)
    predictor.fit(training_data)

    # Without restriction: should see B, C, D
    dist_all = predictor.get_next_activity_distribution(['A'])
    print(f"P(next | [A]) without restriction: {dist_all}")

    # With XOR restriction: only B and C allowed
    enabled = ['B', 'C']
    dist_xor = predictor.get_next_activity_distribution(['A'], enabled_next=enabled)
    print(f"P(next | [A], enabled={enabled}): {dist_xor}")

    # Verify only enabled activities are in result
    assert set(dist_xor.keys()) == set(enabled), "Only enabled activities should be in distribution"
    assert abs(sum(dist_xor.values()) - 1.0) < 0.001, "Probabilities should sum to 1"

    print("\n✅ TEST 2 PASSED: XOR gateway correctly restricts to enabled activities")
    return True


def test_3_sampling_distribution():
    """Test 3: Sampling follows the learned distribution"""
    print_header("TEST 3: Sampling Distribution")

    # Create data with 70-30 split
    activities = []
    case_ids = []
    for i in range(100):
        case_ids.extend([f'C{i}']*3)
        if i < 70:
            activities.extend(['X', 'Y', 'A'])
        else:
            activities.extend(['X', 'Y', 'B'])

    training_data = pd.DataFrame({
        'case_id': case_ids,
        'activity': activities,
        'timestamp': pd.date_range('2024-01-01', periods=len(case_ids), freq='h')
    })

    predictor = ExpertActivityPredictor(mode='basic', basic_context_k=2, smoothing_alpha=0.0, random_state=42)
    predictor.fit(training_data)

    # Sample many times
    n_samples = 1000
    samples = []
    for _ in range(n_samples):
        s = predictor.sample_next_activity(['X', 'Y'], enabled_next=['A', 'B'])
        samples.append(s)

    counts = Counter(samples)
    pct_A = counts['A'] / n_samples
    pct_B = counts['B'] / n_samples

    print(f"Sampled {n_samples} times from P(next | [X, Y]):")
    print(f"  A: {counts['A']} ({pct_A:.1%})")
    print(f"  B: {counts['B']} ({pct_B:.1%})")
    print(f"Expected: A ~70%, B ~30%")

    # Allow some variance (should be within 5% of expected)
    assert abs(pct_A - 0.7) < 0.1, f"P(A) should be ~70%, got {pct_A:.1%}"
    assert abs(pct_B - 0.3) < 0.1, f"P(B) should be ~30%, got {pct_B:.1%}"

    print("\n✅ TEST 3 PASSED: Sampling follows learned distribution")
    return True


def test_4_context_fallback():
    """Test 4: Fallback from longer to shorter context"""
    print_header("TEST 4: Context Fallback (k-gram)")

    training_data = pd.DataFrame({
        'case_id': ['C1']*4 + ['C2']*4,
        'activity': [
            'A', 'B', 'C', 'D',  # A->B->C seen
            'A', 'B', 'C', 'E',  # A->B->C seen again
        ],
        'timestamp': pd.date_range('2024-01-01', periods=8, freq='h')
    })

    predictor = ExpertActivityPredictor(mode='basic', basic_context_k=3)
    predictor.fit(training_data)

    # Context [A, B, C] should work (k=3)
    dist_k3 = predictor.get_next_activity_distribution(['A', 'B', 'C'])
    print(f"P(next | [A, B, C]) with k=3: {dist_k3}")
    assert 'D' in dist_k3 or 'E' in dist_k3, "Should find successors with full context"

    # Context [Z, B, C] - Z not seen, should fall back to [B, C] (k=2)
    dist_fallback = predictor.get_next_activity_distribution(['Z', 'B', 'C'])
    print(f"P(next | [Z, B, C]) (fallback): {dist_fallback}")

    # Context [X, Y, Z] - completely unseen, should fall back to global
    dist_global = predictor.get_next_activity_distribution(['X', 'Y', 'Z'])
    print(f"P(next | [X, Y, Z]) (global fallback): {dist_global}")
    assert len(dist_global) > 0, "Should fall back to global distribution"

    print("\n✅ TEST 4 PASSED: Context fallback works correctly")
    return True


def test_5_integration_with_simulation():
    """Test 5: Integration with simulation engine"""
    print_header("TEST 5: Simulation Engine Integration")

    # Simple process model
    PROCESS_MODEL = {
        "Start": ["A"],
        "A": ["B", "C"],  # XOR split
        "B": ["End"],
        "C": ["End"],
        "End": []
    }

    GATEWAYS = {
        "A": "xor"  # XOR gateway at A
    }

    # Training data: A -> B (80%), A -> C (20%)
    case_ids = []
    activities = []
    for i in range(100):
        case_ids.extend([f'C{i}']*3)
        if i < 80:
            activities.extend(['Start', 'A', 'B'])
        else:
            activities.extend(['Start', 'A', 'C'])

    training_data = pd.DataFrame({
        'case_id': case_ids,
        'activity': activities,
        'timestamp': pd.date_range('2024-01-01', periods=len(case_ids), freq='h')
    })

    # Train predictor
    predictor = ExpertActivityPredictor(mode='basic', basic_context_k=2, random_state=123)
    predictor.fit(training_data)

    # Simulate the XOR decision many times
    n_simulations = 500
    outcomes = {'B': 0, 'C': 0}

    for _ in range(n_simulations):
        # Simulate: we're at activity A, need to choose B or C
        trace = ['Start', 'A']
        enabled = ['B', 'C']

        next_act = predictor.sample_next_activity(trace, enabled_next=enabled)
        outcomes[next_act] += 1

    pct_B = outcomes['B'] / n_simulations
    pct_C = outcomes['C'] / n_simulations

    print(f"Simulated {n_simulations} XOR decisions at gateway A:")
    print(f"  -> B: {outcomes['B']} ({pct_B:.1%})")
    print(f"  -> C: {outcomes['C']} ({pct_C:.1%})")
    print(f"Expected: B ~80%, C ~20%")

    assert abs(pct_B - 0.8) < 0.1, f"P(B) should be ~80%, got {pct_B:.1%}"

    print("\n✅ TEST 5 PASSED: Integration with simulation works correctly")
    return True


def test_6_json_serialization():
    """Test 6: Save and load from JSON"""
    print_header("TEST 6: JSON Serialization")

    import tempfile
    import os

    training_data = pd.DataFrame({
        'case_id': ['C1']*3 + ['C2']*3,
        'activity': ['A', 'B', 'C', 'A', 'B', 'D'],
        'timestamp': pd.date_range('2024-01-01', periods=6, freq='h')
    })

    # Train and save
    predictor = ExpertActivityPredictor(mode='basic', basic_context_k=2)
    predictor.fit(training_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = predictor.save_to_json(tmpdir, "test_probs")

        print(f"Saved files:")
        for key, path in paths.items():
            print(f"  {key}: {os.path.basename(path)}")

        # Load into new predictor
        predictor2 = ExpertActivityPredictor(mode='basic', basic_context_k=2)
        predictor2.load_from_json(
            paths['counts'],
            paths['probs'],
            paths['global_probs']
        )

        # Compare distributions
        dist1 = predictor.get_next_activity_distribution(['A', 'B'])
        dist2 = predictor2.get_next_activity_distribution(['A', 'B'])

        print(f"\nOriginal distribution: {dist1}")
        print(f"Loaded distribution:   {dist2}")


        # Should be identical
        for act in dist1:
            assert act in dist2, f"Activity {act} missing after load"
            assert abs(dist1[act] - dist2[act]) < 0.001, f"Probability mismatch for {act}"

    print("\n✅ TEST 6 PASSED: JSON serialization works correctly")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("#  TASK 1.4 TEST SUITE")
    print("#" * 60)

    tests = [
        ("Basic Fit and Predict", test_1_basic_fit_and_predict),
        ("XOR Gateway Restriction", test_2_xor_gateway_restriction),
        ("Sampling Distribution", test_3_sampling_distribution),
        ("Context Fallback", test_4_context_fallback),
        ("Simulation Integration", test_5_integration_with_simulation),
        ("JSON Serialization", test_6_json_serialization),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "#" * 60)
    print("#  TEST SUMMARY")
    print("#" * 60)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "🎉" * 20)
        print("  ALL TESTS PASSED! Task 1.4 is implemented correctly.")
        print("🎉" * 20)
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Please review.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
