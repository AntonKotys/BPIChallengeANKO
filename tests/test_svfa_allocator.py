"""
Tests for SVFA (Score-based Value Function Approximation) Allocator.

Tests cover:
  1. Score computation with known weights
  2. Assignment selection (lowest score wins)
  3. Postponement when score exceeds threshold w7
  4. Feature computation (ActivityRank, ResourceRank, QueueLength)
  5. Permission-aware assignment
  6. Force-assign remaining tasks
  7. Shared resource_next_free tracking
  8. Processing stats lookup (activity-level and resource-activity level)
  9. ProbFin computation from process model
 10. Stats tracking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from datetime import datetime, timedelta

from ResourceAllocator.ResourceAllocatorAlgo import ResourceAllocatorAlgo
from ResourceAllocator.SVFAllocator import SVFAllocator


T0 = datetime(2016, 2, 1, 10, 0, 0)


def make_base_allocator(resources=None, permissions=None):
    pool = resources or ["R1", "R2", "R3"]
    return ResourceAllocatorAlgo(
        resource_pool=pool,
        permissions=permissions,
        availability_model=None,
        strategy="earliest_available",
        sim_start_time=T0,
    )


BASIC_STATS = {
    "ActivityA": {"mean": 100.0, "var": 25.0},
    "ActivityB": {"mean": 200.0, "var": 100.0},
    "ActivityC": {"mean": 50.0, "var": 10.0},
}

BASIC_PROB_FIN = {
    "ActivityA": 0.0,
    "ActivityB": 0.5,
    "ActivityC": 1.0,
}


class TestScoreComputation(unittest.TestCase):
    """Test that scores follow the formula from the paper."""

    def test_score_prefers_shorter_mean(self):
        """With only MeanAssignment weighted, the task with lower mean
        processing time should be assigned first."""
        base = make_base_allocator(resources=["R1", "R2"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityB", T0, timedelta(seconds=200), "Case_2")

        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(assignments[0]["activity"], "ActivityA")

    def test_prob_fin_reduces_score(self):
        """Activities with higher ProbFin should have lower scores."""
        base = make_base_allocator(resources=["R1", "R2"])
        weights = [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityC", T0, timedelta(seconds=50), "Case_2")

        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(assignments[0]["activity"], "ActivityC")

    def test_queue_length_reduces_score(self):
        """Activities with longer queues should be preferred (lower score)."""
        base = make_base_allocator(resources=["R1", "R2", "R3"])
        weights = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_2")
        svfa.submit_task("ActivityB", T0, timedelta(seconds=200), "Case_3")

        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 3)
        self.assertIn(assignments[0]["activity"], ["ActivityA"])


class TestPostponement(unittest.TestCase):
    """Test threshold w7 causes postponement."""

    def test_postpone_when_all_scores_above_threshold(self):
        base = make_base_allocator(resources=["R1"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 0)
        self.assertEqual(svfa.stats["postponements"], 1)
        self.assertEqual(len(svfa.pending_tasks), 1)

    def test_no_postpone_with_high_threshold(self):
        base = make_base_allocator(resources=["R1"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 1)


class TestAvailableResources(unittest.TestCase):
    """Test resource availability based on resource_next_free."""

    def test_busy_resources_excluded(self):
        base = make_base_allocator(resources=["R1", "R2"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        base.resource_next_free["R1"] = T0 + timedelta(hours=1)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_2")

        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0]["resource"], "R2")
        self.assertEqual(len(svfa.pending_tasks), 1)


class TestPermissions(unittest.TestCase):
    """Test permission-aware SVFA assignment."""

    def test_permissions_respected(self):
        perms = {"ActivityA": {"R1"}, "ActivityB": {"R2"}}
        base = make_base_allocator(resources=["R1", "R2"], permissions=perms)
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityB", T0, timedelta(seconds=200), "Case_2")

        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 2)
        res_map = {a["activity"]: a["resource"] for a in assignments}
        self.assertEqual(res_map["ActivityA"], "R1")
        self.assertEqual(res_map["ActivityB"], "R2")


class TestForceAssign(unittest.TestCase):
    """Test force-assign remaining tasks."""

    def test_force_assigns_all(self):
        base = make_base_allocator(resources=["R1"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityB", T0, timedelta(seconds=200), "Case_2")

        assignments = svfa.make_assignments(T0)
        self.assertEqual(len(assignments), 0)

        forced = svfa.force_assign_remaining(T0)
        self.assertEqual(len(forced), 2)
        self.assertEqual(len(svfa.pending_tasks), 0)


class TestResourceTracking(unittest.TestCase):
    """Test that assignments update shared resource_next_free."""

    def test_resource_next_free_updated(self):
        base = make_base_allocator(resources=["R1"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.make_assignments(T0)

        self.assertGreater(base.resource_next_free["R1"], T0)

    def test_sequential_assignments_respect_busy_time(self):
        base = make_base_allocator(resources=["R1"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        a1 = svfa.make_assignments(T0)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_2")
        a2 = svfa.make_assignments(T0)

        self.assertEqual(len(a2), 0)

        a2 = svfa.make_assignments(a1[0]["finish_time"])
        self.assertEqual(len(a2), 1)
        self.assertGreaterEqual(a2[0]["actual_start"], a1[0]["finish_time"])


class TestProcessingStatsLookup(unittest.TestCase):
    """Test processing stats lookup for compound activities."""

    def test_compound_activity_sums_parts(self):
        stats = {
            "PartA": {"mean": 100.0, "var": 25.0},
            "PartB": {"mean": 200.0, "var": 50.0},
        }
        base = make_base_allocator(resources=["R1"])
        svfa = SVFAllocator(base, processing_stats=stats)

        mean = svfa._get_mean("R1", "PartA & PartB")
        self.assertAlmostEqual(mean, 300.0)

        var = svfa._get_var("R1", "PartA & PartB")
        self.assertAlmostEqual(var, 75.0)

    def test_resource_activity_lookup_priority(self):
        stats = {
            "ActivityA": {"mean": 100.0, "var": 25.0},
            ("R1", "ActivityA"): {"mean": 80.0, "var": 20.0},
        }
        base = make_base_allocator(resources=["R1", "R2"])
        svfa = SVFAllocator(base, processing_stats=stats)

        self.assertAlmostEqual(svfa._get_mean("R1", "ActivityA"), 80.0)
        self.assertAlmostEqual(svfa._get_mean("R2", "ActivityA"), 100.0)


class TestProbFinFromModel(unittest.TestCase):
    """Test ProbFin computation from process model."""

    def test_end_in_successors(self):
        model = {
            "A": ["B", "END"],
            "B": ["C"],
            "C": ["END"],
        }
        prob_fin = SVFAllocator.compute_prob_fin_from_model(model)
        self.assertAlmostEqual(prob_fin["A"], 0.5)
        self.assertAlmostEqual(prob_fin["B"], 0.0)
        self.assertAlmostEqual(prob_fin["C"], 1.0)


class TestStatsTracking(unittest.TestCase):
    """Test SVFA stats are correctly tracked."""

    def test_stats(self):
        base = make_base_allocator(resources=["R1", "R2"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        svfa.submit_task("ActivityB", T0, timedelta(seconds=200), "Case_2")
        svfa.make_assignments(T0)

        stats = svfa.get_stats()
        self.assertEqual(stats["total_tasks_submitted"], 2)
        self.assertEqual(stats["assignments_made"], 2)
        self.assertGreaterEqual(stats["decision_steps"], 1)


class TestReturnFormat(unittest.TestCase):
    """Test assignment dict format."""

    REQUIRED_KEYS = {
        "resource", "planned_start", "actual_start",
        "finish_time", "delay_seconds", "was_delayed",
        "case_id", "activity",
    }

    def test_return_format(self):
        base = make_base_allocator(resources=["R1"])
        weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e9]
        svfa = SVFAllocator(base, weights, BASIC_STATS, BASIC_PROB_FIN)

        svfa.submit_task("ActivityA", T0, timedelta(seconds=100), "Case_1")
        results = svfa.make_assignments(T0)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(set(r.keys()), self.REQUIRED_KEYS)
        self.assertIsInstance(r["resource"], str)
        self.assertIsInstance(r["planned_start"], datetime)
        self.assertIsInstance(r["finish_time"], datetime)


class TestDistributionStats(unittest.TestCase):
    """Test compute_stats_from_distributions (requires distributions.json)."""

    def test_compute_stats(self):
        import os
        dist_path = os.path.join(
            os.path.dirname(__file__), "distributions.json"
        )
        if not os.path.exists(dist_path):
            self.skipTest("distributions.json not found")

        stats = SVFAllocator.compute_stats_from_distributions(dist_path)
        self.assertGreater(len(stats), 0)
        for act, s in stats.items():
            self.assertIn("mean", s)
            self.assertIn("var", s)
            self.assertGreaterEqual(s["mean"], 0)


if __name__ == "__main__":
    unittest.main()
