"""
Bayesian Optimization trainer for SVFA weights.

Uses scikit-optimize (skopt) to find weights [w1..w7] that minimize the
average cycle time when running the simulation with the SVFAllocator.

Usage (standalone):
    python -m ResourceAllocator.train_svfa --n_calls 50 --n_sims 5

The script saves the best weights to svfa_weights.json.

Based on Section 5.2.2 of Middelhuis et al. (2025):
    - Search space: 0 <= w_i <= 100 for i in {1, ..., 7}
    - Bayesian optimization via scikit-optimize (gp_minimize)
    - Objective: mean cycle time over multiple simulation runs
"""

import argparse
import json
import time
import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


def compute_mean_cycle_time(engine) -> float:
    """
    Compute mean cycle time from a completed simulation engine.
    Cycle time = time of last event - time of first event for each case.
    """
    import pandas as pd

    if not engine.log_rows:
        return float("inf")

    df = pd.DataFrame(engine.log_rows)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

    cycle_times = []
    for case_id, grp in df.groupby("case:concept:name"):
        t_start = grp["time:timestamp"].min()
        t_end = grp["time:timestamp"].max()
        ct = (t_end - t_start).total_seconds()
        cycle_times.append(ct)

    return float(np.mean(cycle_times)) if cycle_times else float("inf")


def run_single_simulation(weights, n_cases=50):
    """
    Run one simulation with the given SVFA weights and return the mean
    cycle time.  Imports are local to avoid circular dependencies.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from simulation_engine_core_V1_8 import (
        run_simulation, train_predictor_from_csv, NUM_CASES
    )

    predictor = train_predictor_from_csv("bpi2017.csv", mode="basic", context_k=2)

    engine = run_simulation(
        predictor=predictor,
        n_cases=n_cases,
        output_prefix="/dev/null",
        allocation_strategy="svfa",
        svfa_weights=weights,
    )
    return compute_mean_cycle_time(engine)


def objective(weights_list, n_sims=5, n_cases=50):
    """
    Objective function for Bayesian optimization.
    Runs n_sims simulations with the given weights and returns the mean
    cycle time averaged over all runs.
    """
    results = []
    for _ in range(n_sims):
        try:
            ct = run_single_simulation(list(weights_list), n_cases=n_cases)
            results.append(ct)
        except Exception as e:
            print(f"  Simulation failed: {e}")
            results.append(float("inf"))

    mean_ct = float(np.mean(results))
    print(f"  Weights: {[f'{w:.3f}' for w in weights_list]}  "
          f"-> Mean CT: {mean_ct:.1f}s")
    return mean_ct


def train_weights(
    n_calls: int = 50,
    n_random_starts: int = 10,
    n_sims: int = 5,
    n_cases: int = 50,
    output_path: str = "svfa_weights.json",
):
    """
    Run Bayesian optimization to find optimal SVFA weights.

    Parameters
    ----------
    n_calls : int
        Total number of Bayesian optimization evaluations.
    n_random_starts : int
        Number of random initial evaluations.
    n_sims : int
        Number of simulation runs per weight evaluation.
    n_cases : int
        Number of cases per simulation run.
    output_path : str
        Path to save the best weights as JSON.

    Returns
    -------
    list[float]
        Best weights found [w1 .. w7].
    """
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "scikit-optimize is required for weight training. "
            "Install it with: pip install scikit-optimize"
        )

    space = [
        Real(0.0, 100.0, name="w1_mean_assignment"),
        Real(0.0, 100.0, name="w2_var_assignment"),
        Real(0.0, 100.0, name="w3_activity_rank"),
        Real(0.0, 100.0, name="w4_resource_rank"),
        Real(0.0, 100.0, name="w5_prob_fin"),
        Real(0.0, 100.0, name="w6_queue_length"),
        Real(0.0, 100.0, name="w7_threshold"),
    ]

    def obj_fn(w):
        return objective(w, n_sims=n_sims, n_cases=n_cases)

    print(f"Starting Bayesian optimization: {n_calls} calls, "
          f"{n_sims} sims/call, {n_cases} cases/sim")
    t0 = time.time()

    result = gp_minimize(
        obj_fn,
        space,
        acq_func="EI",
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        noise=0.1 ** 2,
        random_state=42,
    )

    elapsed = time.time() - t0
    best_weights = list(result.x)
    best_ct = float(result.fun)

    print(f"\nOptimization complete in {elapsed:.0f}s")
    print(f"Best weights: {[f'{w:.4f}' for w in best_weights]}")
    print(f"Best mean cycle time: {best_ct:.1f}s")

    output = {
        "weights": best_weights,
        "best_cycle_time": best_ct,
        "n_calls": n_calls,
        "n_sims": n_sims,
        "n_cases": n_cases,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Weights saved to {output_path}")

    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SVFA weights via Bayesian optimization"
    )
    parser.add_argument("--n_calls", type=int, default=50)
    parser.add_argument("--n_random_starts", type=int, default=10)
    parser.add_argument("--n_sims", type=int, default=5)
    parser.add_argument("--n_cases", type=int, default=50)
    parser.add_argument("--output", type=str, default="svfa_weights.json")
    args = parser.parse_args()

    train_weights(
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts,
        n_sims=args.n_sims,
        n_cases=args.n_cases,
        output_path=args.output,
    )
