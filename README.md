# BPI Challenge ANKO — Business Process Simulation, Prediction & Optimization

Course assignment for **Business Process Prediction, Simulation, and Optimization** at TU Munich.
The project builds a discrete-event simulation model from the [BPIC 2017](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) event log, which captures a loan application process at a Dutch financial institution (31,509 traces, 1.2M events). The model reproduces the real-world process — from case arrival through resource allocation to event logging — and is used to evaluate and optimize different resource allocation strategies.

**Team:** Anton Kotys, Emi Mano, Lukas Vester, Saifullozhon Tanikulov

---

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

Run a simulation (500 cases, earliest-available allocation):

```bash
python simulation_engine_core_final_version.py
```

Output event logs (CSV + XES) are written to `sim_outputs/`.

---

## Project Structure

```
.
├── simulation_engine_core_final_version.py   # Main simulation engine (latest)
├── simulation_engine_core_V1_8.py            # Previous stable engine version
├── simulation_engine_core_V1.7.1py           # Hotfix engine version
│
├── task_1_3/                                 # 1.3 Processing Times
│   ├── task_1_3_processing_times.py          #     Distribution fitting (AIC selection)
│   └── task_1_3_processing_times_ml.py       #     XGBoost regression with temporal context
│
├── task_1_4/                                 # 1.4 Next Activity Prediction
│   ├── task_1_4_next_activity.py             #     k-gram (basic) + RandomForest (advanced)
│   ├── task_1_4_token_replay.py              #     Token replay for decision point extraction
│   ├── task_1_4_metrics.py                   #     Evaluation metrics
│   ├── evaluate_task_1_4.py                  #     Base evaluation script
│   ├── evaluate_task_1_4_mapped.py           #     Evaluation with activity mapping
│   └── evaluate_task_1_4_benchmark.py        #     Model benchmarking (RF, HistGB)
│
├── task_1_5/                                 # 1.5 Resource Availability
│   ├── task_1_5_resource_availability.py     #     Basic availability model
│   └── task_1_5_rolling_stochastic_availability.py  # Stochastic shift model
│
├── task_1_6/                                 # 1.6 Resource Permissions
│   ├── 1.6_grid_search.py                    #     Role-discovery hyperparameter search
│   └── 1.6_test.py                           #     Permission validation tests
│
├── ResourceAllocator/                        # 2.1 Resource Allocation Strategies
│   ├── ResourceAllocatorAlgo.py              #     Base allocator (random, round-robin, earliest)
│   ├── BatchAllocator.py                     #     K-Batching (Zeng & Zhao 2005)
│   ├── SVFAllocator.py                       #     SVFA scoring (Middelhuis et al. 2025)
│   └── train_svfa.py                         #     Bayesian optimization for SVFA weights
│
├── DynamicSpawnRates/                        # 1.2 Instance Arrival Rates
│   └── DynamicArrivalModel.py                #     Context-aware dynamic spawn rates
│
├── task_2_2/                                 # 2.2–2.3 Evaluation & Optimization
│   ├── task_2_2_evaluation.py                #     Evaluation metrics
│   ├── task_2_2_scaling_experiment.py        #     Scaling experiment (100/500/1000 cases)
│   └── task_2_2_firing_employees.py          #     Employee firing optimization
│
├── tests/                                    # Test Scripts
│   ├── test_task_1_4.py                      #     Tests for next activity predictor
│   ├── test_svfa_allocator.py                #     Tests for SVFA allocator
│   └── verify_task_1_4.py                    #     Verification of predictor constraints
│
├── analysis/                                 # Standalone analysis & exploration scripts
│   ├── *.py                                  #     Process mining, clustering, diagnostics
│   └── plots/                                #     Generated plots (heuristic nets, clusters, etc.)
│
├── archive/                                  # Old versions (kept for reference)
│   └── engine_core_versions/                 #     Engine core V1.0 through V1.9
│
├── sim_outputs/                              # Simulation output logs and plots
│   ├── scaling_experiment_outputs/           #     V1 scaling experiment results
│   ├── scaling_experiment_outputs_v2/        #     V2 with SVFA optimization sweeps
│   └── scaling_experiment_outputs_v3/        #     V3 final results
│
├── bpianko9.0.bpmn                           # BPMN process model
├── bpi2017.csv                               # BPIC 2017 event log (not in git)
├── BPI_Challenge_2017.xes.gz                 # Original XES event log (compressed)
├── distributions.json                        # Fitted processing time distributions
├── ml_metrics.json                           # ML model evaluation metrics
├── svfa_weights_optimized.json               # Trained SVFA weights
├── gateway_comparison_report.txt             # XOR gateway prediction comparison report
├── SimpleMetrics.ipynb                       # Jupyter notebook with basic metrics
│
└── requirements.txt                          # Python dependencies
```

---

## Simulation Engine

The simulation engine (`simulation_engine_core_final_version.py`) is the central component of the project. It spawns new process instances, enforces control-flow constraints prescribed by the BPMN model, tracks resource availabilities, and logs the simulation in CSV/XES format.

### Core Data Structure

Each simulated event is a **SimEvent** containing a timestamp, case identifier, activity name, allocated resource, and delay-related fields (`planned_start`, `actual_start`, `delay_seconds`, `was_delayed`). Events are ordered chronologically in a heap-based priority queue.

### Routing and Control-Flow

Two data structures drive the control flow:
- **PROCESS_MODEL** — defines allowed successor activities for each node.
- **GATEWAYS** — specifies split semantics (XOR / OR) at decision points.

At **XOR gateways**, if an `ExpertActivityPredictor` is available, the engine samples the next activity from learned branching probabilities (see Task 1.4). Otherwise, it falls back to uniform random choice.

At the **OR gateway** after *"W_Call after offers & A_Complete"*, cancellation is modeled as an exclusive outcome with a historically derived probability, while non-cancellation branches are handled as an inclusive OR. Offer creation is capped at 3 per case (98th percentile of the original log).

### Token-Based Replay

Each case maintains a `case_context` with an event counter, an end flag, and an `open_tokens` count. Executing an event consumes one token; scheduling successors produces new tokens. A case finishes when its token count drops to zero.

### Versioned Iterations

To avoid merge conflicts and accidental breakage, each major iteration of the engine was saved as a new file. Old versions are preserved in `archive/engine_core_versions/`.

---

## Task Breakdown

### 1.1 Simulation Engine Core
See the [Simulation Engine](#simulation-engine) section above.

### 1.2 Instance Arrival Rates

From the filtered BPIC-17 log, inter-arrival times were computed for 31,416 applications. The mean inter-arrival time is ~1,089 s (18 min) with high variance (std = 6,025 s), consistent with a Poisson process.

- **Static model:** a single exponential rate lambda = 1/1089.18 ~ 0.000918 cases/s.
- **Dynamic model** (`DynamicSpawnRates/DynamicArrivalModel.py`): separate lambdas per (weekday, 6-hour time bin), selected at runtime based on the current simulation timestamp. Inspired by temporal-context arrival modeling from the literature.

### 1.3 Processing Times

- **Distribution fitting** (`task_1_3/task_1_3_processing_times.py`): for each activity, four candidate distributions (exponential, gamma, Weibull, lognormal) are fitted and the best is selected via AIC. Results are stored in `distributions.json`.
- **ML point estimation** (`task_1_3/task_1_3_processing_times_ml.py`): an XGBoost regression model predicts processing times using temporal context (hour of day, day of week). Metrics (MAE, RMSE) are logged to `ml_metrics.json`.

### 1.4 Next Activities

At each XOR gateway, the engine queries an `ExpertActivityPredictor` to determine the branch.

- **Basic predictor:** a k-gram model (k=2) that learns transition probabilities from valid process-model edges only. Laplace smoothing (alpha=1.0) is applied. A fallback hierarchy (per-activity table -> global table -> uniform) ensures a prediction is always available. The resulting probabilities are intersected with valid successors and renormalized, guaranteeing no impossible transitions.
- **Advanced predictor:** decision points are identified via token replay (pm4py): the BPMN model is converted to a Petri net, and each place with >1 outgoing arc is a decision place. Replaying 31,509 traces yields 706,519 decision instances across **7 XOR gateways**. A RandomForest (500 trees, max depth 25) is trained on an 18-dimensional feature vector: 5-activity n-gram history, 3 temporal features (hour, weekday, elapsed seconds), and 11 case-level attributes (RequestedAmount, LoanGoal, ApplicationType, loop count, trace length, cumulative offers, CreditScore, OfferedAmount, mean inter-event time, distinct resources). Training uses 80/20 group-shuffle split grouped by case ID.
- **Evaluation:** the basic k-gram achieves 0.5% distribution error vs. historical branching proportions (random: 37.5%). Per-instance accuracy: basic 51.1%, advanced RandomForest **81.4%** across all 7 gateways.

### 1.5 Resource Availability

A two-component model:

1. **Stochastic daily shifts:** historical start/end times are analyzed per resource; the 10th and 90th quantiles define the range from which daily shift windows are sampled.
2. **Resource presence filtering:** resources are eligible only if they historically show >= 244 min average availability/month *and* >= 5 active days/month (both at the 20th quantile). System resources (non-human) are always available.

### 1.6 Resource Permissions

- **Basic:** a permission dictionary `activity -> {allowed resources}` is extracted from the historical log. Clustered activities (connected with "&") require a resource to have handled *all* sub-activities.
- **Advanced role discovery:** resources are clustered into roles based on activity profiles. A threshold parameter filters out anomalies (e.g., cross-department help). Historical permissions serve as a fallback baseline; role-based permissions expand upon it. Grid search (`1.6_grid_search.py`) over (n_roles, threshold) selected n_roles=10, threshold=0.1, yielding 3,001 permission pairs (~24% increase over the 2,414 baseline).

### 1.7 Resource Allocation — Base

A three-step process: (1) check who is available now, (2) check who is permitted for the activity, (3) choose one valid candidate. If no resource satisfies both constraints, a fallback to the broader available set prevents deadlock.

### 2.1 Resource Allocation Strategies

Five strategies are implemented in `ResourceAllocator/`:

| Strategy | Description |
|---|---|
| **Random** | Uniform random selection among eligible resources |
| **Round-robin** | Cyclic assignment for even workload distribution |
| **Earliest-available** | Assign to the resource that can start soonest |
| **K-Batching** | Accumulate tasks, then solve the batch via LPT (4/3-approximation) or Hungarian algorithm. Timeout flush at 3600 s to prevent starvation |
| **SVFA** | Score-based Value Function Approximation: a learned linear combination of 6 features (mean/var assignment time, activity/resource rank, finish probability, queue length) plus a postponement threshold. All 7 weights optimized via Bayesian optimization (`skopt.gp_minimize`) to minimize mean cycle time |

### 2.2 Evaluation

Metrics: average cycle time (days), average activity delay (hours), average resource occupation (%), weighted resource fairness (Jain index), share of delayed human activities (%).

Scaling experiments at 100 / 500 / 1000 cases show:
- **Earliest-available** consistently achieves the lowest cycle time and delay.
- **K-Batch** and **SVFA** achieve the highest fairness but at the cost of significantly higher delays and cycle times.
- Resource occupation increases with case count across all strategies; fairness generally improves under higher load.

### 2.3 Firing Employees

Under the earliest-available strategy, the two employees with the lowest task count (User_85 and User_103) were removed. Result: cycle time increased by only ~3 hours (21.89 -> 22.01 days), while fairness improved (0.621 -> 0.637) and delayed share decreased (52.2% -> 44.6%).

---

## Key Dependencies

- **pm4py** — process mining, BPMN/Petri net conversion, token replay
- **scikit-learn** — RandomForest, GradientBoosting, LabelEncoder
- **scipy** — Hungarian algorithm (`linear_sum_assignment`)
- **scikit-optimize** — Bayesian optimization for SVFA weights
- **xgboost** — processing time regression
- **pandas / numpy** — data processing
- **rustxes** (optional) — fast XES parsing

All randomness uses a global seed of **42** for reproducibility.

---

## Contributions

- **Anton Kotys** — simulation engine core, resource availabilities (basic + advanced), evaluation metrics (basic + advanced, scaling performance), employee firing analysis.
- **Emi Mano** — instance spawn rates, random resource allocation, base allocation heuristics, dynamic spawn rates.
- **Lukas Vester** — processing times, resource permissions, test cases, advanced role discovery.
- **Saifullozhon Tanikulov** — next activities, batch resource allocation, token replay for next activity prediction, SVFA.

---

## References

1. van Dongen, B. (2017). BPI Challenge 2017. 4TU.ResearchData. https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
2. Middelhuis, J. et al. (2025). Learning policies for resource allocation in business processes. *Inf. Syst.*, 128, art. 102492.
3. Zeng, D. D. & Zhao, J. L. (2005). Effective role resolution in workflow management. *INFORMS J. Comput.*, 17(3), 374–387.
