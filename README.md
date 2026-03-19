# BPI Challenge ANKO — Business Process Simulation, Prediction & Optimization

Course assignment for **Business Process Prediction, Simulation, and Optimization** at TU Munich.
The project builds a simulation model from the [BPIC 2017](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) event log (loan application process).

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

---

## Project Structure

```
.
├── simulation_engine_core_final_version.py   # Main simulation engine (latest)
├── simulation_engine_core_V1_8.py            # Previous stable engine version
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
├── bpianko9.0.bpmn                           # BPMN process model
├── bpi2017.csv                               # BPIC 2017 event log (not in git)
├── distributions.json                        # Fitted processing time distributions
├── ml_metrics.json                           # ML model evaluation metrics
├── svfa_weights_optimized.json               # Trained SVFA weights
│
├── sim_outputs/                              # Simulation output logs and plots
│
├── analysis/                                 # Standalone analysis & exploration scripts
│
├── archive/                                  # Old versions (kept for reference)
│   └── engine_core_versions/                 #     Engine core V1.0 through V1.9
│
└── requirements.txt                          # Python dependencies
```

---

## Simulation Engine

The simulation engine (`simulation_engine_core_final_version.py`) spawns process instances, enforces control-flow constraints from the BPMN model, tracks resource availability, and logs events in CSV/XES format. Key design decisions:

- **Hard-coded process model** — the BPMN model contains clustered activities and special splits that pm4py cannot reproduce automatically.
- **Token-counting mechanism** — lightweight alternative to full Petri net semantics that supports splits and parallel branches.
- **Versioned iterations** — each major change created a new file to avoid breaking teammates' work. Old versions are in `archive/engine_core_versions/`.

---

## Task Breakdown

### 1.3 Processing Times
Distribution fitting (exponential, gamma, Weibull, lognormal) selected via AIC. Advanced: XGBoost regression with temporal context.

### 1.4 Next Activities
- **Basic:** k-gram model (k=2) with Laplace smoothing, constrained by the process model.
- **Advanced:** Token replay (pm4py) identifies 7 XOR decision points. RandomForest (500 trees) trained on 18 features achieves 81.4% accuracy.

### 1.5 Resource Availability
Stochastic shift sampling from historical data with resource presence filtering.

### 1.6 Resource Permissions
Historical permission extraction with advanced role discovery via clustering.

### 2.1 Resource Allocation
Five strategies: random, round-robin, earliest-available, K-Batching (LPT/Hungarian), SVFA (Bayesian-optimized scoring).

### 2.2 Evaluation
Scaling experiments (100/500/1000 cases) measuring cycle time, activity delay, resource occupation, and Jain fairness index.

### 2.3 Firing Employees
Brute-force search for the optimal pair of employees to remove while minimizing performance degradation.

---

## Key Dependencies

- **pm4py** — process mining, BPMN/Petri net conversion, token replay
- **scikit-learn** — RandomForest, GradientBoosting, LabelEncoder
- **scipy** — Hungarian algorithm (linear_sum_assignment)
- **scikit-optimize** — Bayesian optimization for SVFA weights
- **pandas / numpy** — data processing

---

## References

1. van Dongen, B. (2017). BPI Challenge 2017. 4TU.ResearchData. https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
2. Middelhuis, J. et al. (2025). Learning policies for resource allocation in business processes. *Inf. Syst.*, 128, art. 102492.
3. Zeng, D. D. & Zhao, J. L. (2005). Effective role resolution in workflow management. *INFORMS J. Comput.*, 17(3), 374–387.
