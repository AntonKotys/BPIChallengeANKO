# Report Sections — Samat Tanikulov

---

## 1.4 Next Activities

The next activity predictor is responsible for determining which branch a case follows at each XOR gateway. The implementation can be found in the task_1_4_next_activity.py file. Two versions were developed: a basic k-gram predictor with process model constraints, and an advanced approach that combines token replay with an enriched machine learning model.

### Basic Task

The basic predictor pairs a k-gram language model with a structural filter derived from the BPMN model. This means that the predictor can only output transitions that are valid according to the process model, which prevents impossible activity sequences from being generated during simulation.

The training procedure works as follows:
1. We iterate over all cases in the filtered BPIC-17 event log and extract consecutive activity pairs.
2. For each pair (A, B), the predictor checks whether B is in the set of allowed successors of A according to the process model. If not, the pair is discarded.
3. Valid pairs are recorded under k-gram contexts of length 1 through k. We set k = 2 throughout.
4. The raw counts are converted to probability distributions using Laplace smoothing with pseudocount α = 1.0:

P(next | current, context) = (count(next) + α) / (total + α · |valid_successors|)

Laplace smoothing was necessary because some branches at certain gateways appear rarely in the historical data. Without it, the predictor would assign zero probability to valid but infrequent paths.

At prediction time, the algorithm applies a fallback hierarchy. It first searches for the longest matching k-gram context in a per-activity table. If no match is found, it falls back to a global k-gram table, and finally to a flat distribution over all activities. Regardless of which level returns a result, the output is intersected with the set of valid successors and renormalized.

The major metric for this task is the average XOR difference between historical branching proportions and our predictor. Random branching produces a 37.5% difference, whilst our trained predictor achieves only 0.5%.

### Advanced Task

For the advanced task, we developed two separate components: a token replay module to extract structurally correct decision point data, and a machine learning model that leverages both trace history and case-level attributes.

**Token Replay**

A conformance-checking module was implemented in the task_1_4_token_replay.py file using the pm4py library. The module operates as follows:
1. The BPMN model is converted into a Petri net.
2. Every place with more than one outgoing arc is identified as a decision place. These correspond to XOR and OR gateways in the original model.
3. All 31,509 historical traces are replayed against the Petri net using pm4py's token replay algorithm.
4. Whenever a token arrives at a decision place and a transition fires, the module records the case identifier, the chosen transition, the full prefix of visible activities, and the set of available alternatives.

This procedure yielded 706,519 decision instances across 7 decision places. The data produced by token replay was considerably cleaner than what sequential pair counting produces, since the replay respects the concurrency and synchronization semantics of the Petri net rather than relying solely on the order of events in the flat log.

When the predictor is instantiated with use_token_replay=True, it first runs the basic k-gram fit as a fallback and then overwrites the transition counts with the replay-derived data. The prediction logic (context lookup, smoothing, sampling) remains the same.

**Case-Attribute-Enriched ML Model**

The k-gram predictor, even when combined with token replay data, is still a frequency-based sampler. It cannot condition on case-specific attributes such as the requested loan amount or the applicant's credit score. In the context of a loan application process, these attributes are likely to influence whether an application gets approved, rejected, or sent for another round of offers.

To address this limitation, we trained a RandomForest classifier (500 trees, max depth 25) on an enriched feature set. In addition to the activity n-gram and temporal features (hour, weekday, elapsed seconds since case start), the following case-level attributes were included:
● RequestedAmount
● Encoded LoanGoal and ApplicationType
● Loop counter (number of times the current activity has appeared in the trace)
● Trace length up to the current point
● Cumulative count of offer-related events
● Most recent CreditScore and OfferedAmount observed in the trace
● Mean inter-event time
● Number of distinct resources that have worked on the case

This results in 11 case-level features in total. We initially trained a GradientBoosting model, which reached 63.2% accuracy on XOR decisions. Switching to RandomForest with the same feature set improved this to 81.4%. We attribute the difference to RandomForest's better handling of mixed categorical and continuous features and its lower tendency to overfit towards the majority class at imbalanced gateways.

### Evaluation

One complication arose during evaluation. Our BPMN model uses activity clustering, meaning that certain raw activities are merged into single nodes (e.g., "W_Complete application" and "A_Concept" become "W_Complete application & A_Concept"). The event log, however, retains the original unclustered names. As a result, 5 out of 7 XOR gateways reference activity names that do not appear as consecutive transitions in the raw log. On the raw data, we could therefore only evaluate 2 gateways. This was a team-wide modelling decision and could not be changed without affecting other parts of the simulation.

To obtain a more complete evaluation, we ran an additional experiment in which the activity mapping was applied to the event log before training. Raw activities were replaced by their clustered equivalents, and consecutive duplicates within each trace were collapsed. This brought all 7 gateways into scope and increased the number of evaluable XOR decisions from approximately 8,200 to over 18,400.

Table X: Results on mapped event log (all 7 gateways)

| Metric | Random | Basic (k-gram) | Best (RF enriched) |
|--------|--------|---------------|---------------------|
| Avg XOR diff vs historical | 37.5% | 0.5% | n/a |
| XOR prediction accuracy | ~33-50% | 51.1% | 81.4% |


Table X: Per-gateway breakdown

| Gateway | Basic | RF enriched | Decisions |
|---------|-------|-------------|-----------|
| A_Submitted | 67.3% | 100.0% | 4,136 |
| O_Sent (mail and online) | 59.5% | 86.0% | 858 |
| A_Create Application | 54.4% | 83.9% | 6,302 |
| W_Call after offers & A_Complete | 37.3% | 69.6% | 7,090 |

The basic k-gram predictor achieves a very low distribution error (0.5%), meaning that the simulated process visits each branch in approximately the same proportions as the real process. For a simulation whose primary goal is to reproduce aggregate behavior, this is the more relevant metric.

For per-instance prediction accuracy, the enriched RandomForest significantly outperforms the k-gram baseline (81.4% vs 51.1%). The most difficult gateway is the three-way split at "W_Call after offers & A_Complete," where the historical distribution is approximately 62/26/13 between validate, new offer, and cancel. The k-gram predictor barely outperforms random selection at this gateway (37.3%), whilst the RandomForest reaches 69.6%.

We also experimented with different window sizes (k = 3, 5, and 7). The accuracy differences were negligible (81.0% to 81.4%), which suggests that the case-level attributes carry more predictive signal than longer activity histories.

---

## 2.1 Resource Allocation — Advanced Methods

In addition to the basic heuristics (random, round-robin, earliest-available), we implemented two advanced strategies.

### K-Batching

Based on Zeng and Zhao (2005) [3]. Instead of immediate assignment, the allocator accumulates K tasks (default K=5), then solves a Parallel Machines Scheduling Problem using either:
- **LPT heuristic**: Tasks sorted by decreasing duration, greedily assigned to earliest-free resource (4/3-approximation for makespan).
- **Hungarian algorithm**: Optimal one-to-one assignment via `scipy.optimize.linear_sum_assignment` (O(n³)).

A timeout (default 3600s) prevents starvation. Implemented in `ResourceAllocator/BatchAllocator.py`, wrapping the base `ResourceAllocatorAlgo`.

### SVFA (Score-based Value Function Approximation)

Based on Middelhuis et al. (2025) [2]. Scores every (resource, task) pair using a learned linear combination of six features:

Score(r, k) = w₁·MeanAssignment + w₂·VarAssignment + w₃·ActivityRank + w₄·ResourceRank − w₅·ProbFin − w₆·QueueLength

If no pair scores below threshold w₇, assignments are postponed. Weights are trained via Bayesian optimization (scikit-optimize, `gp_minimize`) minimizing mean cycle time. Implemented in `ResourceAllocator/SVFAllocator.py`.

Both strategies are integrated into the simulation engine as `allocation_strategy="k_batch"` / `"svfa"`.

---

## 2.2 Evaluation — Advanced Allocation Strategies

Scaling experiment: 100, 500, and 1,000 cases, all five strategies.

**Results at 1,000 Cases:**

| Strategy | Cycle Time (days) | Activity Delay (h) | Fairness (Jain) |
|----------|-------------------|--------------------|-----------------| 
| Earliest Available | **19.45** | **8.98** | 0.60 |
| Round Robin | 26.37 | 20.50 | 0.61 |
| Random | 29.69 | 23.31 | 0.58 |
| K-Batch | 58.77 | 99.66 | **0.91** |
| SVFA | 86.06 | 173.24 | 0.88 |

**Key finding: efficiency-fairness trade-off.** K-Batch and SVFA achieve ~50% higher fairness (Jain index 0.88–0.91 vs 0.60) but at the cost of longer cycle times. At low load (100 cases), SVFA achieves the **lowest** cycle time (13.61 days vs 14.30 for earliest-available), demonstrating that intelligent assignment provides genuine value when coordination overhead is negligible. Under high load, batching/scoring delays compound in a cascading effect.

**Practical recommendation:** Use earliest-available for SLA-critical processes; use K-Batch/SVFA when workload fairness matters (preventing burnout). A hybrid approach switching strategies based on load could combine both benefits.

---

## References

[2] J. Middelhuis et al., "Learning policies for resource allocation in business processes," *Inf. Syst.*, vol. 128, art. 102492, 2025.

[3] D. D. Zeng and J. L. Zhao, "Effective role resolution in workflow management," *INFORMS J. Comput.*, vol. 17, no. 3, pp. 374–387, 2005.
