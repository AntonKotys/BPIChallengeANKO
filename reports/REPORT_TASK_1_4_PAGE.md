# 1.4 Next Activities

As described in Section 1.1, at each XOR gateway the simulation engine queries an ExpertActivityPredictor to determine which branch a case should follow. We implemented two versions of this predictor in task_1_4_next_activity.py.

## Basic

The basic predictor is a k-gram model (k = 2) that learns transition probabilities exclusively from valid process-model edges. During training, every consecutive activity pair (A → B) in the event log is checked against the PROCESS_MODEL dictionary; pairs where B is not a valid successor of A are discarded. The remaining counts are converted to probability distributions with Laplace smoothing (α = 1.0):

P(next | current, context) = (count(next) + α) / (total + α · |valid successors|)

At prediction time, a fallback hierarchy is applied: the predictor first looks up the longest matching k-gram context in a per-activity table, then falls back to a global k-gram table, and finally to a uniform distribution. The result is always intersected with the set of valid successors and renormalized, so the predictor can never produce an impossible transition.

## Advanced

For the advanced task, decision points are first identified via token replay using pm4py. The BPMN model is converted into a Petri net, and every place with more than one outgoing arc is marked as a decision place. All 31,509 historical traces are replayed, yielding 706,519 decision instances across 7 XOR gateways. Compared to sequential pair counting, this approach respects the concurrency and synchronization semantics of the Petri net.

A RandomForest classifier (500 trees, max depth 25) is then trained on an 18-dimensional feature vector: a 5-activity n-gram history, three temporal features (hour, weekday, elapsed seconds since case start), and 11 case-level attributes — RequestedAmount, encoded LoanGoal and ApplicationType, loop count, trace length, cumulative number of offer events, most recent CreditScore and OfferedAmount, mean inter-event time, and number of distinct resources that have worked on the case. Training uses an 80/20 group-shuffle split (grouped by case ID) to prevent data leakage between training and validation.

## Evaluation

One complication arose from the activity clustering used in our BPMN model. Because the event log retains the original unclustered activity names, 5 of 7 gateways could not be evaluated directly. To obtain a complete evaluation, we applied the same activity mapping to the event log and collapsed consecutive duplicates within each trace, bringing all 7 gateways into scope.

| Metric | Random | Basic (k-gram) | Advanced (RF) |
|--------|--------|----------------|---------------|
| Avg XOR diff vs historical | 37.5% | 0.5% | n/a |
| XOR prediction accuracy | ~33–50% | 51.1% | **81.4%** |

The basic predictor achieves a very low distribution error (0.5%), meaning the simulated process visits each branch in approximately the same proportions as the real process — the most relevant metric for aggregate simulation fidelity. The RandomForest significantly improves per-instance accuracy (81.4% vs 51.1%). The hardest gateway is the three-way split at "W_Call after offers & A_Complete" (historical distribution ≈ 62/26/13), where the k-gram predictor scores 37.3% and the RandomForest reaches 69.6%. Experiments with window sizes k = 3, 5, 7 showed negligible differences, indicating that case-level attributes carry more predictive signal than longer activity histories.
