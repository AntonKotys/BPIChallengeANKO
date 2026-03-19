import pandas as pd
from collections import Counter
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

# Load and preprocess
df = pd.read_csv("bpi2017filtered.csv")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
df = df.sort_values(["case:concept:name", "time:timestamp"])

# Build traces
grouped = df.groupby("case:concept:name")["concept:name"].apply(list)

# Define variant logic families
variant_1 = grouped[grouped.apply(
    lambda trace: "A_Cancelled" in trace and "O_Cancelled" in trace
)]
variant_2 = grouped[grouped.apply(
    lambda trace:
        any(x in trace for x in ["W_Call incomplete files", "A_Incomplete"]) and
        any(x in trace for x in ["O_Accepted", "A_Pending"])
)]
variant_3 = grouped[grouped.apply(
    lambda trace:
        any(x in trace for x in ["O_Accepted", "A_Pending"]) and
        not any(x in trace for x in ["W_Call incomplete files", "A_Incomplete"])
)]
variant_9 = grouped[grouped.apply(
    lambda trace: any(x in trace for x in ["A_Denied", "O_Refused"])
)]

variants = {
    "Variant 1 – Cancelled (No Response)": variant_1,
    "Variant 2 – Accepted w/ Incomplete Files": variant_2,
    "Variant 3 – Accepted w/o Incomplete Files": variant_3,
    "Variant 9 – Denied by Bank": variant_9
}

#Extract top 10 sequences per family
def top_sequences(traces, n=10):
    counter = Counter(tuple(trace) for trace in traces)
    return counter.most_common(n)

# Collect all top sequences
sequences = []
for name, var in variants.items():
    if len(var) == 0:
        continue
    top10 = top_sequences(var, n=10)
    for i, (seq, count) in enumerate(top10, start=1):
        sequences.append(list(seq))

print(f"Collected {len(sequences)} total top trace variants across all variant families.\n")

#Filtering for exact matches
case_groups = df.groupby("case:concept:name")
total_cases = len(case_groups)
matching_case_ids = set()

for case_id, group in case_groups:
    trace = group["concept:name"].tolist()
    for seq in sequences:
        if trace == seq:  # exact match
            matching_case_ids.add(case_id)
            break  # stop once one matches

filtered_df_exact = df[df["case:concept:name"].isin(matching_case_ids)]

print(f"Found {len(matching_case_ids)} exact-matching cases out of {total_cases} total.")
print(f"Filtered dataframe now has {len(filtered_df_exact)} events.")

filtered_df_exact.to_csv("bpi2017_invariant_cases_strict_famAggregated.csv", index=False)
print("💾 Saved as bpi2017_invariant_cases_strict_famAggregated.csv")

df_filtered = pd.read_csv("bpi2017_invariant_cases_strict_famAggregated.csv")
df = pd.read_csv("bpi2017filtered.csv")

parameters = {
    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
    heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: 1,
    heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 300
}
event_log = log_converter.apply(df_filtered)
event_log2 = log_converter.apply(df)

heu_net_app = heuristics_miner.apply_heu(event_log)
gviz_app = hn_visualizer.apply(heu_net_app)
hn_visualizer.save(gviz_app, "heuristic_event_seqFiltered_strict_famAggregated.png")

heu_net_app_param = heuristics_miner.apply_heu(event_log, parameters=parameters)
gviz_app = hn_visualizer.apply(heu_net_app_param)
hn_visualizer.save(gviz_app, "heuristic_event_seqFiltered_strict_famAggregated_param.png")

net, im, fm = hn_converter.apply(heu_net_app_param)

# Quality Metrics
# fitness_result = fitness_evaluator.apply(event_log2, net, im, fm, variant=fitness_evaluator.Variants.TOKEN_BASED)
# fitness = fitness_result["average_trace_fitness"]
# print(f" Fitness: {fitness:.4f}")
#
# precision = precision_evaluator.apply(event_log2, net, im, fm)
# print(f" Precision: {precision:.4f}")

generalization = generalization_evaluator.apply(event_log2, net, im, fm)
print(f"Generalization: {generalization:.4f}")

simplicity = simplicity_evaluator.apply(net, im, fm)
print(f"Simplicity: {simplicity:.4f}")
