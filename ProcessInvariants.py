from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.filtering.log.variants import variants_filter
import pandas as pd
df = pd.read_csv("bpi2017filtered.csv")

# Different possible executed log events sequences :  ---------------------------
target_sequence1 = [
    "A_Create Application",
    "A_Submitted",
    "W_Handle leads",
    "A_Concept",
    "W_Complete application",
    "A_Accepted",
    "O_Create Offer",
    "O_Created",
    "O_Sent (mail and online)",
    "W_Call after offers",
    "A_Complete",
    "A_Cancelled",
    "O_Cancelled"
]
target_sequence3 = [
    "A_Create Application",
    "A_Submitted",
    "W_Handle leads",
    "A_Concept",
    "W_Complete application",
    "A_Accepted",
    "O_Create Offer",
    "O_Created",
    "O_Sent (mail and online)",
    "W_Call after offers",
    "A_Complete",
    "W_Validate application",
    "A_Validating",
    "O_Returned",
    "W_Call incomplete files",
    "A_Incomplete",
    "W_Validate application",
    "A_Validating",
    "O_Accepted",
    "A_Pending"
]
target_sequence4 = [
    "A_Create Application",
    "A_Submitted",
    "W_Handle leads",
    "A_Concept",
    "W_Complete application",
    "A_Accepted",
    "O_Create Offer",
    "O_Created",
    "O_Sent (mail and online)",
    "W_Call after offers",
    "A_Complete",
    "W_Validate application",
    "A_Validating",
    "O_Returned",
    "O_Accepted",
    "A_Pending"
]
target_sequence5 = [
    "A_Create Application",
    "A_Submitted",
    "W_Handle leads",
    "A_Concept",
    "W_Complete application",
    "A_Accepted",
    "O_Create Offer",
    "O_Created",
    "O_Sent (mail and online)",
    "W_Call after offers",
    "A_Complete",
    "W_Validate application",
    "A_Validating",
    "O_Returned",
    "A_Denied",
    "O_Refused"
]

# -------------------------------------------------

sequences = []
case_groups = df.groupby("case:concept:name")

sequences.extend([target_sequence1, target_sequence3, target_sequence4, target_sequence5])

total_cases = len(case_groups)

# we iterate here for each above defined sequence and calculate the number of cases which do hold to these sequences
for i, sequence in enumerate(sequences, start=1):
    matching_cases = 0
    for _, group in case_groups:
        trace = group["concept:name"].tolist()
        pos = 0
        for event in trace:
            if pos < len(sequence) and event == sequence[pos]:
                pos += 1
        if pos == len(sequence):
            matching_cases += 1

    percentage = (matching_cases / total_cases) * 100
    print(f" Sequence {i}: {matching_cases} of {total_cases} cases ({percentage:.2f}%) follow the target sequence.")

matching_case_ids = set()

#We loop here  through all cases and check if they satisfy any of the defined sequence
for case_id, group in case_groups:
    trace = group["concept:name"].tolist()
    for sequence in sequences:
        pos = 0
        for event in trace:
            if pos < len(sequence) and event == sequence[pos]:
                pos += 1
        if pos == len(sequence):
            matching_case_ids.add(case_id)
            break  # Stop checking other sequences if one matches

filtered_df = df[df["case:concept:name"].isin(matching_case_ids)]

print(f" Found {len(matching_case_ids)} matching cases out of {len(case_groups)} total.")
print(f" Filtered dataframe now has {len(filtered_df)} events.")

filtered_df.to_csv("bpi2017_invariant_cases.csv", index=False)

parameters = {
    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.95,
    heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_ACT_COUNT: 1000,
    heuristics_miner.Variants.CLASSIC.value.Parameters.MIN_DFG_OCCURRENCES: 2500
}

df_filtered = pd.read_csv("bpi2017_invariant_cases.csv")
df = pd.read_csv("bpi2017filtered.csv")
event_log = log_converter.apply(df_filtered)
event_log2 = log_converter.apply(df)
heu_net_app = heuristics_miner.apply_heu(event_log)
gviz_app = hn_visualizer.apply(heu_net_app)
hn_visualizer.save(gviz_app, "heuristic_event_seqFiltered.png")

heu_net_simplified = heuristics_miner.apply_heu(event_log, parameters = parameters)
gviz_simple = hn_visualizer.apply(heu_net_simplified)
hn_visualizer.save(gviz_simple, "heuristic_event_seqFiltered_simplified.png")

# Quality Metrics
# net, im, fm = hn_converter.apply(heu_net_simplified)

# fitness_result = fitness_evaluator.apply(event_log2, net, im, fm, variant=fitness_evaluator.Variants.TOKEN_BASED)
# fitness = fitness_result["average_trace_fitness"]
# print(f" Fitness: {fitness:.4f}")
#
# precision = precision_evaluator.apply(event_log2, net, im, fm)
# print(f" Precision: {precision:.4f}")
#
# generalization = generalization_evaluator.apply(event_log2, net, im, fm)
# print(f"Generalization: {generalization:.4f}")
#
# simplicity = simplicity_evaluator.apply(net, im, fm)
# print(f"Simplicity: {simplicity:.4f}")


