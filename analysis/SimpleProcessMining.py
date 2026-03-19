from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.objects.conversion.heuristics_net import converter as hn_converter
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import pandas as pd

# Load the event log
# event_log = xes_importer.apply(log_path)

# Convert PM4Py event log to pandas DataFrame
# df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
# Convert pandas df into csv file for future reuse
# df.to_csv("bpi2017.csv", index=False)

df = pd.read_csv("bpi2017.csv")

end_activities = ["A_Cancelled", "A_Denied", "A_Pending", "O_Cancelled", "O_Refused"]

# Keep only cases that contain at least one endpoint activity
cases_with_end = df[df['concept:name'].isin(end_activities)]['case:concept:name'].unique()
df_filtered = df[df['case:concept:name'].isin(cases_with_end)]

print(f"After Step 1: {len(df_filtered['case:concept:name'].unique())} cases remain.")

# Further filter to keep only cases that have A Pending, A Cancelled or A Denied
final_decision_acts = ["A_Pending", "A_Cancelled", "A_Denied"]
cases_with_final_decision = df_filtered[df_filtered['concept:name'].isin(final_decision_acts)]['case:concept:name'].unique()

df_filtered = df_filtered[df_filtered['case:concept:name'].isin(cases_with_final_decision)]

df_filtered = df_filtered.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

df_filtered.to_csv("bpi2017filtered.csv", index=False)

print(f"After Step 2: {len(df_filtered['case:concept:name'].unique())} cases remain.")


original_cases = df['case:concept:name'].nunique()
filtered_cases = df_filtered['case:concept:name'].nunique()
removed_cases = original_cases - filtered_cases

print(f"Original cases: {original_cases}")
print(f"Filtered cases: {filtered_cases}")
print(f"Removed incomplete cases: {removed_cases}")

app_df = df_filtered[df_filtered['EventOrigin'] == 'Application'].copy()
offer_df = df_filtered[df_filtered['EventOrigin'] == 'Offer'].copy()
workflow_df = df_filtered[df_filtered['EventOrigin'] == 'Workflow'].copy()

event_log = log_converter.apply(df_filtered)
app_log = log_converter.apply(app_df)
offer_log = log_converter.apply(offer_df)
workflow_log = log_converter.apply(workflow_df)

#Heuristics miners -----------------------------------------------------------------

heu_net_app = heuristics_miner.apply_heu(event_log)
gviz_app = hn_visualizer.apply(heu_net_app)
hn_visualizer.save(gviz_app, "heuristic_event.png")


# Quality metrics --------

net, im, fm = hn_converter.apply(heu_net_app)

fitness_result = fitness_evaluator.apply(event_log, net, im, fm, variant=fitness_evaluator.Variants.TOKEN_BASED)
fitness = fitness_result["average_trace_fitness"]
print(f" Fitness: {fitness:.4f}")
#
# Turn Multiprocessing on to get a quicker result : does not work on my M1 Chip
# parameters = {
#     precision_evaluator.etconformance_token.Parameters.MULTIPROCESSING: True
# }
# precision = precision_evaluator.apply(event_log, net, im, fm, parameters=parameters)
# print(f" Precision: {precision:.4f}")
#
generalization = generalization_evaluator.apply(event_log, net, im, fm)
print(f"Generalization: {generalization:.4f}")

simplicity = simplicity_evaluator.apply(net)
print(f"Simplicity: {simplicity:.4f}")
# Quality metrics --------

# Application log
heu_net_app = heuristics_miner.apply_heu(app_log)
gviz_app = hn_visualizer.apply(heu_net_app)
hn_visualizer.save(gviz_app, "heuristic_application.png")

# Offer log
heu_net_offer = heuristics_miner.apply_heu(offer_log)
gviz_offer = hn_visualizer.apply(heu_net_offer)
hn_visualizer.save(gviz_offer, "heuristic_offer.png")

# Workflow log
heu_net_workflow = heuristics_miner.apply_heu(workflow_log)
gviz_workflow = hn_visualizer.apply(heu_net_workflow)
hn_visualizer.save(gviz_workflow, "heuristic_workflow.png")

#Heuristics miners -----------------------------------------------------------------


#Inductive miners -----------------------------------------------------------------
tree = inductive_miner.apply(event_log)
net, im, fm = pt_converter.apply(tree)
gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.save(gviz, "inductive_model.png")

# Quality metrics --------

fitness_result = fitness_evaluator.apply(event_log, net, im, fm, variant=fitness_evaluator.Variants.TOKEN_BASED)
fitness = fitness_result["average_trace_fitness"]
print(f" Fitness: {fitness:.4f}")
# precision = precision_evaluator.apply(event_log, net, im, fm, )
# print(f" Precision: {precision:.4f}")

generalization = generalization_evaluator.apply(event_log, net, im, fm)
print(f"Generalization: {generalization:.4f}")

simplicity = simplicity_evaluator.apply(net)
print(f"Simplicity: {simplicity:.4f}")

# Quality metrics --------

#Inductive miners -----------------------------------------------------------------


