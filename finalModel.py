from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import pandas as pd

df = pd.read_csv("bpi2017filtered.csv")
event_log = log_converter.apply(df)

bpmn_model = bpmn_importer.apply("bpianko9.0.bpmn")

net, im, fm = bpmn_converter.apply(bpmn_model)

fitness_result = fitness_evaluator.apply(event_log, net, im, fm, variant=fitness_evaluator.Variants.TOKEN_BASED)
fitness = fitness_result["average_trace_fitness"]

print(f" Fitness: {fitness:.4f}")

precision = precision_evaluator.apply(event_log, net, im, fm)
print(f" Precision: {precision:.4f}")

generalization = generalization_evaluator.apply(event_log, net, im, fm)
print(f"Generalization: {generalization:.4f}")

simplicity = simplicity_evaluator.apply(net)
print(f"Simplicity: {simplicity:.4f}")