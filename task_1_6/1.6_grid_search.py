import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from simulation_engine_core_final_version import learn_resource_permissions, learn_advanced_resource_permissions

print("Loading data...")
sim = pd.read_csv("sim_outputs/sim_predicted_random_basic_roles.csv").dropna(subset=["concept:name","org:resource"])
hist = pd.read_csv("bpi2017.csv").dropna(subset=["concept:name","org:resource"])

for df in (sim, hist):
    df["concept:name"] = df["concept:name"].astype(str).str.strip()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()
sim = sim[sim["concept:name"] != "END"]

strict_perms = learn_resource_permissions(hist)

def ok_strict(row):
    acts = [a.strip() for a in row["concept:name"].split("&")]
    res = row["org:resource"]
    return all(res in strict_perms.get(a, set()) for a in acts)

sim["passes_strict"] = sim.apply(ok_strict, axis=1)
strict_total_pairs = sum(len(res_set) for res_set in strict_perms.values())

n_roles_list = [5, 10, 15, 20]
thresholds_list = [0.05, 0.1, 0.2, 0.3]

print(f"Strict Baseline Allowed Pairs: {strict_total_pairs}\n")
print(f"{'n_roles':<10} {'threshold':<10} {'Generalizations':<20} {'Total Allowed Pairs':<20}")

for n in n_roles_list:
    for t in thresholds_list:
        adv_perms = learn_advanced_resource_permissions(hist, n_roles=n, threshold=t)
        
        def ok_adv(row):
            acts = [a.strip() for a in row["concept:name"].split("&")]
            res = row["org:resource"]
            return all(res in adv_perms.get(a, set()) for a in acts)
        
        passes_adv = sim.apply(ok_adv, axis=1)
        
        generalizations = sum((~sim["passes_strict"]) & passes_adv)
        
        total_pairs = sum(len(res_set) for res_set in adv_perms.values())
        
        print(f"{n:<10} {t:<10} {generalizations:<20} {total_pairs:<20}")