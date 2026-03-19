import pandas as pd

from archive.engine_core_versions.simulation_engine_core_V1_7 import learn_advanced_resource_permissions

# choose csv file to check
sim = pd.read_csv("sim_outputs/sim_predicted_random_advanced_roles.csv").dropna(subset=["concept:name","org:resource"])
hist = pd.read_csv("bpi2017.csv").dropna(subset=["concept:name","org:resource"])

# normalize
for df in (sim, hist):
    df["concept:name"] = df["concept:name"].astype(str).str.strip()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()

# drop END
sim = sim[sim["concept:name"] != "END"]

# build hist permission sets
perms = hist.groupby("concept:name")["org:resource"].apply(set).to_dict()
adv_perms = learn_advanced_resource_permissions(hist, n_roles=5, threshold=0.1)

def ok(row):
    acts = [a.strip() for a in row["concept:name"].split("&")]
    res = row["org:resource"]
    return all(res in perms.get(a, set()) for a in acts)

def ok_advanced(row):
    acts = [a.strip() for a in row["concept:name"].split("&")]
    res = row["org:resource"]
    return all(res in adv_perms.get(a, set()) for a in acts)

viol = sim[~sim.apply(ok, axis=1)]
adv_viol = sim[~sim.apply(ok_advanced, axis=1)]

print("Role-based generalizations (fail normal, but pass advanced):", len(viol) - len(adv_viol))

print("violations (split-aware):", len(viol))
print(viol[["case:concept:name","concept:name","org:resource"]].head(200))
print("violations (split-unaware):", len(sim[~sim.apply(lambda r: r["org:resource"] in perms.get(r["concept:name"], set()), axis=1)]))
print("total:", len(sim))
# print(sim[~sim.apply(lambda r: r["org:resource"] in perms.get(r["concept:name"], set()), axis=1)][["case:concept:name","concept:name","org:resource"]].head(20))

print("Actual errors (fail advanced):", len(adv_viol))

if len(adv_viol) > 0:
    print("\nActual Errors (split-aware):")
    print(adv_viol[["case:concept:name","concept:name","org:resource"]].head(20))