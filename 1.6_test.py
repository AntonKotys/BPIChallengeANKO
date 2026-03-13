import pandas as pd

# choose csv file to check
sim = pd.read_csv("sim_predicted_random.csv").dropna(subset=["concept:name","org:resource"])
hist = pd.read_csv("bpi2017.csv").dropna(subset=["concept:name","org:resource"])

# normalize
for df in (sim, hist):
    df["concept:name"] = df["concept:name"].astype(str).str.strip()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()

# drop END
sim = sim[sim["concept:name"] != "END"]

# build hist permission sets
perms = hist.groupby("concept:name")["org:resource"].apply(set).to_dict()

def ok(row):
    acts = [a.strip() for a in row["concept:name"].split("&")]
    res = row["org:resource"]
    return all(res in perms.get(a, set()) for a in acts)

viol = sim[~sim.apply(ok, axis=1)]
print("violations (split-aware):", len(viol))
print(viol[["case:concept:name","concept:name","org:resource"]].head(20))
print("violations (split-unaware):", len(sim[~sim.apply(lambda r: r["org:resource"] in perms.get(r["concept:name"], set()), axis=1)]))
print("total:", len(sim))
# print(sim[~sim.apply(lambda r: r["org:resource"] in perms.get(r["concept:name"], set()), axis=1)][["case:concept:name","concept:name","org:resource"]].head(20))