import pandas as pd

sim = pd.read_csv("sim_random.csv")
hist = pd.read_csv("bpi2017.csv")

print(len(sim.merge(hist, on=["concept:name","org:resource"], how="left", indicator=True)
          .query("_merge=='left_only'")))

print("missing org:resource rows:", sim["org:resource"].isna().sum())


