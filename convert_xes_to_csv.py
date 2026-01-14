import pm4py
import pandas as pd

XES_PATH = "BPI_Challenge_2017.xes.gz"
CSV_PATH = "bpi2017.csv"

log = pm4py.read_xes(XES_PATH)
df = pm4py.convert_to_dataframe(log)

df.to_csv(CSV_PATH, index=False)
print(f"Saved {CSV_PATH} with {len(df)} rows")
