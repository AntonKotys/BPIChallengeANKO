import pandas as pd
from collections import Counter

df = pd.read_csv("bpi2017filtered.csv")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
df = df.sort_values(["case:concept:name", "time:timestamp"])
grouped = df.groupby("case:concept:name")["concept:name"].apply(list)

# Define logical families
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
    "Variant 2 – Accepted with Incomplete Files": variant_2,
    "Variant 3 – Accepted without Incomplete Files": variant_3,
    "Variant 4 – Denied by Bank": variant_9
}

def top_sequences(traces, n=10):
    counter = Counter(tuple(trace) for trace in traces)
    return counter.most_common(n)

for name, var in variants.items():
    print(f"\n\n=== {name} ({len(var)} cases) ===")
    if len(var) == 0:
        print("️No matching cases found.")
        continue

    top10 = top_sequences(var, n=10)

    for i, (seq, count) in enumerate(top10, start=1):
        print(f"\nVariant {i} – {count} cases")
        print(" → ".join(seq))
        print("\nPython list format:")
        print("[")
        for act in seq:
            print(f"    '{act}',")
        print("]")
