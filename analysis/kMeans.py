import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("bpi2017filtered.csv")

#Ensure timestamps are in datetime format
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="mixed", errors="coerce", utc=True)


# 1. Case duration (in days)
case_duration = (
    df.groupby("case:concept:name")["time:timestamp"]
    .agg(lambda x: (x.max() - x.min()).total_seconds() / 86400)  # convert to days
    .rename("case_duration_days")
)

# 2. Count of “W_Call incomplete files” per case
call_counts = (
    df[df["concept:name"] == "W_Call incomplete files"]
    .groupby("case:concept:name")["concept:name"]
    .count()
    .rename("num_incomplete_calls")
)

features = pd.concat([case_duration, call_counts], axis=1).fillna(0)

# Standardize features for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Run K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
features["cluster"] = kmeans.fit_predict(X_scaled)

print(features.groupby("cluster").agg({
    "case_duration_days": ["mean", "std"],
    "num_incomplete_calls": ["mean", "std", "count"]
}))

plt.figure(figsize=(8,6))
plt.scatter(
    features["num_incomplete_calls"],
    features["case_duration_days"],
    c=features["cluster"],
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Number of 'W_Call incomplete files'")
plt.ylabel("Case Duration (days)")
plt.title("K-Means Clustering: Case Duration vs Incomplete Calls")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.savefig("kmeans_clusters.png", dpi=300, bbox_inches="tight")
