from datetime import datetime
import pandas as pd
import numpy as np

class DynamicArrivalModel:
    """
    Context-dependent arrival model.
    Learns one arrival rate lambda for each
    (weekday, time_bin) combination.
    """

    def __init__(self, lambda_table: dict, default_lambda: float, bin_hours: int = 6):
        self.lambda_table = lambda_table
        self.default_lambda = default_lambda
        self.bin_hours = bin_hours

    def get_bin(self, t: datetime) -> int:
        return t.hour // self.bin_hours

    def get_lambda(self, t: datetime) -> float:
        weekday = t.weekday()
        bin_idx = self.get_bin(t)
        return self.lambda_table.get((weekday, bin_idx), self.default_lambda)

    def sample_interarrival(self, t: datetime, max_interarrival: float) -> float:
        lam = self.get_lambda(t)

        # I added a fallback if the rate is invalid
        if lam is None or lam <= 0:
            return max_interarrival

        sampled = np.random.exponential(scale=1 / lam)
        return min(sampled, max_interarrival)


def fit_dynamic_arrival_model(csv_path: str, bin_hours: int = 6) -> DynamicArrivalModel:
    """
    I implemented a dynamic arrival model from the event log that goes like:
    1. Extract first timestamp per case = case arrival time
    2. Group arrivals by (weekday, time_bin)
    3. Estimate lambda =arrivals / observed_seconds_in_that_context
    """

    df = pd.read_csv(csv_path, usecols=["case:concept:name", "time:timestamp"])
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="ISO8601", utc=True)
    arrivals = (
        df.groupby("case:concept:name")["time:timestamp"]
        .min()
        .sort_values()
        .reset_index(drop=True)
    )

    arr_df = pd.DataFrame({"arrival_time": arrivals})
    arr_df["weekday"] = arr_df["arrival_time"].dt.weekday
    arr_df["hour"] = arr_df["arrival_time"].dt.hour
    arr_df["bin"] = arr_df["hour"] // bin_hours
    arr_df["date"] = arr_df["arrival_time"].dt.date

    arrivals_per_context = arr_df.groupby(["weekday", "bin"]).size().to_dict()

    min_day = arr_df["arrival_time"].min().floor("D")
    max_day = arr_df["arrival_time"].max().floor("D")
    calendar_days = pd.date_range(start=min_day, end=max_day, freq="D", tz="UTC")

    weekday_day_counts = {}
    for d in calendar_days:
        wd = d.weekday()
        weekday_day_counts[wd] = weekday_day_counts.get(wd, 0) + 1

    seconds_per_bin = bin_hours * 3600
    lambda_table = {}

    for (weekday, bin_idx), count in arrivals_per_context.items():
        n_days_for_weekday = weekday_day_counts.get(weekday, 1)

        observed_seconds = n_days_for_weekday * seconds_per_bin

        lam = count / observed_seconds if observed_seconds > 0 else 0.0
        lambda_table[(weekday, bin_idx)] = lam

    total_seconds = (arr_df["arrival_time"].max() - arr_df["arrival_time"].min()).total_seconds()
    default_lambda = len(arr_df) / total_seconds if total_seconds > 0 else 1 / 1089.18

    return DynamicArrivalModel(
        lambda_table=lambda_table,
        default_lambda=default_lambda,
        bin_hours=bin_hours
    )