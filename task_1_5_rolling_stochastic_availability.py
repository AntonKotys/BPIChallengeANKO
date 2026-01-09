from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class DailyWindow:
    start: time
    end: time


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mins_to_time(m: float) -> time:
    m = _clamp(m, 0.0, 1439.0)
    return time(int(m // 60), int(m % 60), 0)


class RollingStochasticAvailabilityModel:
    """
    Advanced Task 1.5:
    - Breaks strict 2-week periodicity
    - Rolling-window estimation of start/end parameters over time (drift)
    - Stochastic daily window sampling (noise)
    - Cached per (resource, date) so one consistent shift per day
    """

    def __init__(
        self,
        daily_stats: pd.DataFrame,
        window_days: int = 28,
        min_points: int = 6,
        q_start: float = 0.05,
        q_end: float = 0.95,
        min_shift_minutes: int = 360,
        random_seed: int = 42,
        condition_on_weekday: bool = False,
    ):
        self.daily_stats = daily_stats  # columns: resource, date, weekday, day_start, day_end
        self.window_days = int(window_days)
        self.min_points = int(min_points)
        self.min_shift_minutes = int(min_shift_minutes)
        self.condition_on_weekday = bool(condition_on_weekday)
        self.q_start = float(q_start)
        self.q_end = float(q_end)
        self.rng = np.random.default_rng(random_seed)

        # cache sampled daily windows
        self.daily_cache: Dict[Tuple[str, date], DailyWindow] = {}
        # cache computed rolling params
        self.param_cache: Dict[Tuple[str, date, Optional[int]], Optional[Tuple[float, float, float, float]]] = {}

    @staticmethod
    def fit_from_csv(
        csv_path: str,
        resource_col: str = "org:resource",
        ts_col: str = "time:timestamp",
        q_start: float = 0.05,
        q_end: float = 0.95,
    ) -> "RollingStochasticAvailabilityModel":
        df = pd.read_csv(csv_path, usecols=[resource_col, ts_col]).dropna()
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        df[resource_col] = df[resource_col].astype(str)

        df["date"] = df[ts_col].dt.date
        df["weekday"] = df[ts_col].dt.weekday
        df["tod_min"] = (
            df[ts_col].dt.hour * 60
            + df[ts_col].dt.minute
            + df[ts_col].dt.second / 60.0
        )

        # per resource-day derive daily start/end from quantiles
        daily = (
            df.groupby([resource_col, "date", "weekday"], sort=False)["tod_min"]
            .agg(
                day_start=lambda s: float(np.quantile(s, q_start)),
                day_end=lambda s: float(np.quantile(s, q_end)),
                n="count",
            )
            .reset_index()
            .rename(columns={resource_col: "resource"})
        )

        # filter out ultra sparse days
        daily = daily[daily["n"] >= 10].copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.date

        return RollingStochasticAvailabilityModel(
            daily_stats=daily,
            q_start=q_start,
            q_end=q_end,
        )

    def _rolling_params(self, resource: str, d: date, weekday: int) -> Optional[Tuple[float, float, float, float]]:
        key = (resource, d, weekday if self.condition_on_weekday else None)
        if key in self.param_cache:
            return self.param_cache[key]

        start_date = d - timedelta(days=self.window_days)
        g = self.daily_stats[
            (self.daily_stats["resource"] == resource)
            & (self.daily_stats["date"] < d)
            & (self.daily_stats["date"] >= start_date)
        ]

        if self.condition_on_weekday:
            g = g[g["weekday"] == weekday]

        if len(g) < self.min_points:
            self.param_cache[key] = None
            return None

        start_vals = g["day_start"].to_numpy(dtype=float)
        end_vals = g["day_end"].to_numpy(dtype=float)

        mu_s = float(np.mean(start_vals))
        mu_e = float(np.mean(end_vals))
        sd_s = float(np.std(start_vals, ddof=1)) if len(start_vals) > 1 else 0.0
        sd_e = float(np.std(end_vals, ddof=1)) if len(end_vals) > 1 else 0.0

        # ensure some variation
        # sd_s = max(sd_s, 5.0)
        # sd_e = max(sd_e, 5.0)
        sd_s = min(max(sd_s, 5.0), 60.0)  # max 1 hour
        sd_e = min(max(sd_e, 5.0), 60.0)

        self.param_cache[key] = (mu_s, sd_s, mu_e, sd_e)
        return self.param_cache[key]

    def _sample_window_for_date(self, resource: str, d: date) -> Optional[DailyWindow]:
        cache_key = (resource, d)
        if cache_key in self.daily_cache:
            return self.daily_cache[cache_key]

        weekday = datetime.combine(d, time(0, 0)).weekday()
        params = self._rolling_params(resource, d, weekday)
        if params is None:
            return None

        mu_s, sd_s, mu_e, sd_e = params

        start_m = float(self.rng.normal(mu_s, sd_s))
        end_m = float(self.rng.normal(mu_e, sd_e))

        start_m = _clamp(start_m, 0.0, 1439.0)
        end_m = _clamp(end_m, 0.0, 1439.0)

        # repair invalid / too short window
        if end_m <= start_m:
            end_m = start_m + self.min_shift_minutes
        if end_m > 1439.0:
            end_m = 1439.0
        if end_m <= start_m:
            return None

        w = DailyWindow(start=_mins_to_time(start_m), end=_mins_to_time(end_m))
        self.daily_cache[cache_key] = w
        return w

    def next_available(self, resource: str, t: datetime) -> datetime:
        resource = str(resource)

        for day_offset in range(0, 30):
            d = t.date() + timedelta(days=day_offset)
            cand = datetime.combine(d, t.time() if day_offset == 0 else time(0, 0, 0))

            w = self._sample_window_for_date(resource, d)
            if w is None:
                continue

            day_start = datetime.combine(d, w.start)
            day_end = datetime.combine(d, w.end)

            if cand <= day_start:
                return day_start
            if day_start <= cand < day_end:
                return cand

        return t
