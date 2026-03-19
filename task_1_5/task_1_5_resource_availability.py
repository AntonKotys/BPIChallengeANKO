from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class DailyWindow:
    start: time
    end: time

class TwoWeekAvailabilityModel:
    """
    Basic (Task 1.5):
    resource availability repeats on a 2-week cycle.
    window per (resource, week_parity, weekday) inferred from historical timestamps.
    """
    def __init__(self, windows: Dict[Tuple[str, int, int], DailyWindow]):
        self.windows = windows

    @staticmethod
    def fit_from_csv(
        csv_path: str,
        resource_col: str = "org:resource",
        ts_col: str = "time:timestamp",
        q_low: float = 0.10,
        q_high: float = 0.90,
        min_window_minutes: int = 60,
    ) -> "TwoWeekAvailabilityModel":
        df = pd.read_csv(csv_path, usecols=[resource_col, ts_col])
        df = df.dropna(subset=[resource_col, ts_col])
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])
        df[resource_col] = df[resource_col].astype(str)

        # week parity based on ISO week number
        iso = df[ts_col].dt.isocalendar()
        df["week_parity"] = (iso.week.astype(int) % 2)
        df["weekday"] = df[ts_col].dt.weekday  # Mon=0..Sun=6

        # minutes since midnight
        mins = df[ts_col].dt.hour * 60 + df[ts_col].dt.minute + df[ts_col].dt.second / 60.0
        df["tod_min"] = mins

        windows: Dict[Tuple[str, int, int], DailyWindow] = {}

        for (r, wp, wd), g in df.groupby([resource_col, "week_parity", "weekday"], sort=False):
            if len(g) < 5:
                continue

            lo = float(np.quantile(g["tod_min"], q_low))
            hi = float(np.quantile(g["tod_min"], q_high))

            # enforce minimum window length
            if hi - lo < min_window_minutes:
                mid = (hi + lo) / 2.0
                lo = mid - min_window_minutes / 2.0
                hi = mid + min_window_minutes / 2.0

            lo = max(0.0, min(1439.0, lo))
            hi = max(0.0, min(1439.0, hi))

            start_t = time(int(lo // 60), int(lo % 60), 0)
            end_t = time(int(hi // 60), int(hi % 60), 0)

            # skip invalid or overnight windows (basic model only)
            if end_t <= start_t:
                continue

            windows[(r, int(wp), int(wd))] = DailyWindow(start=start_t, end=end_t)

        return TwoWeekAvailabilityModel(windows)

    def next_available(self, resource: str, t: datetime) -> datetime:
        """
        Return earliest datetime >= t where resource is within its availability window.
        If no window exists for that day/parity, search forward day-by-day.
        """
        resource = str(resource)

        # search forward up to, say, 30 days to avoid infinite loops on sparse resources
        for day_offset in range(0, 30):
            d = t.date() + timedelta(days=day_offset)
            cand = datetime.combine(d, t.time() if day_offset == 0 else time(0, 0, 0))

            iso_week = cand.isocalendar().week
            wp = int(iso_week % 2)
            wd = cand.weekday()

            w = self.windows.get((resource, wp, wd))
            if w is None:
                continue

            day_start = datetime.combine(d, w.start)
            day_end = datetime.combine(d, w.end)


            if cand <= day_start:
                return day_start
            if day_start <= cand < day_end:
                return cand

            # cand is after window -> try next day
            continue

        # fallback: if we never find availability, just return t (or raise)
        return t
