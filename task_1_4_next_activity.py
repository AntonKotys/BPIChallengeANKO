"""
Task 1.4 - Next Activity Prediction for XOR Gateway Branching

This module provides probability-based predictions for XOR gateway decisions
in business process simulation, learned from historical event logs.

KEY FEATURES:
1. Uses trace history (k-gram context) to predict next activity
2. ALWAYS respects enabled_next constraint (only predicts valid outgoing activities)
3. Basic mode: k-gram probabilities with Laplace smoothing
4. Advanced mode: ML-based with trace history features (also respects enabled_next)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


class ExpertActivityPredictor:
    """
    Task 1.4: Activity predictor for XOR gateway branching.

    CRITICAL: This predictor ALWAYS:
    1. Uses trace history (preceding activities) to make predictions
    2. Restricts predictions to enabled_next (valid outgoing activities from PROCESS_MODEL)

    Modes:
    - basic: k-gram probabilities learned from event log
    - advanced: ML model with trace + temporal features
    """

    def __init__(
        self,
        mode: str = "basic",
        window_size: int = 5,
        basic_context_k: int = 2,
        smoothing_alpha: float = 1.0,
        min_probability: float = 0.0,
        random_state: int = 42,
    ):
        self.mode = mode
        self.window_size = int(window_size)
        self.basic_k = int(basic_context_k)
        self.alpha = float(smoothing_alpha)
        self.min_probability = float(min_probability)
        self.rng = np.random.default_rng(random_state)

        # --- BASIC MODE DATA STRUCTURES ---
        # counts_by_ctx[k][ctx_tuple][next] = count
        self.counts_by_ctx = {k: defaultdict(lambda: defaultdict(int)) for k in range(1, self.basic_k + 1)}
        # probs_by_ctx[k][ctx_tuple] = {next: prob}
        self.probs_by_ctx = {k: {} for k in range(1, self.basic_k + 1)}
        self.global_next_counts = defaultdict(int)
        self.global_next_probs = {}

        # --- ADVANCED MODE DATA STRUCTURES ---
        self.activity_encoder = LabelEncoder()
        self.activities_set = set()
        self.model = None

    # =========================================================================
    # FIT METHODS
    # =========================================================================

    def fit(self, df: pd.DataFrame):
        """
        Train the predictor from historical event log.

        Args:
            df: DataFrame with columns: case_id, activity, timestamp
        """
        # Standardize column names
        df = self._standardize_columns(df)
        df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

        # Fit encoder
        self.activity_encoder.fit(df["activity"].astype(str).unique())
        self.activities_set = set(self.activity_encoder.classes_.tolist())

        # Always fit basic k-gram (used as fallback for advanced too)
        self._fit_basic_kgram(df)

        # Fit advanced model if requested
        if self.mode == "advanced":
            self._fit_advanced_with_context(df)

        return self

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to case_id, activity, timestamp."""
        df = df.copy()
        rename_map = {
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp'
        }
        for old_name, new_name in rename_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        return df

    def _fit_basic_kgram(self, df: pd.DataFrame):
        """Learn k-gram transition probabilities from event log."""
        # Reset
        for k in self.counts_by_ctx:
            self.counts_by_ctx[k].clear()
            self.probs_by_ctx[k].clear()
        self.global_next_counts.clear()
        self.global_next_probs = {}

        # Count transitions
        for _, group in df.groupby("case_id", sort=False):
            acts = group["activity"].astype(str).tolist()
            if len(acts) < 2:
                continue

            for i in range(len(acts) - 1):
                nxt = acts[i + 1]
                self.global_next_counts[nxt] += 1

                # Contexts of length 1..k
                for k in range(1, self.basic_k + 1):
                    start = i - k + 1
                    if start < 0:
                        continue
                    ctx = tuple(acts[start: i + 1])
                    self.counts_by_ctx[k][ctx][nxt] += 1

        # Convert counts to probabilities (for each context observed)
        for k in range(1, self.basic_k + 1):
            for ctx, next_counts in self.counts_by_ctx[k].items():
                self.probs_by_ctx[k][ctx] = self._smoothed_probs_over_outgoing(next_counts)

        self.global_next_probs = self._smoothed_probs_over_outgoing(self.global_next_counts)

    def _smoothed_probs_over_outgoing(self, next_counts: dict) -> dict:
        """Apply Laplace smoothing to counts."""
        outgoing = list(next_counts.keys())
        if not outgoing:
            return {}
        m = len(outgoing)
        total = sum(next_counts.values())
        denom = total + self.alpha * m
        return {a: (next_counts[a] + self.alpha) / denom for a in outgoing}

    def _fit_advanced_with_context(self, df: pd.DataFrame):
        """Fit ML model using trace history + temporal features."""
        df = df.copy()
        df["activity"] = df["activity"].astype(str)
        df["act_idx"] = self.activity_encoder.transform(df["activity"])
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.dayofweek
        df["case_start_time"] = df.groupby("case_id")["timestamp"].transform("min")
        df["elapsed_seconds"] = (df["timestamp"] - df["case_start_time"]).dt.total_seconds()

        X_all, y_all, groups = [], [], []

        for case_id, group in df.groupby("case_id", sort=False):
            acts = group["act_idx"].values
            hours = group["hour"].values
            wdays = group["weekday"].values
            elapsed = group["elapsed_seconds"].values

            for i in range(1, len(acts)):
                target = acts[i]
                curr_idx = i - 1

                # Build n-gram feature from trace history
                history_window = acts[max(0, curr_idx - self.window_size + 1): curr_idx + 1]
                if len(history_window) < self.window_size:
                    pad = np.full(self.window_size - len(history_window), -1)
                    ngram_feat = np.concatenate([pad, history_window])
                else:
                    ngram_feat = history_window

                # Temporal features
                context_feat = [hours[curr_idx], wdays[curr_idx], elapsed[curr_idx]]
                feature_row = np.concatenate([ngram_feat, context_feat])

                X_all.append(feature_row)
                y_all.append(target)
                groups.append(case_id)

        X = np.array(X_all)
        y = np.array(y_all)
        groups = np.array(groups)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))

        self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.model.fit(X[train_idx], y[train_idx])

    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================

    def get_next_activity_distribution(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get probability distribution over possible next activities.

        CRITICAL: If enabled_next is provided, the distribution is ONLY over
        those activities (enforcing process model constraints).

        Args:
            prefix_activities: List of activities executed so far in the case (trace history)
            enabled_next: List of valid next activities from PROCESS_MODEL (XOR outgoing arcs)

        Returns:
            Dictionary mapping activity -> probability
        """
        if not prefix_activities:
            # No history - use uniform over enabled or global
            if enabled_next:
                u = 1.0 / len(enabled_next)
                return {a: u for a in enabled_next}
            return self.global_next_probs.copy()

        prefix_activities = [str(a) for a in prefix_activities]

        # Find best context counts: try longest k first, then shorter, then global
        # This is the key to using trace history - we find the best matching k-gram
        best_counts = None
        matched_k = 0

        for k in range(min(self.basic_k, len(prefix_activities)), 0, -1):
            ctx = tuple(prefix_activities[-k:])
            counts_dict = self.counts_by_ctx[k].get(ctx)
            if counts_dict:
                best_counts = dict(counts_dict)  # Make a copy
                matched_k = k
                break

        if best_counts is None:
            best_counts = dict(self.global_next_counts)

        # CRITICAL: If enabled_next is provided, restrict distribution to those activities ONLY
        if enabled_next is not None:
            enabled_next = [str(a) for a in enabled_next]
            if not enabled_next:
                return {}

            # Compute Laplace-smoothed distribution OVER enabled_next only
            denom = 0.0
            tmp = {}
            for a in enabled_next:
                c = float(best_counts.get(a, 0))
                tmp[a] = c + self.alpha
                denom += tmp[a]

            if denom <= 0:
                # Uniform fallback
                u = 1.0 / len(enabled_next)
                return {a: u for a in enabled_next}

            dist = {a: v / denom for a, v in tmp.items()}
        else:
            # No constraint - use all observed next activities
            dist = self._smoothed_probs_over_outgoing(best_counts)

        # Apply min_probability filter
        if dist and self.min_probability > 0.0:
            filtered = {a: p for a, p in dist.items() if p >= self.min_probability}
            if filtered:
                dist = filtered

        # Renormalize
        if dist:
            s = sum(dist.values())
            if s > 0:
                dist = {a: p / s for a, p in dist.items()}

        return dist

    def sample_next_activity(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Sample the next activity based on learned probabilities.

        THIS IS THE MAIN METHOD CALLED BY THE SIMULATION ENGINE.

        CRITICAL GUARANTEES:
        1. Uses trace history (prefix_activities) to determine probabilities
        2. ONLY returns an activity from enabled_next (if provided)
        3. Never returns an impossible transition

        Args:
            prefix_activities: List of activities executed so far (trace history)
            enabled_next: List of valid next activities from PROCESS_MODEL

        Returns:
            Sampled activity name, or None if sampling fails
        """
        if self.mode == "basic":
            return self._sample_next_activity_basic(prefix_activities, enabled_next)
        else:
            return self._sample_next_activity_advanced(prefix_activities, enabled_next)

    def _sample_next_activity_basic(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Sample using k-gram probabilities (basic mode).

        Uses trace history to find best matching context, then samples
        from learned transition probabilities restricted to enabled_next.
        """
        dist = self.get_next_activity_distribution(prefix_activities, enabled_next=enabled_next)

        if not dist:
            if enabled_next:
                return self.rng.choice(enabled_next)
            return None

        activities = list(dist.keys())
        probs = np.array([dist[a] for a in activities], dtype=float)

        # Safety check
        if probs.sum() <= 0:
            if enabled_next:
                return self.rng.choice(enabled_next)
            return self.rng.choice(activities) if activities else None

        probs = probs / probs.sum()
        return self.rng.choice(activities, p=probs)

    def _sample_next_activity_advanced(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None,
        current_timestamp=None,
        case_start_timestamp=None
    ) -> Optional[str]:
        """
        Sample using ML model (advanced mode).

        Uses trace history + temporal features, then restricts to enabled_next.
        """
        if self.model is None or not enabled_next:
            return self._sample_next_activity_basic(prefix_activities, enabled_next)

        # Get ML model's probability distribution
        dist = self._get_advanced_distribution(prefix_activities, current_timestamp, case_start_timestamp)

        if not dist:
            return self._sample_next_activity_basic(prefix_activities, enabled_next)

        # Filter to only enabled_next activities
        enabled_set = set(str(a) for a in enabled_next)
        filtered = {a: p for a, p in dist.items() if a in enabled_set}

        # If ML model has no predictions for enabled activities, use Laplace smoothing
        if not filtered:
            for a in enabled_next:
                filtered[a] = self.alpha / 100.0

        if not filtered:
            return self._sample_next_activity_basic(prefix_activities, enabled_next)

        # Renormalize and sample
        total = sum(filtered.values())
        if total <= 0:
            return self.rng.choice(enabled_next)

        acts = list(filtered.keys())
        probs = np.array([filtered[a] / total for a in acts], dtype=float)
        return self.rng.choice(acts, p=probs)

    def _get_advanced_distribution(
        self,
        prefix_activities: List[str],
        current_timestamp=None,
        case_start_timestamp=None
    ) -> Dict[str, float]:
        """Get probability distribution from ML model."""
        if self.model is None:
            return {}

        # Map activities to indices
        def map_known(a: str) -> int:
            try:
                return self.activity_encoder.transform([a])[0]
            except:
                return -1

        encoded_hist = np.array([map_known(str(x)) for x in prefix_activities], dtype=int)

        # Build n-gram feature
        if len(encoded_hist) < self.window_size:
            pad = np.full(self.window_size - len(encoded_hist), -1)
            ngram_feat = np.concatenate([pad, encoded_hist])
        else:
            ngram_feat = encoded_hist[-self.window_size:]

        # Temporal features
        if current_timestamp is None:
            hour, wday, elapsed = 12, 2, 0
        else:
            hour = current_timestamp.hour
            wday = current_timestamp.weekday()
            elapsed = 0
            if case_start_timestamp is not None:
                elapsed = (current_timestamp - case_start_timestamp).total_seconds()

        context_feat = np.array([hour, wday, elapsed], dtype=float)
        features = np.concatenate([ngram_feat, context_feat]).reshape(1, -1)

        # Get class probabilities
        proba = self.model.predict_proba(features)[0]
        classes = self.model.classes_

        # Convert to activity names
        dist = {}
        for idx, prob in zip(classes, proba):
            act_name = self.activity_encoder.inverse_transform([idx])[0]
            dist[act_name] = prob

        return dist

    # =========================================================================
    # LEGACY API (for backward compatibility)
    # =========================================================================

    def sample_next_activity_basic(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None
    ) -> Optional[str]:
        """Legacy method name - redirects to _sample_next_activity_basic."""
        return self._sample_next_activity_basic(prefix_activities, enabled_next)

    def predict_next_activity(
        self,
        case_prefix_activities: List[str],
        current_timestamp=None,
        case_start_timestamp=None,
        enabled_next: Optional[List[str]] = None,
        return_distribution: bool = False,
    ):
        """
        Unified prediction API.
        """
        if return_distribution:
            return self.get_next_activity_distribution(case_prefix_activities, enabled_next=enabled_next)
        return self.sample_next_activity(case_prefix_activities, enabled_next=enabled_next)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def save_basic_probs(
        self,
        output_dir: str,
        filename_prefix: str = "branch_probs",
        decision_points_only: bool = True
    ) -> dict:
        """Save learned probabilities to JSON files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def ctx_to_key(ctx_tuple):
            return " || ".join(ctx_tuple)

        out = {}

        # Save counts
        counts_json = {}
        for k in range(1, self.basic_k + 1):
            layer = {}
            for ctx, next_counts in self.counts_by_ctx[k].items():
                if decision_points_only and len(next_counts.keys()) < 2:
                    continue
                layer[ctx_to_key(ctx)] = dict(next_counts)
            counts_json[str(k)] = layer

        p_counts = Path(output_dir) / f"{filename_prefix}_counts_k{self.basic_k}.json"
        with open(p_counts, "w", encoding="utf-8") as f:
            json.dump(counts_json, f, ensure_ascii=False, indent=2)
        out["counts"] = str(p_counts)

        # Save probs
        probs_json = {}
        for k in range(1, self.basic_k + 1):
            layer = {}
            for ctx, next_probs in self.probs_by_ctx[k].items():
                if decision_points_only:
                    cnt = self.counts_by_ctx[k].get(ctx, {})
                    if len(cnt.keys()) < 2:
                        continue
                layer[ctx_to_key(ctx)] = dict(next_probs)
            probs_json[str(k)] = layer

        p_probs = Path(output_dir) / f"{filename_prefix}_probs_k{self.basic_k}.json"
        with open(p_probs, "w", encoding="utf-8") as f:
            json.dump(probs_json, f, ensure_ascii=False, indent=2)
        out["probs"] = str(p_probs)

        # Save global probs
        p_global = Path(output_dir) / f"{filename_prefix}_global_probs.json"
        with open(p_global, "w", encoding="utf-8") as f:
            json.dump(self.global_next_probs, f, ensure_ascii=False, indent=2)
        out["global_probs"] = str(p_global)

        print(f"[Task 1.4] Saved probability files to {output_dir}")
        return out

    def load_basic_probs(self, counts_path: str, global_probs_path: str):
        """Load probabilities from JSON files."""
        with open(counts_path, "r", encoding="utf-8") as f:
            counts_json = json.load(f)

        with open(global_probs_path, "r", encoding="utf-8") as f:
            self.global_next_probs = json.load(f)

        # Rebuild counts_by_ctx
        for k_str, layer in counts_json.items():
            k = int(k_str)
            if k not in self.counts_by_ctx:
                self.counts_by_ctx[k] = defaultdict(lambda: defaultdict(int))
            for ctx_key, next_counts in layer.items():
                ctx = tuple(ctx_key.split(" || "))
                for nxt, count in next_counts.items():
                    self.counts_by_ctx[k][ctx][nxt] = count

        # Rebuild probs_by_ctx
        for k in self.counts_by_ctx:
            self.probs_by_ctx[k] = {}
            for ctx, next_counts in self.counts_by_ctx[k].items():
                self.probs_by_ctx[k][ctx] = self._smoothed_probs_over_outgoing(next_counts)

        return self