"""
Task 1.4: Next Activity Prediction for XOR Gateways (Process-Model-Aware)

This module implements:
- Basic: Branching probabilities learned from event log, CONSTRAINED by process model
- Advanced: ML model that takes trace history into account

Key improvement: During training, only learns transitions that are VALID according
to the provided process model. This ensures no "impossible" transitions are predicted.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# EXPERT ACTIVITY PREDICTOR CLASS (Process-Model-Aware)
# ============================================================

class ExpertActivityPredictor:
    """
    Task 1.4 Next Activity Predictor (Process-Model-Aware)

    Key feature: Learns transitions ONLY for valid process model edges.
    This prevents predicting activities that cannot follow the current activity.

    Basic mode:
        - Learns P(next | current_activity, preceding k activities) from event log
        - At XOR gateways, samples next activity based on learned probabilities
        - ONLY considers transitions that are valid per process model

    Advanced mode:
        - Uses ML model (GradientBoosting) with contextual features
        - Takes timestamp, hour, weekday, elapsed time into account
    """

    def __init__(
        self,
        mode: str = "basic",
        window_size: int = 5,
        basic_context_k: int = 2,
        smoothing_alpha: float = 1.0,
        min_probability: float = 0.0,
        random_state: int = 42,
        process_model: Optional[Dict[str, List[str]]] = None,
        gateways: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            mode: 'basic' or 'advanced'
            window_size: History window for advanced ML features
            basic_context_k: Max k for k-gram context (basic mode)
            smoothing_alpha: Laplace smoothing parameter
            min_probability: Filter out activities with prob < this
            random_state: Random seed for reproducibility
            process_model: Dict mapping activity -> list of valid successors
            gateways: Dict mapping activity -> gateway type ('xor', 'or', etc.)
        """
        self.mode = mode
        self.window_size = int(window_size)
        self.basic_k = int(basic_context_k)
        self.alpha = float(smoothing_alpha)
        self.min_probability = float(min_probability)
        self.rng = np.random.default_rng(random_state)

        # Process model for constraining transitions
        self.process_model = process_model or {}
        self.gateways = gateways or {}

        # Basic mode: Per-activity transition probabilities
        # transition_counts[current_activity][context_tuple][next_activity] = count
        self.transition_counts: Dict[str, Dict[int, Dict[tuple, Dict[str, int]]]] = {}
        # transition_probs[current_activity][context_tuple] = {next: prob}
        self.transition_probs: Dict[str, Dict[int, Dict[tuple, Dict[str, float]]]] = {}

        # Global fallback (without current activity)
        self.counts_by_ctx = {k: defaultdict(lambda: defaultdict(int))
                              for k in range(1, self.basic_k + 1)}
        self.probs_by_ctx = {k: {} for k in range(1, self.basic_k + 1)}
        self.global_next_counts = defaultdict(int)
        self.global_next_probs = {}

        # Advanced mode: ML model
        self.activity_encoder = LabelEncoder()
        self.activities_set = set()
        self.model = None

    # =========================================================
    # FIT METHODS
    # =========================================================

    def fit(self, df: pd.DataFrame, process_model: Optional[Dict] = None,
            gateways: Optional[Dict] = None) -> 'ExpertActivityPredictor':
        """
        Fit the predictor on historical event log data.

        Args:
            df: DataFrame with columns: case_id, activity, timestamp
            process_model: Optional process model (overrides constructor)
            gateways: Optional gateways (overrides constructor)

        Returns:
            self (for method chaining)
        """
        if process_model is not None:
            self.process_model = process_model
        if gateways is not None:
            self.gateways = gateways

        # Ensure proper column names
        df = self._normalize_columns(df)
        df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

        # Fit encoder for all activities
        self.activity_encoder.fit(df["activity"].astype(str).unique())
        self.activities_set = set(self.activity_encoder.classes_.tolist())

        # Fit basic k-gram model (process-model-aware)
        self._fit_basic_kgram_process_aware(df)

        # Also fit global fallback
        self._fit_global_fallback(df)

        # Optionally fit advanced ML model
        if self.mode == "advanced":
            self._fit_advanced_ml(df)

        return self

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has case_id, activity, timestamp columns."""
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

    def _fit_basic_kgram_process_aware(self, df: pd.DataFrame):
        """
        Learn transition probabilities that respect the process model.

        For each activity, learns P(next | context) ONLY for valid successors.
        """
        # Initialize structures
        self.transition_counts = {}
        self.transition_probs = {}

        # Count transitions per (current_activity, context) -> next
        for case_id, group in df.groupby("case_id", sort=False):
            activities = group["activity"].astype(str).tolist()

            if len(activities) < 2:
                continue

            for i in range(len(activities) - 1):
                current_act = activities[i]
                next_act = activities[i + 1]

                # Check if this transition is valid in process model
                if self.process_model:
                    valid_successors = self.process_model.get(current_act, [])
                    if valid_successors and next_act not in valid_successors:
                        # Skip this transition - it's not valid per process model
                        continue

                # Initialize structure for this activity if needed
                if current_act not in self.transition_counts:
                    self.transition_counts[current_act] = {
                        k: defaultdict(lambda: defaultdict(int))
                        for k in range(1, self.basic_k + 1)
                    }

                # Count for each context length k
                for k in range(1, self.basic_k + 1):
                    # Context is the k activities BEFORE current (not including current)
                    start_idx = i - k
                    if start_idx < 0:
                        # Use shorter context
                        context = tuple(activities[0:i]) if i > 0 else ()
                    else:
                        context = tuple(activities[start_idx:i])

                    if len(context) == k or (len(context) < k and len(context) == i):
                        # Valid context
                        self.transition_counts[current_act][k][context][next_act] += 1

        # Convert counts to probabilities
        for activity, k_dict in self.transition_counts.items():
            self.transition_probs[activity] = {}

            for k, ctx_dict in k_dict.items():
                self.transition_probs[activity][k] = {}

                for context, next_counts in ctx_dict.items():
                    self.transition_probs[activity][k][context] = self._compute_smoothed_probs(
                        next_counts,
                        valid_successors=self.process_model.get(activity, [])
                    )

    def _fit_global_fallback(self, df: pd.DataFrame):
        """Fit global fallback probabilities (legacy k-gram without current activity)."""
        # Reset
        for k in self.counts_by_ctx:
            self.counts_by_ctx[k] = defaultdict(lambda: defaultdict(int))
            self.probs_by_ctx[k] = {}
        self.global_next_counts = defaultdict(int)

        for case_id, group in df.groupby("case_id", sort=False):
            activities = group["activity"].astype(str).tolist()

            if len(activities) < 2:
                continue

            for i in range(len(activities) - 1):
                next_act = activities[i + 1]
                self.global_next_counts[next_act] += 1

                for k in range(1, self.basic_k + 1):
                    start_idx = i - k + 1
                    if start_idx < 0:
                        continue
                    context = tuple(activities[start_idx : i + 1])
                    self.counts_by_ctx[k][context][next_act] += 1

        # Convert to probs
        for k in range(1, self.basic_k + 1):
            for context, next_counts in self.counts_by_ctx[k].items():
                self.probs_by_ctx[k][context] = self._compute_smoothed_probs(next_counts)

        self.global_next_probs = self._compute_smoothed_probs(self.global_next_counts)

    def _compute_smoothed_probs(self, next_counts: Dict[str, int],
                                 valid_successors: List[str] = None) -> Dict[str, float]:
        """
        Apply Laplace smoothing to convert counts to probabilities.

        If valid_successors provided, smooths over those activities.
        Otherwise, smooths over observed activities.
        """
        if not next_counts and not valid_successors:
            return {}

        # Determine activities to include
        if valid_successors:
            activities = list(set(valid_successors))
        else:
            activities = list(next_counts.keys())

        if not activities:
            return {}

        num_activities = len(activities)
        total = sum(next_counts.get(a, 0) for a in activities)
        denominator = total + self.alpha * num_activities

        if denominator <= 0:
            # Uniform if no data
            return {a: 1.0 / num_activities for a in activities}

        return {
            act: (next_counts.get(act, 0) + self.alpha) / denominator
            for act in activities
        }

    def _fit_advanced_ml(self, df: pd.DataFrame):
        """Train ML model with contextual features for advanced prediction."""
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

                history = acts[max(0, curr_idx - self.window_size + 1) : curr_idx + 1]
                if len(history) < self.window_size:
                    pad = np.full(self.window_size - len(history), -1)
                    ngram_feat = np.concatenate([pad, history])
                else:
                    ngram_feat = history

                context_feat = [hours[curr_idx], wdays[curr_idx], elapsed[curr_idx]]
                feature_row = np.concatenate([ngram_feat, context_feat])

                X_all.append(feature_row)
                y_all.append(target)
                groups.append(case_id)

        if not X_all:
            return

        X = np.array(X_all)
        y = np.array(y_all)
        groups = np.array(groups)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))

        self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.model.fit(X[train_idx], y[train_idx])

        val_acc = self.model.score(X[val_idx], y[val_idx])
        print(f"[Advanced Model] Validation accuracy: {val_acc:.3f}")

    # =========================================================
    # PREDICTION METHODS
    # =========================================================

    def get_next_activity_distribution(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None,
        current_activity: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Get probability distribution over possible next activities.

        PROCESS-MODEL-AWARE: Uses per-activity transition tables.

        Args:
            prefix_activities: List of activities executed so far in the trace
            enabled_next: If provided (XOR outgoing), restrict distribution to these
            current_activity: The current activity (last in trace). If None, inferred from prefix.

        Returns:
            Dictionary mapping activity names to probabilities
        """
        if not prefix_activities:
            return {}

        prefix_activities = [str(a) for a in prefix_activities]

        # Determine current activity
        if current_activity is None:
            current_activity = prefix_activities[-1]
        current_activity = str(current_activity)

        # Get valid successors from process model
        if enabled_next is not None:
            valid_next = [str(a) for a in enabled_next]
        elif self.process_model and current_activity in self.process_model:
            valid_next = [str(a) for a in self.process_model[current_activity]]
        else:
            valid_next = None  # No constraint

        # Try to get distribution from per-activity table
        if current_activity in self.transition_probs:
            dist = self._get_dist_for_activity(current_activity, prefix_activities, valid_next)
            if dist:
                return dist

        # Fallback to global k-gram
        return self._get_global_fallback_dist(prefix_activities, valid_next)

    def _get_dist_for_activity(
        self,
        current_activity: str,
        prefix: List[str],
        valid_next: Optional[List[str]]
    ) -> Dict[str, float]:
        """Get distribution from per-activity transition table."""
        activity_probs = self.transition_probs.get(current_activity, {})

        # Try different context lengths (longest first)
        for k in range(min(self.basic_k, len(prefix) - 1), 0, -1):
            probs_k = activity_probs.get(k, {})

            # Build context (k activities before current)
            if len(prefix) > k:
                context = tuple(prefix[-(k+1):-1])
            else:
                context = tuple(prefix[:-1])

            if context in probs_k:
                dist = dict(probs_k[context])

                # Filter to valid_next if provided
                if valid_next is not None:
                    dist = self._filter_and_smooth(dist, valid_next)

                return dist

        # No matching context found
        return {}

    def _get_global_fallback_dist(
        self,
        prefix: List[str],
        valid_next: Optional[List[str]]
    ) -> Dict[str, float]:
        """Get distribution from global k-gram fallback."""
        best_counts = None

        for k in range(min(self.basic_k, len(prefix)), 0, -1):
            context = tuple(prefix[-k:])
            counts_dict = self.counts_by_ctx[k].get(context)
            if counts_dict:
                best_counts = dict(counts_dict)
                break

        if best_counts is None:
            best_counts = dict(self.global_next_counts)

        if valid_next is not None:
            return self._filter_and_smooth(best_counts, valid_next, is_counts=True)

        return self._compute_smoothed_probs(best_counts)

    def _filter_and_smooth(
        self,
        dist_or_counts: Dict[str, float],
        valid_next: List[str],
        is_counts: bool = False
    ) -> Dict[str, float]:
        """Filter distribution to valid activities and re-normalize with smoothing."""
        if not valid_next:
            return {}

        valid_next = [str(a) for a in valid_next]

        if is_counts:
            # Laplace smooth over valid activities
            tmp = {}
            for act in valid_next:
                count = float(dist_or_counts.get(act, 0))
                tmp[act] = count + self.alpha
        else:
            # Already probabilities - just filter and re-smooth
            tmp = {}
            for act in valid_next:
                # Use existing prob as pseudo-count
                p = dist_or_counts.get(act, 0.0)
                tmp[act] = p + self.alpha

        total = sum(tmp.values())
        if total <= 0:
            return {a: 1.0 / len(valid_next) for a in valid_next}

        return {a: v / total for a, v in tmp.items()}

    def sample_next_activity(
        self,
        prefix_activities: List[str],
        enabled_next: Optional[List[str]] = None,
        current_activity: Optional[str] = None,
    ) -> Optional[str]:
        """
        Sample next activity from the probability distribution.

        IMPORTANT: Always respects process model constraints.

        Args:
            prefix_activities: List of activities executed so far
            enabled_next: If provided, only sample from these (XOR outgoing arcs)
            current_activity: Current activity (defaults to last in prefix)

        Returns:
            Sampled activity name, or None if no valid options
        """
        dist = self.get_next_activity_distribution(
            prefix_activities,
            enabled_next=enabled_next,
            current_activity=current_activity
        )

        if not dist:
            # Fallback: uniform over enabled_next
            if enabled_next:
                return self.rng.choice([str(a) for a in enabled_next])
            return None

        activities = list(dist.keys())
        probabilities = np.array([dist[a] for a in activities], dtype=float)
        probabilities = probabilities / probabilities.sum()

        return self.rng.choice(activities, p=probabilities)

    def predict_next_activity(
        self,
        case_prefix_activities: List[str],
        current_timestamp=None,
        case_start_timestamp=None,
        enabled_next: Optional[List[str]] = None,
        return_distribution: bool = False,
        current_activity: Optional[str] = None,
    ):
        """
        Unified prediction API for both basic and advanced modes.

        Args:
            case_prefix_activities: Activities executed so far in the case
            current_timestamp: Current simulation time (for advanced mode)
            case_start_timestamp: Case start time (for advanced mode)
            enabled_next: Allowed next activities (XOR outgoing)
            return_distribution: If True, return dict of probabilities
            current_activity: Current activity (defaults to last in prefix)

        Returns:
            If return_distribution: Dict[str, float]
            Otherwise: sampled activity name (str)
        """
        if self.mode == "basic":
            if return_distribution:
                return self.get_next_activity_distribution(
                    case_prefix_activities,
                    enabled_next=enabled_next,
                    current_activity=current_activity
                )
            return self.sample_next_activity(
                case_prefix_activities,
                enabled_next=enabled_next,
                current_activity=current_activity
            )

        # Advanced mode uses ML model
        return self._predict_advanced(
            case_prefix_activities,
            current_timestamp,
            case_start_timestamp,
            enabled_next=enabled_next
        )

    def _predict_advanced(
        self,
        prefix_activities: List[str],
        current_timestamp=None,
        case_start_timestamp=None,
        enabled_next: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Advanced ML-based prediction."""
        if self.model is None:
            # Fallback to basic
            return self.sample_next_activity(prefix_activities, enabled_next=enabled_next)

        def map_to_idx(act: str) -> int:
            if act in self.activities_set:
                return self.activity_encoder.transform([act])[0]
            return -1

        encoded = np.array([map_to_idx(str(a)) for a in prefix_activities], dtype=int)

        if len(encoded) < self.window_size:
            pad = np.full(self.window_size - len(encoded), -1)
            ngram_feat = np.concatenate([pad, encoded])
        else:
            ngram_feat = encoded[-self.window_size:]

        if current_timestamp is None:
            hour, wday, elapsed = 12, 2, 0.0
        else:
            hour = current_timestamp.hour
            wday = current_timestamp.weekday()
            elapsed = 0.0
            if case_start_timestamp is not None:
                elapsed = (current_timestamp - case_start_timestamp).total_seconds()

        context_feat = np.array([hour, wday, elapsed], dtype=float)
        features = np.concatenate([ngram_feat, context_feat]).reshape(1, -1)

        pred_idx = self.model.predict(features)[0]
        predicted = self.activity_encoder.inverse_transform([pred_idx])[0]

        # Check if prediction is valid
        if enabled_next and predicted not in enabled_next:
            # Prediction invalid - fallback to basic sampling
            return self.sample_next_activity(prefix_activities, enabled_next=enabled_next)

        return predicted

    # =========================================================
    # SERIALIZATION METHODS
    # =========================================================

    def save_to_json(
        self,
        output_dir: str,
        filename_prefix: str = "next_activity_probs",
        decision_points_only: bool = True
    ) -> Dict[str, str]:
        """Save learned probabilities to JSON files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def ctx_to_key(ctx: tuple) -> str:
            return " || ".join(ctx) if ctx else "<empty>"

        outputs = {}

        # Save per-activity transition probs
        activity_probs_json = {}
        for activity, k_dict in self.transition_probs.items():
            activity_probs_json[activity] = {}
            for k, ctx_probs in k_dict.items():
                layer = {}
                for ctx, probs in ctx_probs.items():
                    if decision_points_only and len(probs) < 2:
                        continue
                    layer[ctx_to_key(ctx)] = probs
                if layer:
                    activity_probs_json[activity][str(k)] = layer

        probs_path = Path(output_dir) / f"{filename_prefix}_activity_probs.json"
        with open(probs_path, "w", encoding="utf-8") as f:
            json.dump(activity_probs_json, f, ensure_ascii=False, indent=2)
        outputs["activity_probs"] = str(probs_path)

        # Save global probs (fallback)
        global_path = Path(output_dir) / f"{filename_prefix}_global_probs.json"
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(self.global_next_probs, f, ensure_ascii=False, indent=2)
        outputs["global_probs"] = str(global_path)

        # Save process model if available
        if self.process_model:
            model_path = Path(output_dir) / f"{filename_prefix}_process_model.json"
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump(self.process_model, f, ensure_ascii=False, indent=2)
            outputs["process_model"] = str(model_path)

        return outputs

    def load_from_json(self, activity_probs_path: str, global_path: str,
                       process_model_path: str = None):
        """Load previously saved probabilities from JSON files."""
        def key_to_ctx(key: str) -> tuple:
            if key == "<empty>":
                return ()
            return tuple(key.split(" || "))

        with open(activity_probs_path, "r", encoding="utf-8") as f:
            activity_probs_json = json.load(f)

        with open(global_path, "r", encoding="utf-8") as f:
            self.global_next_probs = json.load(f)

        if process_model_path:
            with open(process_model_path, "r", encoding="utf-8") as f:
                self.process_model = json.load(f)

        # Reconstruct transition_probs
        self.transition_probs = {}
        for activity, k_dict in activity_probs_json.items():
            self.transition_probs[activity] = {}
            for k_str, ctx_probs in k_dict.items():
                k = int(k_str)
                self.transition_probs[activity][k] = {}
                for ctx_key, probs in ctx_probs.items():
                    ctx = key_to_ctx(ctx_key)
                    self.transition_probs[activity][k][ctx] = probs


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def train_predictor_from_log(
    log_df: pd.DataFrame,
    process_model: Dict[str, List[str]] = None,
    gateways: Dict[str, str] = None,
    mode: str = "basic",
    context_k: int = 2,
    **kwargs
) -> ExpertActivityPredictor:
    """
    Convenience function to train a predictor from an event log DataFrame.

    Args:
        log_df: DataFrame with case_id/case:concept:name, activity/concept:name, timestamp
        process_model: Dict mapping activity -> list of valid successors
        gateways: Dict mapping activity -> gateway type
        mode: 'basic' or 'advanced'
        context_k: Context window size for k-gram (basic mode)
        **kwargs: Additional predictor parameters

    Returns:
        Trained ExpertActivityPredictor
    """
    predictor = ExpertActivityPredictor(
        mode=mode,
        basic_context_k=context_k,
        process_model=process_model,
        gateways=gateways,
        **kwargs
    )
    predictor.fit(log_df)
    return predictor


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    print("Task 1.4: Next Activity Predictor (Process-Model-Aware)")
    print("=" * 60)

    # Define a simple process model
    process_model = {
        "A": ["B", "C"],  # XOR: A can go to B or C
        "B": ["D"],       # Sequence: B -> D
        "C": ["D"],       # Sequence: C -> D
        "D": ["END"],
    }

    # Create sample training data
    sample_data = pd.DataFrame({
        "case_id": ["C1"]*4 + ["C2"]*4 + ["C3"]*4 + ["C4"]*4 + ["C5"]*4,
        "activity": [
            "A", "B", "D", "END",  # A->B path
            "A", "B", "D", "END",  # A->B path
            "A", "B", "D", "END",  # A->B path
            "A", "C", "D", "END",  # A->C path
            "A", "C", "D", "END",  # A->C path
        ],
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="h")
    })

    print("\nTraining data: 3 cases with A->B, 2 cases with A->C")
    print("Expected P(B|A) = 60%, P(C|A) = 40%")

    # Train predictor with process model
    predictor = ExpertActivityPredictor(
        mode="basic",
        basic_context_k=2,
        process_model=process_model
    )
    predictor.fit(sample_data)

    # Test prediction at activity A
    print("\nPredicting from activity A:")
    dist = predictor.get_next_activity_distribution(
        ["A"],
        enabled_next=["B", "C"],
        current_activity="A"
    )
    print(f"P(next | current=A) = {dist}")

    # Verify it respects process model
    print("\nTrying to predict from A with invalid successor E:")
    dist_invalid = predictor.get_next_activity_distribution(
        ["A"],
        enabled_next=["B", "C", "E"],  # E is not in process model
        current_activity="A"
    )
    print(f"Distribution (E should get low/smoothed prob): {dist_invalid}")

    # Sample multiple times
    print("\nSampling 100 times from A:")
    from collections import Counter
    samples = [predictor.sample_next_activity(["A"], enabled_next=["B", "C"]) for _ in range(100)]
    print(f"Results: {Counter(samples)}")

    print("\n" + "=" * 60)
    print("Module ready for integration with simulation_engine_core.py")