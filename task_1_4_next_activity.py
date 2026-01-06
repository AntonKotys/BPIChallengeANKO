"""
Task 1.4: Next Activity Prediction for XOR Gateways

This module implements:
- Basic: Branching probabilities learned from event log (k-gram based)
- Advanced: ML model that takes trace history into account

Integration with simulation_engine_core.py:
- The ExpertActivityPredictor is used in route_next() for XOR gateways
- Instead of random.choice(), we sample based on learned probabilities
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
# EXPERT ACTIVITY PREDICTOR CLASS
# ============================================================

class ExpertActivityPredictor:
    """
    Task 1.4 Next Activity Predictor

    Basic mode:
        - Learns P(next | preceding k activities) from event log
        - At XOR gateways, samples next activity based on learned probabilities
        - Falls back to shorter contexts if k-gram not seen

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
    ):
        """
        Args:
            mode: 'basic' or 'advanced'
            window_size: History window for advanced ML features
            basic_context_k: Max k for k-gram context (basic mode)
            smoothing_alpha: Laplace smoothing parameter
            min_probability: Filter out activities with prob < this
            random_state: Random seed for reproducibility
        """
        self.mode = mode
        self.window_size = int(window_size)
        self.basic_k = int(basic_context_k)
        self.alpha = float(smoothing_alpha)
        self.min_probability = float(min_probability)
        self.rng = np.random.default_rng(random_state)

        # Basic mode: k-gram transition counts and probabilities
        # counts_by_ctx[k][ctx_tuple][next_activity] = count
        self.counts_by_ctx = {k: defaultdict(lambda: defaultdict(int))
                              for k in range(1, self.basic_k + 1)}
        # probs_by_ctx[k][ctx_tuple] = {next_activity: probability}
        self.probs_by_ctx = {k: {} for k in range(1, self.basic_k + 1)}

        # Global fallback probabilities
        self.global_next_counts = defaultdict(int)
        self.global_next_probs = {}

        # Advanced mode: ML model
        self.activity_encoder = LabelEncoder()
        self.activities_set = set()
        self.model = None

    # =========================================================
    # FIT METHODS
    # =========================================================

    def fit(self, df: pd.DataFrame) -> 'ExpertActivityPredictor':
        """
        Fit the predictor on historical event log data.

        Args:
            df: DataFrame with columns: case_id, activity, timestamp

        Returns:
            self (for method chaining)
        """
        # Ensure proper column names
        df = self._normalize_columns(df)
        df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

        # Fit encoder for all activities
        self.activity_encoder.fit(df["activity"].astype(str).unique())
        self.activities_set = set(self.activity_encoder.classes_.tolist())

        # Always fit basic k-gram model
        self._fit_basic_kgram(df)

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

    def _fit_basic_kgram(self, df: pd.DataFrame):
        """
        Learn k-gram transition probabilities from the event log.

        For each trace, counts transitions:
        P(next_activity | last_k_activities)
        """
        # Reset counts
        for k in self.counts_by_ctx:
            self.counts_by_ctx[k] = defaultdict(lambda: defaultdict(int))
            self.probs_by_ctx[k] = {}
        self.global_next_counts = defaultdict(int)
        self.global_next_probs = {}

        # Count all transitions
        for case_id, group in df.groupby("case_id", sort=False):
            activities = group["activity"].astype(str).tolist()

            if len(activities) < 2:
                continue

            for i in range(len(activities) - 1):
                next_act = activities[i + 1]
                self.global_next_counts[next_act] += 1

                # Count for each context length k = 1, 2, ..., basic_k
                for k in range(1, self.basic_k + 1):
                    start_idx = i - k + 1
                    if start_idx < 0:
                        continue

                    # Context is the last k activities (including current)
                    context = tuple(activities[start_idx : i + 1])
                    self.counts_by_ctx[k][context][next_act] += 1

        # Convert counts to probabilities with Laplace smoothing
        for k in range(1, self.basic_k + 1):
            for context, next_counts in self.counts_by_ctx[k].items():
                self.probs_by_ctx[k][context] = self._compute_smoothed_probs(next_counts)

        self.global_next_probs = self._compute_smoothed_probs(self.global_next_counts)

    def _compute_smoothed_probs(self, next_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Apply Laplace smoothing to convert counts to probabilities.
        Only smooths over activities that have been observed for this context.
        """
        if not next_counts:
            return {}

        activities = list(next_counts.keys())
        num_activities = len(activities)
        total = sum(next_counts.values())
        denominator = total + self.alpha * num_activities

        return {
            act: (next_counts[act] + self.alpha) / denominator
            for act in activities
        }

    def _fit_advanced_ml(self, df: pd.DataFrame):
        """
        Train ML model with contextual features for advanced prediction.
        Features: activity n-gram + hour + weekday + elapsed time
        """
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

                # Build n-gram feature (padded with -1 if needed)
                history = acts[max(0, curr_idx - self.window_size + 1) : curr_idx + 1]
                if len(history) < self.window_size:
                    pad = np.full(self.window_size - len(history), -1)
                    ngram_feat = np.concatenate([pad, history])
                else:
                    ngram_feat = history

                # Context features
                context_feat = [hours[curr_idx], wdays[curr_idx], elapsed[curr_idx]]
                feature_row = np.concatenate([ngram_feat, context_feat])

                X_all.append(feature_row)
                y_all.append(target)
                groups.append(case_id)

        X = np.array(X_all)
        y = np.array(y_all)
        groups = np.array(groups)

        # Train/validation split by case
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))

        self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.model.fit(X[train_idx], y[train_idx])

        # Report validation accuracy
        val_acc = self.model.score(X[val_idx], y[val_idx])
        print(f"[Advanced Model] Validation accuracy: {val_acc:.3f}")

    # =========================================================
    # PREDICTION METHODS
    # =========================================================

    def get_next_activity_distribution(
            self,
            prefix_activities: List[str],
            enabled_next: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get probability distribution over possible next activities.

        Args:
            prefix_activities: List of activities executed so far in the trace
            enabled_next: If provided (XOR outgoing), restrict distribution to these

        Returns:
            Dictionary mapping activity names to probabilities
        """
        if not prefix_activities:
            return {}

        prefix_activities = [str(a) for a in prefix_activities]

        # Find best matching context (longest k-gram first)
        best_counts = None
        for k in range(min(self.basic_k, len(prefix_activities)), 0, -1):
            context = tuple(prefix_activities[-k:])
            counts_dict = self.counts_by_ctx[k].get(context)
            if counts_dict:
                best_counts = dict(counts_dict)  # Copy the defaultdict
                break

        # Fallback to global counts if no context match
        if best_counts is None:
            best_counts = dict(self.global_next_counts)

        # If enabled_next provided, compute distribution over those activities only
        if enabled_next is not None:
            enabled_next = [str(a) for a in enabled_next]
            if not enabled_next:
                return {}

            # Laplace smooth over enabled activities
            tmp = {}
            for act in enabled_next:
                count = float(best_counts.get(act, 0))
                tmp[act] = count + self.alpha

            total = sum(tmp.values())
            if total <= 0:
                # Uniform distribution if no counts
                uniform = 1.0 / len(enabled_next)
                return {a: uniform for a in enabled_next}

            dist = {a: v / total for a, v in tmp.items()}
        else:
            # Return distribution over all observed successors
            dist = self._compute_smoothed_probs(best_counts)

        # Apply minimum probability filter
        if dist and self.min_probability > 0.0:
            filtered = {a: p for a, p in dist.items() if p >= self.min_probability}
            if filtered:
                dist = filtered

        # Renormalize
        if dist:
            total = sum(dist.values())
            if total > 0:
                dist = {a: p / total for a, p in dist.items()}

        return dist

    def sample_next_activity(
            self,
            prefix_activities: List[str],
            enabled_next: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Sample next activity from the probability distribution.

        This is the main method called by the simulation engine at XOR gateways.

        Args:
            prefix_activities: List of activities executed so far
            enabled_next: If provided, only sample from these (XOR outgoing arcs)

        Returns:
            Sampled activity name, or None if no valid options
        """
        dist = self.get_next_activity_distribution(prefix_activities, enabled_next)
        if not dist:
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
    ):
        """
        Unified prediction API for both basic and advanced modes.

        Args:
            case_prefix_activities: Activities executed so far in the case
            current_timestamp: Current simulation time (for advanced mode)
            case_start_timestamp: Case start time (for advanced mode)
            enabled_next: Allowed next activities (XOR outgoing)
            return_distribution: If True, return dict of probabilities

        Returns:
            If return_distribution: Dict[str, float]
            Otherwise: sampled activity name (str)
        """
        if self.mode == "basic":
            if return_distribution:
                return self.get_next_activity_distribution(
                    case_prefix_activities,
                    enabled_next=enabled_next
                )
            return self.sample_next_activity(
                case_prefix_activities,
                enabled_next=enabled_next
            )

        # Advanced mode uses ML model
        return self._predict_advanced(
            case_prefix_activities,
            current_timestamp,
            case_start_timestamp
        )

    def _predict_advanced(
            self,
            prefix_activities: List[str],
            current_timestamp=None,
            case_start_timestamp=None
    ) -> Optional[str]:
        """Advanced ML-based prediction."""
        if self.model is None:
            return None

        def map_to_idx(act: str) -> int:
            if act in self.activities_set:
                return self.activity_encoder.transform([act])[0]
            return -1

        encoded = np.array([map_to_idx(str(a)) for a in prefix_activities], dtype=int)

        # Pad or truncate to window_size
        if len(encoded) < self.window_size:
            pad = np.full(self.window_size - len(encoded), -1)
            ngram_feat = np.concatenate([pad, encoded])
        else:
            ngram_feat = encoded[-self.window_size:]

        # Extract time features
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
        return self.activity_encoder.inverse_transform([pred_idx])[0]

    # =========================================================
    # SERIALIZATION METHODS
    # =========================================================

    def save_to_json(
            self,
            output_dir: str,
            filename_prefix: str = "next_activity_probs",
            decision_points_only: bool = True
    ) -> Dict[str, str]:
        """
        Save learned probabilities to JSON files.

        Args:
            output_dir: Directory to save files
            filename_prefix: Prefix for output filenames
            decision_points_only: If True, only save contexts with 2+ successors

        Returns:
            Dict of output file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        def ctx_to_key(ctx: tuple) -> str:
            return " || ".join(ctx)

        outputs = {}

        # Save counts
        counts_json = {}
        for k in range(1, self.basic_k + 1):
            layer = {}
            for ctx, next_counts in self.counts_by_ctx[k].items():
                if decision_points_only and len(next_counts) < 2:
                    continue
                layer[ctx_to_key(ctx)] = dict(next_counts)
            counts_json[str(k)] = layer

        counts_path = Path(output_dir) / f"{filename_prefix}_counts_k{self.basic_k}.json"
        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump(counts_json, f, ensure_ascii=False, indent=2)
        outputs["counts"] = str(counts_path)

        # Save probabilities
        probs_json = {}
        for k in range(1, self.basic_k + 1):
            layer = {}
            for ctx, next_probs in self.probs_by_ctx[k].items():
                if decision_points_only:
                    counts = self.counts_by_ctx[k].get(ctx, {})
                    if len(counts) < 2:
                        continue
                layer[ctx_to_key(ctx)] = next_probs
            probs_json[str(k)] = layer

        probs_path = Path(output_dir) / f"{filename_prefix}_probs_k{self.basic_k}.json"
        with open(probs_path, "w", encoding="utf-8") as f:
            json.dump(probs_json, f, ensure_ascii=False, indent=2)
        outputs["probs"] = str(probs_path)

        # Save global probabilities
        global_path = Path(output_dir) / f"{filename_prefix}_global_probs.json"
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(self.global_next_probs, f, ensure_ascii=False, indent=2)
        outputs["global_probs"] = str(global_path)

        return outputs

    def load_from_json(self, counts_path: str, probs_path: str, global_path: str):
        """Load previously saved probabilities from JSON files."""
        def key_to_ctx(key: str) -> tuple:
            return tuple(key.split(" || "))

        with open(counts_path, "r", encoding="utf-8") as f:
            counts_json = json.load(f)

        with open(probs_path, "r", encoding="utf-8") as f:
            probs_json = json.load(f)

        with open(global_path, "r", encoding="utf-8") as f:
            self.global_next_probs = json.load(f)

        # Reconstruct counts
        for k_str, layer in counts_json.items():
            k = int(k_str)
            if k not in self.counts_by_ctx:
                self.counts_by_ctx[k] = defaultdict(lambda: defaultdict(int))
            for ctx_key, next_counts in layer.items():
                ctx = key_to_ctx(ctx_key)
                self.counts_by_ctx[k][ctx] = defaultdict(int, next_counts)

        # Reconstruct probs
        for k_str, layer in probs_json.items():
            k = int(k_str)
            if k not in self.probs_by_ctx:
                self.probs_by_ctx[k] = {}
            for ctx_key, next_probs in layer.items():
                ctx = key_to_ctx(ctx_key)
                self.probs_by_ctx[k][ctx] = next_probs

        # Reconstruct global counts from probs (approximate)
        self.global_next_counts = defaultdict(int)
        for act in self.global_next_probs.keys():
            self.global_next_counts[act] = 1  # Just mark as seen


# ============================================================
# INTEGRATION WITH SIMULATION ENGINE
# ============================================================

def create_predictor_aware_route_function(
        predictor: ExpertActivityPredictor,
        process_model: dict,
        gateways: dict,
        case_traces: dict,
):
    """
    Create a route_next function that uses the predictor for XOR gateways.

    This function is meant to replace the random route_next in SimulationEngine.

    Args:
        predictor: Trained ExpertActivityPredictor
        process_model: Dict mapping activity -> list of outgoing activities
        gateways: Dict mapping activity -> gateway type ('xor', 'or', etc.)
        case_traces: Dict[case_id -> list of executed activities] (mutable, updated during sim)

    Returns:
        A route_next function compatible with SimulationEngine
    """
    import random

    def route_next_with_predictor(activity: str, case_id: str) -> List[str]:
        """
        Decide next activities using learned probabilities for XOR gateways.
        """
        outgoing = process_model.get(activity, [])
        if not outgoing:
            return []

        gateway_type = gateways.get(activity)

        # Get trace history for this case
        trace = case_traces.get(case_id, [])

        # Special handling for OR-split with exclusive cancel option
        if activity == "W_Call after offers & A_Complete":
            cancel_act = "A_Cancelled & O_Cancelled"
            or_candidates = [a for a in outgoing if a != cancel_act]

            # Check if cancel should be chosen
            if cancel_act in outgoing:
                # Use predictor to get probability of cancel
                dist = predictor.get_next_activity_distribution(trace, outgoing)
                cancel_prob = dist.get(cancel_act, 0.2)

                if random.random() < cancel_prob:
                    return [cancel_act]

            # Otherwise normal OR split
            if or_candidates:
                k = random.randint(1, len(or_candidates))
                return random.sample(or_candidates, k)
            return []

        # XOR gateway: sample based on learned probabilities
        if gateway_type == "xor":
            sampled = predictor.sample_next_activity(trace, enabled_next=outgoing)
            if sampled:
                return [sampled]
            # Fallback to random if predictor fails
            return [random.choice(outgoing)]

        # OR gateway: random non-empty subset
        if gateway_type == "or":
            k = random.randint(1, len(outgoing))
            return random.sample(outgoing, k)

        # Default: all outgoing (sequence/parallel)
        return outgoing

    return route_next_with_predictor


# ============================================================
# MODIFIED SIMULATION ENGINE WITH PREDICTOR
# ============================================================

class SimulationEngineWithPredictor:
    """
    Extended SimulationEngine that integrates ExpertActivityPredictor
    for probability-based XOR gateway decisions.

    Key additions:
    - Tracks execution trace per case
    - Uses predictor.sample_next_activity() at XOR gateways
    """

    import heapq
    from datetime import datetime, timedelta

    def __init__(
            self,
            process_model: dict,
            start_time,
            gateways: dict = None,
            predictor: ExpertActivityPredictor = None
    ):
        self.model = process_model
        self.gateways = gateways or {}
        self.now = start_time
        self.event_queue = []
        self.log_rows = []
        self.case_context = {}

        # Task 1.4: Activity predictor and trace tracking
        self.predictor = predictor
        self.case_traces = {}  # case_id -> list of executed activities

    def schedule_event(self, time, case_id: str, activity: str):
        import heapq

        class SimEvent:
            def __init__(self, time, case_id, activity):
                self.time = time
                self.case_id = case_id
                self.activity = activity
            def __lt__(self, other):
                return self.time < other.time

        heapq.heappush(self.event_queue, SimEvent(time, case_id, activity))

    def log_event(self, case_id: str, activity: str, timestamp):
        self.log_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": timestamp.isoformat()
        })

        # Track executed activities for predictor
        if case_id not in self.case_traces:
            self.case_traces[case_id] = []
        self.case_traces[case_id].append(activity)

    def route_next(self, activity: str, case_id: str) -> List[str]:
        """
        Decide which outgoing activities to schedule.

        TASK 1.4: Uses predictor for XOR gateways instead of random choice.
        """
        import random

        outgoing = self.model.get(activity, [])
        if not outgoing:
            return []

        gateway_type = self.gateways.get(activity)
        trace = self.case_traces.get(case_id, [])

        # Special case: OR-split with exclusive cancel option
        if activity == "W_Call after offers & A_Complete":
            cancel_act = "A_Cancelled & O_Cancelled"
            or_candidates = [a for a in outgoing if a != cancel_act]

            if cancel_act in outgoing:
                # Use predictor if available
                if self.predictor:
                    dist = self.predictor.get_next_activity_distribution(trace, outgoing)
                    cancel_prob = dist.get(cancel_act, 0.2)
                else:
                    cancel_prob = 0.2

                if random.random() < cancel_prob:
                    return [cancel_act]

            if or_candidates:
                k = random.randint(1, len(or_candidates))
                return random.sample(or_candidates, k)
            return []

        # XOR gateway: use predictor for probability-based sampling
        if gateway_type == "xor":
            if self.predictor:
                # TASK 1.4 BASIC: Sample based on learned probabilities
                sampled = self.predictor.sample_next_activity(trace, enabled_next=outgoing)
                if sampled:
                    return [sampled]
            # Fallback to random
            return [random.choice(outgoing)]

        # OR gateway: random subset
        if gateway_type == "or":
            k = random.randint(1, len(outgoing))
            return random.sample(outgoing, k)

        # Default: all outgoing
        return outgoing

    def run(self, duration_function, max_events_per_case: int = 200):
        """Run the simulation."""
        import heapq

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.now = event.time

            ctx = self.case_context.setdefault(
                event.case_id,
                {"event_count": 0, "ended": False, "start_time": event.time}
            )

            if ctx["ended"]:
                continue

            self.log_event(event.case_id, event.activity, event.time)

            if event.activity == "END":
                ctx["ended"] = True
                continue

            ctx["event_count"] += 1

            if ctx["event_count"] > max_events_per_case:
                ctx["ended"] = True
                continue

            # Route to next activities (uses predictor for XOR)
            next_activities = self.route_next(event.activity, event.case_id)

            for next_act in next_activities:
                dur = duration_function(next_act, event.time, ctx)
                self.schedule_event(event.time + dur, event.case_id, next_act)

    def export_csv(self, path: str):
        import pandas as pd
        df = pd.DataFrame(self.log_rows)
        df = df.sort_values(["case:concept:name", "time:timestamp"])
        df.to_csv(path, index=False)
        print(f"CSV exported to {path}")

    def export_xes(self, path: str):
        import pandas as pd
        from pm4py.objects.log.obj import EventLog, Trace, Event
        from pm4py.objects.log.exporter.xes import exporter as xes_exporter

        df = pd.DataFrame(self.log_rows)
        df = df.sort_values(["case:concept:name", "time:timestamp"])

        log = EventLog()
        for case_id, group in df.groupby("case:concept:name"):
            trace = Trace()
            trace.attributes["concept:name"] = case_id
            for _, row in group.iterrows():
                trace.append(Event({
                    "concept:name": row["concept:name"],
                    "time:timestamp": pd.to_datetime(row["time:timestamp"])
                }))
            log.append(trace)

        xes_exporter.apply(log, path)
        print(f"XES exported to {path}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def train_predictor_from_log(
        log_df: pd.DataFrame,
        mode: str = "basic",
        context_k: int = 2,
        **kwargs
) -> ExpertActivityPredictor:
    """
    Convenience function to train a predictor from an event log DataFrame.

    Args:
        log_df: DataFrame with case_id/case:concept:name, activity/concept:name, timestamp
        mode: 'basic' or 'advanced'
        context_k: Context window size for k-gram (basic mode)
        **kwargs: Additional predictor parameters

    Returns:
        Trained ExpertActivityPredictor
    """
    predictor = ExpertActivityPredictor(
        mode=mode,
        basic_context_k=context_k,
        **kwargs
    )
    predictor.fit(log_df)
    return predictor


def evaluate_predictor(
        predictor: ExpertActivityPredictor,
        test_df: pd.DataFrame,
        print_report: bool = True
) -> dict:
    """
    Evaluate predictor accuracy on test data.

    Args:
        predictor: Trained ExpertActivityPredictor
        test_df: Test DataFrame with case_id, activity, timestamp
        print_report: Whether to print results

    Returns:
        Dict with accuracy metrics
    """
    test_df = predictor._normalize_columns(test_df)
    test_df = test_df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

    correct = 0
    total = 0

    for case_id, group in test_df.groupby("case_id", sort=False):
        activities = group["activity"].astype(str).tolist()

        for i in range(len(activities) - 1):
            prefix = activities[:i+1]
            actual_next = activities[i+1]

            dist = predictor.get_next_activity_distribution(prefix)
            if dist:
                predicted = max(dist, key=dist.get)
                if predicted == actual_next:
                    correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    if print_report:
        print(f"\n{'='*40}")
        print(f"  PREDICTOR EVALUATION")
        print(f"{'='*40}")
        print(f"Total predictions: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    print("Task 1.4: Next Activity Predictor Module")
    print("=" * 50)

    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        "case_id": ["C1"]*5 + ["C2"]*4 + ["C3"]*5,
        "activity": [
            "A", "B", "C", "D", "E",  # Case 1
            "A", "B", "D", "E",       # Case 2
            "A", "C", "D", "C", "E"   # Case 3
        ],
        "timestamp": pd.date_range("2024-01-01", periods=14, freq="h")
    })

    print("\nSample training data:")
    print(sample_data)

    # Train predictor
    predictor = ExpertActivityPredictor(mode="basic", basic_context_k=2)
    predictor.fit(sample_data)


    # Test prediction
    print("\nTest predictions:")
    test_prefix = ["A", "B"]
    dist = predictor.get_next_activity_distribution(test_prefix)
    print(f"P(next | {test_prefix}) = {dist}")

    sampled = predictor.sample_next_activity(test_prefix)
    print(f"Sampled next activity: {sampled}")

    # With enabled_next constraint (XOR simulation)
    enabled = ["C", "D"]
    dist_constrained = predictor.get_next_activity_distribution(test_prefix, enabled_next=enabled)
    print(f"\nP(next | {test_prefix}, enabled={enabled}) = {dist_constrained}")

    print("\n" + "=" * 50)
    print("Module ready for integration with simulation_engine_core.py")
