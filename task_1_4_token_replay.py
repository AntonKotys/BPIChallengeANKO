"""
Task 1.4 Advanced: Token Replay for Decision Point Identification.

Uses pm4py token-based replay to identify decision points (XOR gateways)
in a Petri net and extract clean branching data for predictor training.

Usage:
    extractor = TokenReplayDecisionExtractor()
    extractor.load_model_from_bpmn("model.bpmn")
    decisions = extractor.replay_and_extract(event_log)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime

try:
    import pm4py
    from pm4py.objects.petri_net.obj import PetriNet, Marking
    from pm4py.objects.petri_net.utils import petri_utils
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay_algo
    from pm4py.objects.log.obj import EventLog, Trace, Event
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False


# ============================================================
# TOKEN REPLAY DECISION EXTRACTOR
# ============================================================

class TokenReplayDecisionExtractor:
    """
    Extracts decision point data from event logs using token-based replay.

    Workflow:
        1. Load a process model (BPMN file or discover from log)
        2. Convert to Petri net
        3. Identify decision places (places with >1 outgoing transition)
        4. Replay historical traces on the Petri net
        5. At each decision place, record which branch was taken + trace prefix
        6. Return structured decision data for predictor training
    """

    def __init__(self):
        if not PM4PY_AVAILABLE:
            raise ImportError(
                "pm4py is required for token replay. "
                "Install via: pip install pm4py"
            )

        self.net: Optional[PetriNet] = None
        self.initial_marking: Optional[Marking] = None
        self.final_marking: Optional[Marking] = None
        self.decision_places: Dict[Any, List[Any]] = {}
        self.decision_data: List[Dict] = []
        self.replay_stats: Dict = {}

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def load_model_from_bpmn(self, bpmn_path: str) -> 'TokenReplayDecisionExtractor':
        """
        Load BPMN model and convert to Petri net.

        Args:
            bpmn_path: Path to .bpmn file

        Returns:
            self (for method chaining)
        """
        try:
            bpmn_graph = pm4py.read_bpmn(bpmn_path)
            self.net, self.initial_marking, self.final_marking = (
                pm4py.convert_to_petri_net(bpmn_graph)
            )
        except Exception:
            from pm4py.objects.bpmn.importer import importer as bpmn_importer
            from pm4py.objects.conversion.bpmn import converter as bpmn_converter
            bpmn_graph = bpmn_importer.apply(bpmn_path)
            self.net, self.initial_marking, self.final_marking = (
                bpmn_converter.apply(bpmn_graph)
            )

        self._identify_decision_places()
        return self

    def load_model_from_petri_net(
        self, net: PetriNet, im: Marking, fm: Marking
    ) -> 'TokenReplayDecisionExtractor':
        """
        Load directly from Petri net objects (useful for testing).

        Args:
            net: PetriNet object
            im: Initial marking
            fm: Final marking

        Returns:
            self (for method chaining)
        """
        self.net = net
        self.initial_marking = im
        self.final_marking = fm
        self._identify_decision_places()
        return self

    def discover_model_from_log(
        self, log: EventLog
    ) -> 'TokenReplayDecisionExtractor':
        """
        Discover Petri net from event log using inductive miner.
        Use this as fallback when BPMN model is unavailable or replay fitness is low.

        Args:
            log: pm4py EventLog

        Returns:
            self (for method chaining)
        """
        self.net, self.initial_marking, self.final_marking = (
            pm4py.discover_petri_net_inductive(log)
        )
        self._identify_decision_places()
        return self

    # =========================================================
    # DECISION PLACE IDENTIFICATION
    # =========================================================

    def _identify_decision_places(self):
        """
        Find places in the Petri net with >1 outgoing arc.
        These correspond to XOR/OR split gateways (choice points).
        """
        self.decision_places = {}
        if self.net is None:
            return

        for place in self.net.places:
            out_arcs = list(place.out_arcs)
            if len(out_arcs) > 1:
                transitions = [arc.target for arc in out_arcs]
                self.decision_places[place] = transitions

    def get_decision_place_info(self) -> List[Dict]:
        """
        Return human-readable info about identified decision places.
        Useful for debugging and reporting.
        """
        info = []
        for place, transitions in self.decision_places.items():
            labels = []
            for t in transitions:
                if t.label:
                    labels.append(t.label)
                else:
                    labels.append(f"[silent:{t.name}]")
            info.append({
                "place": str(place),
                "n_outgoing": len(transitions),
                "transition_labels": labels,
            })
        return info

    # =========================================================
    # REPLAY AND DECISION EXTRACTION
    # =========================================================

    def replay_and_extract(
        self,
        log: EventLog,
        min_fitness: float = 0.0,
        activity_mapping: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Replay traces on the Petri net and extract decision point data.

        For each decision point encountered during replay, records:
        - case_id: identifier of the case
        - decision_place: the Petri net place where the decision was made
        - chosen_transition: label of the transition that fired (the branch taken)
        - prefix_activities: visible activities executed before this decision
        - all_options: labels of all possible transitions at this decision place
        - position_in_trace: index in the activated transitions sequence
        - trace_is_fit: whether the full trace replayed successfully

        Args:
            log: pm4py EventLog to replay
            min_fitness: Skip traces with fitness below this threshold (0.0 = keep all)
            activity_mapping: Optional dict to map log activity names before replay

        Returns:
            List of decision instance dicts
        """
        if self.net is None:
            raise ValueError(
                "No model loaded. Call load_model_from_bpmn() or "
                "discover_model_from_log() first."
            )

        if activity_mapping:
            log = self._apply_activity_mapping(log, activity_mapping)

        replayed = token_replay_algo.apply(
            log, self.net, self.initial_marking, self.final_marking
        )

        self.decision_data = []
        fit_count = 0
        total_count = len(replayed)
        total_decisions = 0

        for trace_idx, trace_result in enumerate(replayed):
            is_fit = trace_result.get('trace_is_fit', False)
            trace_fitness = trace_result.get('trace_fitness', 0.0)

            if is_fit:
                fit_count += 1

            if trace_fitness < min_fitness:
                continue

            activated = trace_result.get('activated_transitions', [])
            if not activated:
                continue

            case_id = str(
                log[trace_idx].attributes.get('concept:name', f'trace_{trace_idx}')
            )

            prefix = []

            for trans_idx, transition in enumerate(activated):
                for arc in transition.in_arcs:
                    place = arc.source
                    if place in self.decision_places:
                        all_options = self.decision_places[place]

                        option_labels = []
                        for t in all_options:
                            lbl = t.label if t.label else f"[silent:{t.name}]"
                            option_labels.append(lbl)

                        chosen_label = (
                            transition.label
                            if transition.label
                            else f"[silent:{transition.name}]"
                        )

                        self.decision_data.append({
                            'case_id': case_id,
                            'decision_place': str(place),
                            'chosen_transition': chosen_label,
                            'prefix_activities': list(prefix),
                            'all_options': option_labels,
                            'position_in_trace': trans_idx,
                            'trace_is_fit': is_fit,
                        })
                        total_decisions += 1

                if transition.label:
                    prefix.append(transition.label)

        self.replay_stats = {
            'total_traces': total_count,
            'fit_traces': fit_count,
            'fitness_ratio': fit_count / total_count if total_count > 0 else 0.0,
            'decision_instances': total_decisions,
            'decision_places_found': len(self.decision_places),
        }

        return self.decision_data

    def _apply_activity_mapping(
        self, log: EventLog, mapping: Dict[str, str]
    ) -> EventLog:
        """Apply activity name mapping to an event log (non-destructive copy)."""
        import copy
        mapped_log = EventLog()
        for trace in log:
            new_trace = Trace()
            new_trace.attributes = dict(trace.attributes)
            for event in trace:
                new_event = Event(dict(event))
                act = new_event.get('concept:name', '')
                if act in mapping:
                    new_event['concept:name'] = mapping[act]
                new_trace.append(new_event)
            mapped_log.append(new_trace)
        return mapped_log

    # =========================================================
    # OUTPUT METHODS
    # =========================================================

    def get_decision_training_data(self) -> pd.DataFrame:
        """
        Convert extracted decision data to a DataFrame suitable for training.

        Returns:
            DataFrame with columns: case_id, decision_place, chosen_activity,
            prefix, prefix_length, n_options, options, position_in_trace
        """
        if not self.decision_data:
            return pd.DataFrame()

        rows = []
        for d in self.decision_data:
            rows.append({
                'case_id': d['case_id'],
                'decision_place': d['decision_place'],
                'chosen_activity': d['chosen_transition'],
                'prefix': '||'.join(d['prefix_activities']),
                'prefix_length': len(d['prefix_activities']),
                'n_options': len(d['all_options']),
                'options': '||'.join(d['all_options']),
                'position_in_trace': d['position_in_trace'],
            })

        return pd.DataFrame(rows)

    def get_gateway_statistics(self) -> Dict[str, Dict]:
        """
        Compute branching statistics per decision place.

        Returns:
            Dict mapping place name -> {total_decisions, branches: {act: {count, probability}}}
        """
        stats = defaultdict(lambda: defaultdict(int))

        for d in self.decision_data:
            place = d['decision_place']
            chosen = d['chosen_transition']
            stats[place][chosen] += 1

        result = {}
        for place, counts in stats.items():
            total = sum(counts.values())
            result[place] = {
                'total_decisions': total,
                'branches': {
                    act: {
                        'count': count,
                        'probability': count / total if total > 0 else 0.0,
                    }
                    for act, count in sorted(
                        counts.items(), key=lambda x: -x[1]
                    )
                },
            }

        return result

    def print_summary(self):
        """Print a human-readable summary of replay results."""
        if not self.replay_stats:
            print("[Token Replay] No replay has been performed yet.")
            return

        s = self.replay_stats
        print("\n" + "=" * 60)
        print("  TOKEN REPLAY SUMMARY")
        print("=" * 60)
        print(f"  Total traces:       {s['total_traces']}")
        print(f"  Fit traces:         {s['fit_traces']} ({s['fitness_ratio']:.1%})")
        print(f"  Decision places:    {s['decision_places_found']}")
        print(f"  Decision instances: {s['decision_instances']}")

        gw_stats = self.get_gateway_statistics()
        if gw_stats:
            print(f"\n  Decision Places:")
            for place, data in gw_stats.items():
                print(f"\n    {place} ({data['total_decisions']} decisions):")
                for act, info in data['branches'].items():
                    print(
                        f"      {act}: {info['count']} "
                        f"({info['probability']:.1%})"
                    )


# ============================================================
# HELPER: DataFrame <-> EventLog conversion
# ============================================================

def df_to_event_log(df: pd.DataFrame) -> EventLog:
    """
    Convert a pandas DataFrame to a pm4py EventLog.
    Handles common column name variations from BPI Challenge logs.

    Args:
        df: DataFrame with case_id/case:concept:name, activity/concept:name,
            timestamp/time:timestamp columns

    Returns:
        pm4py EventLog
    """
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py is required")

    df = df.copy()

    col_map = {
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp',
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    ts_col = 'time:timestamp'
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')

    try:
        log = pm4py.convert_to_event_log(df)
    except Exception:
        log = EventLog()
        case_col = 'case:concept:name'
        act_col = 'concept:name'

        for case_id, group in df.groupby(case_col, sort=False):
            trace = Trace()
            trace.attributes['concept:name'] = str(case_id)

            if ts_col in group.columns:
                group = group.sort_values(ts_col)

            for _, row in group.iterrows():
                event = Event({'concept:name': str(row[act_col])})
                if ts_col in group.columns and pd.notna(row.get(ts_col)):
                    event['time:timestamp'] = row[ts_col]
                trace.append(event)

            log.append(trace)

    return log


# ============================================================
# HELPER: Build a simple Petri net for testing
# ============================================================

def build_simple_xor_net():
    """
    Build a minimal Petri net with one XOR split for testing.

    Structure:
        [p_start] -> t_A -> [p_decision] -> t_B -> [p_after_B] -> t_end1 -> [p_end]
                                          -> t_C -> [p_after_C] -> t_end2 -> [p_end]

    Returns:
        (net, im, fm, decision_place)
    """
    net = PetriNet("test_xor_net")

    p_start = PetriNet.Place("p_start")
    p_decision = PetriNet.Place("p_decision")
    p_after_B = PetriNet.Place("p_after_B")
    p_after_C = PetriNet.Place("p_after_C")
    p_end = PetriNet.Place("p_end")

    for p in [p_start, p_decision, p_after_B, p_after_C, p_end]:
        net.places.add(p)

    t_A = PetriNet.Transition("t_A", label="A")
    t_B = PetriNet.Transition("t_B", label="B")
    t_C = PetriNet.Transition("t_C", label="C")
    t_end1 = PetriNet.Transition("t_end1", label="End")
    t_end2 = PetriNet.Transition("t_end2", label="End")

    for t in [t_A, t_B, t_C, t_end1, t_end2]:
        net.transitions.add(t)

    petri_utils.add_arc_from_to(p_start, t_A, net)
    petri_utils.add_arc_from_to(t_A, p_decision, net)
    petri_utils.add_arc_from_to(p_decision, t_B, net)
    petri_utils.add_arc_from_to(p_decision, t_C, net)
    petri_utils.add_arc_from_to(t_B, p_after_B, net)
    petri_utils.add_arc_from_to(t_C, p_after_C, net)
    petri_utils.add_arc_from_to(p_after_B, t_end1, net)
    petri_utils.add_arc_from_to(p_after_C, t_end2, net)
    petri_utils.add_arc_from_to(t_end1, p_end, net)
    petri_utils.add_arc_from_to(t_end2, p_end, net)

    im = Marking({p_start: 1})
    fm = Marking({p_end: 1})

    return net, im, fm, p_decision


def build_two_xor_net():
    """
    Build a Petri net with two sequential XOR splits for testing.

    Structure:
        [start] -> t_Start -> [d1] -> t_A -> [p1] -> t_D -> [d2] -> t_E -> [pE] -> t_end1 -> [end]
                                    -> t_B -> [p2] -> t_D2-> [d2]-> t_F -> [pF] -> t_end2 -> [end]
                                    -> t_C -> [p3] -> ...

    Simplified version with two decision points.
    """
    net = PetriNet("test_two_xor_net")

    p_start = PetriNet.Place("p_start")
    p_d1 = PetriNet.Place("p_d1")
    p_after_A = PetriNet.Place("p_after_A")
    p_after_B = PetriNet.Place("p_after_B")
    p_d2 = PetriNet.Place("p_d2")
    p_after_E = PetriNet.Place("p_after_E")
    p_after_F = PetriNet.Place("p_after_F")
    p_end = PetriNet.Place("p_end")

    for p in [p_start, p_d1, p_after_A, p_after_B, p_d2,
              p_after_E, p_after_F, p_end]:
        net.places.add(p)

    t_start = PetriNet.Transition("t_start", label="Start")
    t_A = PetriNet.Transition("t_A", label="A")
    t_B = PetriNet.Transition("t_B", label="B")
    t_D = PetriNet.Transition("t_D", label="D")
    t_D2 = PetriNet.Transition("t_D2", label="D")
    t_E = PetriNet.Transition("t_E", label="E")
    t_F = PetriNet.Transition("t_F", label="F")
    t_end1 = PetriNet.Transition("t_end1", label="End")
    t_end2 = PetriNet.Transition("t_end2", label="End")

    for t in [t_start, t_A, t_B, t_D, t_D2, t_E, t_F, t_end1, t_end2]:
        net.transitions.add(t)

    petri_utils.add_arc_from_to(p_start, t_start, net)
    petri_utils.add_arc_from_to(t_start, p_d1, net)

    petri_utils.add_arc_from_to(p_d1, t_A, net)
    petri_utils.add_arc_from_to(p_d1, t_B, net)
    petri_utils.add_arc_from_to(t_A, p_after_A, net)
    petri_utils.add_arc_from_to(t_B, p_after_B, net)
    petri_utils.add_arc_from_to(p_after_A, t_D, net)
    petri_utils.add_arc_from_to(p_after_B, t_D2, net)
    petri_utils.add_arc_from_to(t_D, p_d2, net)
    petri_utils.add_arc_from_to(t_D2, p_d2, net)

    petri_utils.add_arc_from_to(p_d2, t_E, net)
    petri_utils.add_arc_from_to(p_d2, t_F, net)
    petri_utils.add_arc_from_to(t_E, p_after_E, net)
    petri_utils.add_arc_from_to(t_F, p_after_F, net)
    petri_utils.add_arc_from_to(p_after_E, t_end1, net)
    petri_utils.add_arc_from_to(p_after_F, t_end2, net)
    petri_utils.add_arc_from_to(t_end1, p_end, net)
    petri_utils.add_arc_from_to(t_end2, p_end, net)

    im = Marking({p_start: 1})
    fm = Marking({p_end: 1})

    return net, im, fm


def build_event_log_from_traces(trace_defs: List[Tuple[str, List[str]]]) -> EventLog:
    """
    Build a pm4py EventLog from a list of (case_id, [activities]) tuples.

    Args:
        trace_defs: List of (case_id, activity_list) tuples

    Returns:
        pm4py EventLog
    """
    log = EventLog()
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    for case_id, activities in trace_defs:
        trace = Trace()
        trace.attributes['concept:name'] = case_id

        for i, act in enumerate(activities):
            from datetime import timedelta
            event = Event({
                'concept:name': act,
                'time:timestamp': base_time + timedelta(hours=i),
            })
            trace.append(event)

        log.append(trace)
        base_time += pd.Timedelta(days=1)

    return log


# ============================================================
# STANDALONE DEMO
# ============================================================

if __name__ == "__main__":
    print("Task 1.4 Advanced: Token Replay Decision Extraction")
    print("=" * 60)

    net, im, fm, _ = build_simple_xor_net()
    print(f"\nPetri net: {len(net.places)} places, {len(net.transitions)} transitions")

    log = build_event_log_from_traces([
        ("C1", ["A", "B", "End"]),
        ("C2", ["A", "B", "End"]),
        ("C3", ["A", "B", "End"]),
        ("C4", ["A", "C", "End"]),
        ("C5", ["A", "C", "End"]),
    ])
    print(f"Event log: {len(log)} traces")

    extractor = TokenReplayDecisionExtractor()
    extractor.load_model_from_petri_net(net, im, fm)

    print(f"\nDecision places found: {len(extractor.decision_places)}")
    for info in extractor.get_decision_place_info():
        print(f"  {info['place']}: {info['transition_labels']}")

    decisions = extractor.replay_and_extract(log)
    extractor.print_summary()

    print(f"\nDecision instances: {len(decisions)}")
    for d in decisions[:5]:
        print(
            f"  {d['case_id']}: prefix={d['prefix_activities']} "
            f"-> {d['chosen_transition']} "
            f"(options: {d['all_options']})"
        )

    print("\n" + "=" * 60)
    print("Module ready for integration with task_1_4_next_activity.py")
