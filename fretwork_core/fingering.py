"""Voice hints, fretboard enumeration, and exact passage optimization."""

from __future__ import annotations

import copy
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .models import (
    BASS_REGISTER_CEILING,
    CHORD_ONSET_WINDOW,
    MAX_FRET,
    STANDARD_TUNING,
    FingeringReport,
    FingeringShape,
    NoteEvent,
)


def cluster_events(
    events: Sequence[NoteEvent], onset_window: float = CHORD_ONSET_WINDOW
) -> list[list[NoteEvent]]:
    """Group near-simultaneous attacks while preserving chronology."""

    if not events:
        return []
    ordered = sorted(events, key=lambda event: (event.start, event.pitch))
    clusters: list[list[NoteEvent]] = []
    current = [ordered[0]]
    anchor = ordered[0].start
    for event in ordered[1:]:
        if event.start - anchor <= onset_window:
            current.append(event)
        else:
            clusters.append(current)
            current = [event]
            anchor = event.start
    clusters.append(current)
    return clusters


def separate_voices(
    events: list[NoteEvent], onset_window: float = CHORD_ONSET_WINDOW
) -> list[NoteEvent]:
    """Separate thumb bass from upper melody/chord attacks."""

    for cluster in cluster_events(events, onset_window):
        cluster.sort(key=lambda event: event.pitch)
        has_bass = cluster[0].pitch <= BASS_REGISTER_CEILING
        upper_count = len(cluster) - int(has_bass)
        for index, event in enumerate(cluster):
            if index == 0 and has_bass:
                event.voice = "bass"
            elif upper_count > 1:
                event.voice = "chord"
            else:
                event.voice = "melody"
    return events


def possible_positions(pitch: int, capo: int) -> list[tuple[int, int]]:
    """Return every (string, capo-relative fret) for a sounding pitch."""

    if not 0 <= capo <= 12:
        raise ValueError("Capo must be between 0 and 12")
    return [
        (string, pitch - (open_pitch + capo))
        for string, open_pitch in STANDARD_TUNING.items()
        if 0 <= pitch - (open_pitch + capo) <= MAX_FRET
    ]


def parse_tab_hints(tab_path: str) -> set[tuple[int, int]]:
    """Extract string/fret preferences from loose ASCII TAB."""

    name_to_number = {"e": 1, "B": 2, "G": 3, "D": 4, "A": 5, "E": 6}
    hints: set[tuple[int, int]] = set()
    lines = Path(tab_path).read_text(encoding="utf-8", errors="replace").splitlines()
    for raw_line in lines:
        stripped = raw_line.strip()
        if "|" not in stripped:
            continue
        label, body = stripped.split("|", 1)
        label = label.strip()
        if label not in name_to_number:
            continue
        number = ""
        for character in body + " ":
            if character.isdigit():
                number += character
            elif number:
                hints.add((name_to_number[label], int(number)))
                number = ""
    return hints


def _role_cost(event: NoteEvent, string: Optional[int]) -> float:
    if string is None:
        return 0.0
    if event.voice == "bass":
        return 0.0 if string in (4, 5, 6) else 1.25
    return 0.0 if string in (1, 2, 3, 4) else 0.75


def _shape_from_assignment(
    cluster: Sequence[NoteEvent],
    assignment: list[tuple[Optional[int], Optional[int]]],
    hints: Optional[set[tuple[int, int]]],
    max_span: int,
) -> FingeringShape:
    frets = [fret for _, fret in assignment if fret is not None and fret > 0]
    span = max(frets) - min(frets) if frets else 0
    center = sum(frets) / len(frets) if frets else None
    unassigned = [index for index, (string, _) in enumerate(assignment) if string is None]
    stretch_over = max(0, span - max_span)
    hint_matches = sum(bool(hints and position in hints) for position in assignment)
    local_cost = (
        span * 1.8
        + stretch_over * 14.0
        + len(unassigned) * 100.0
        + sum(cluster[index].confidence * 10.0 for index in unassigned)
        + sum(
            _role_cost(event, string)
            for event, (string, _) in zip(cluster, assignment)
        )
        - hint_matches * 2.0
        + sum(fret or 0 for _, fret in assignment) * 0.035
    )
    return FingeringShape(
        list(assignment), local_cost, center, span, stretch_over > 0
    )


def _enumerate_subset(
    cluster: Sequence[NoteEvent],
    active_indices: tuple[int, ...],
    capo: int,
    hints: Optional[set[tuple[int, int]]],
    max_span: int,
) -> list[FingeringShape]:
    active = set(active_indices)
    option_sets: list[list[tuple[Optional[int], Optional[int]]]] = []
    for index, event in enumerate(cluster):
        if index not in active:
            option_sets.append([(None, None)])
            continue
        positions = possible_positions(event.pitch, capo)
        if not positions:
            return []
        option_sets.append(list(positions))

    shapes: list[FingeringShape] = []
    assignment: list[tuple[Optional[int], Optional[int]]] = []

    def visit(index: int, used_strings: set[int]) -> None:
        if index == len(cluster):
            shapes.append(_shape_from_assignment(cluster, assignment, hints, max_span))
            return
        for string, fret in option_sets[index]:
            if string is not None and string in used_strings:
                continue
            assignment.append((string, fret))
            if string is not None:
                used_strings.add(string)
            visit(index + 1, used_strings)
            if string is not None:
                used_strings.remove(string)
            assignment.pop()

    visit(0, set())
    return shapes


def enumerate_cluster_fingerings(
    cluster: Sequence[NoteEvent],
    capo: int,
    hints: Optional[set[tuple[int, int]]] = None,
    max_span: int = 4,
) -> list[FingeringShape]:
    """Enumerate the complete playable shape layer for one onset cluster."""

    all_indices = tuple(range(len(cluster)))
    if len(cluster) <= 6:
        complete = _enumerate_subset(cluster, all_indices, capo, hints, max_span)
        if complete:
            return sorted(complete, key=lambda shape: shape.local_cost)

    # Physically impossible ML clusters use complete enumeration over every
    # maximum-cardinality subset of the six strongest detections.
    recovery_indices = tuple(
        sorted(
            range(len(cluster)),
            key=lambda index: cluster[index].confidence,
            reverse=True,
        )[:6]
    )
    for retained_count in range(len(recovery_indices), -1, -1):
        partial: list[FingeringShape] = []
        for active_indices in combinations(recovery_indices, retained_count):
            partial.extend(
                _enumerate_subset(cluster, active_indices, capo, hints, max_span)
            )
        if partial:
            return sorted(partial, key=lambda shape: shape.local_cost)
    raise RuntimeError("Could not construct even an empty fallback fingering shape")


def _center_transition_cost(
    previous_center: Optional[float], current_center: Optional[float]
) -> float:
    if previous_center is None or current_center is None:
        return 0.0
    shift = abs(previous_center - current_center)
    return shift * 0.8 + max(0.0, shift - 5.0) * 1.8


def _transition_cost(
    previous: FingeringShape,
    current: FingeringShape,
    previous_events: Sequence[NoteEvent] = (),
    current_events: Sequence[NoteEvent] = (),
) -> float:
    """Cost one adjacent pair; this is the fingering model's Markov edge."""

    cost = _center_transition_cost(previous.center, current.center)
    previous_by_string = {
        string: fret
        for string, fret in previous.positions
        if string is not None and fret is not None
    }
    current_by_string = {
        string: fret
        for string, fret in current.positions
        if string is not None and fret is not None
    }
    shared = previous_by_string.keys() & current_by_string.keys()
    cost += sum(
        abs(previous_by_string[string] - current_by_string[string]) * 0.16
        for string in shared
    )
    cost += len(previous_by_string.keys() ^ current_by_string.keys()) * 0.10
    if previous_events and current_events:
        current_onset = min(event.start for event in current_events)
        current_pitch_by_string = {
            string: event.pitch
            for event, (string, _) in zip(current_events, current.positions)
            if string is not None
        }
        for event, (string, _) in zip(previous_events, previous.positions):
            if (
                string is not None
                and event.end > current_onset + 0.03
                and string in current_pitch_by_string
                and current_pitch_by_string[string] != event.pitch
            ):
                cost += 6.0 * max(0.5, event.confidence)
    return cost


def _shape_layers(
    clusters: Sequence[Sequence[NoteEvent]],
    capo: int,
    hints: Optional[set[tuple[int, int]]],
    max_span: int,
) -> list[list[FingeringShape]]:
    cache: dict[tuple[tuple[int, str, float], ...], list[FingeringShape]] = {}
    layers = []
    for cluster in clusters:
        key = tuple(
            (event.pitch, event.voice, round(event.confidence, 6)) for event in cluster
        )
        if key not in cache:
            cache[key] = enumerate_cluster_fingerings(cluster, capo, hints, max_span)
        layers.append(cache[key])
    return layers


def _viterbi_shapes(
    clusters: Sequence[Sequence[NoteEvent]],
    layers: Sequence[Sequence[FingeringShape]],
) -> tuple[list[FingeringShape], float]:
    costs: list[list[float]] = [[shape.local_cost for shape in layers[0]]]
    parents: list[list[Optional[int]]] = [[None] * len(layers[0])]
    for layer_index in range(1, len(layers)):
        layer_costs: list[float] = []
        layer_parents: list[Optional[int]] = []
        for shape in layers[layer_index]:
            options = [
                costs[layer_index - 1][previous_index]
                + _transition_cost(
                    previous,
                    shape,
                    clusters[layer_index - 1],
                    clusters[layer_index],
                )
                for previous_index, previous in enumerate(layers[layer_index - 1])
            ]
            parent = min(range(len(options)), key=options.__getitem__)
            layer_costs.append(shape.local_cost + options[parent])
            layer_parents.append(parent)
        costs.append(layer_costs)
        parents.append(layer_parents)

    indices = [0] * len(layers)
    indices[-1] = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    for index in range(len(layers) - 1, 0, -1):
        indices[index - 1] = int(parents[index][indices[index]])
    return (
        [layers[index][shape_index] for index, shape_index in enumerate(indices)],
        costs[-1][indices[-1]],
    )


def assign_positions(
    events: list[NoteEvent],
    capo: int = 0,
    hints: Optional[set[tuple[int, int]]] = None,
    max_span: int = 4,
    onset_window: float = CHORD_ONSET_WINDOW,
) -> FingeringReport:
    """Choose the globally minimum-cost path over the complete shape trellis."""

    clusters = cluster_events(events, onset_window)
    if not clusters:
        return FingeringReport(capo, -100.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    layers = _shape_layers(clusters, capo, hints, max_span)
    chosen, optimization_cost = _viterbi_shapes(clusters, layers)
    for cluster, shape in zip(clusters, chosen):
        for event, (string, fret) in zip(cluster, shape.positions):
            event.string, event.fret = string, fret

    assigned = [event for event in events if event.string is not None]
    fretted = [event.fret for event in assigned if event.fret is not None and event.fret > 0]
    opens = [event for event in assigned if event.fret == 0]
    centers = [shape.center for shape in chosen if shape.center is not None]
    movements = [abs(right - left) for left, right in zip(centers, centers[1:])]
    stretch_violations = sum(shape.violates_stretch for shape in chosen)
    assigned_ratio = len(assigned) / len(events) if events else 0.0
    open_ratio = len(opens) / len(assigned) if assigned else 0.0
    average_fret = sum(fretted) / len(fretted) if fretted else 0.0
    average_movement = sum(movements) / len(movements) if movements else 0.0
    unassigned = len(events) - len(assigned)
    score = (
        assigned_ratio * 100.0
        + open_ratio * 8.0
        - average_fret * 0.7
        - average_movement * 1.4
        - stretch_violations * 15.0
        - unassigned * 20.0
    )
    return FingeringReport(
        capo,
        round(score, 3),
        assigned_ratio,
        open_ratio,
        average_fret,
        average_movement,
        stretch_violations,
        unassigned,
        round(optimization_cost, 6),
        [len(layer) for layer in layers],
    )


def rank_capos(
    events: list[NoteEvent],
    hints: Optional[set[tuple[int, int]]] = None,
    max_span: int = 4,
    candidates: Iterable[int] = range(0, 8),
) -> list[FingeringReport]:
    """Rank capo hypotheses by reachability and whole-passage playability."""

    reports = [
        assign_positions(copy.deepcopy(events), capo, hints=hints, max_span=max_span)
        for capo in candidates
    ]
    return sorted(reports, key=lambda report: report.score, reverse=True)

