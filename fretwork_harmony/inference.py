"""Arpeggio-aware chord ranking and Markov sequence decoding."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, Sequence

from .knowledge import KNOWLEDGE_PATH, load_chord_templates, load_knowledge
from .models import Candidate, ChordInferenceConfig, ChordLabel, ChordTemplate


def _value(event: object, name: str, default: float = 0.0) -> float:
    return float(getattr(event, name, default))


def _pitch_class_evidence(
    events: Iterable[object], capo: int
) -> tuple[dict[int, float], int]:
    weights: dict[int, float] = {}
    counts: dict[int, int] = {}
    valid = []
    for event in events:
        pitch = int(getattr(event, "pitch"))
        confidence = min(1.0, max(0.0, _value(event, "confidence")))
        pitch_class = (pitch - capo) % 12
        weights[pitch_class] = max(weights.get(pitch_class, 0.0), confidence)
        counts[pitch_class] = counts.get(pitch_class, 0) + 1
        valid.append(event)
    for pitch_class, count in counts.items():
        weights[pitch_class] += min(0.12, 0.03 * (count - 1))
    bass = min(valid, key=lambda event: int(getattr(event, "pitch")))
    return weights, (int(getattr(bass, "pitch")) - capo) % 12


def _symbol(
    root: int,
    bass: int,
    suffix: str,
    chord_pcs: set[int],
    root_names: Sequence[str],
    inversions: bool,
) -> str:
    result = f"{root_names[root]}{suffix}"
    if inversions and bass != root and bass in chord_pcs:
        result += f"/{root_names[bass]}"
    return result


def _rank_window(
    events: Sequence[object],
    capo: int,
    templates: Sequence[ChordTemplate],
    root_names: Sequence[str],
    include_inversions: bool,
) -> list[Candidate]:
    weights, bass_pc = _pitch_class_evidence(events, capo)
    observed, total_weight = set(weights), sum(weights.values()) or 1.0
    candidates: list[Candidate] = []
    for root_pc in range(12):
        for template in templates:
            chord_pcs = {(root_pc + interval) % 12 for interval in template.intervals}
            required = {(root_pc + interval) % 12 for interval in template.required}
            if len(observed & chord_pcs) < 2:
                continue
            precision = sum(weights[pc] for pc in observed & chord_pcs) / total_weight
            completeness = len(observed & chord_pcs) / len(chord_pcs)
            required_fraction = len(observed & required) / len(required)
            bass_fit = 1.0 if bass_pc == root_pc else 0.60 if bass_pc in chord_pcs else 0.0
            score = (
                0.48 * precision
                + 0.25 * required_fraction
                + 0.12 * completeness
                + 0.10 * bass_fit
                + 0.05 * (root_pc in observed)
            )
            sounding_root, sounding_bass = (root_pc + capo) % 12, (bass_pc + capo) % 12
            sounding_pcs = {(pitch_class + capo) % 12 for pitch_class in chord_pcs}
            candidates.append(
                Candidate(
                    symbol=_symbol(
                        root_pc,
                        bass_pc,
                        template.suffix,
                        chord_pcs,
                        root_names,
                        include_inversions,
                    ),
                    sounding_symbol=_symbol(
                        sounding_root,
                        sounding_bass,
                        template.suffix,
                        sounding_pcs,
                        root_names,
                        include_inversions,
                    ),
                    score=score,
                    root_pc=root_pc,
                    bass_pc=bass_pc,
                    quality=template.quality,
                    evidence_pcs=tuple(sorted(observed)),
                    priority=template.priority,
                    retrieval_id=template.id,
                )
            )
    return sorted(
        candidates,
        key=lambda item: (-item.score, item.priority, item.root_pc, item.symbol),
    )


def _transition_penalty(previous: Candidate, current: Candidate) -> float:
    if previous.symbol == current.symbol:
        return 0.0
    if previous.root_pc == current.root_pc:
        return 0.22
    motion = (current.root_pc - previous.root_pc) % 12
    if motion in (5, 7):
        return 0.48
    if motion in (3, 9):
        return 0.58
    if motion in (2, 10):
        return 0.68
    return 0.82


def decode_candidate_sequence(
    layers: Sequence[Sequence[Candidate]], transition_weight: float
) -> list[Candidate]:
    """Viterbi-decode one uninterrupted candidate trellis."""

    if not layers:
        return []
    if any(not layer for layer in layers):
        raise ValueError("candidate layers must be non-empty")
    costs = [[-math.log(max(item.score, 1e-9)) for item in layers[0]]]
    parents: list[list[int | None]] = [[None] * len(layers[0])]
    for layer_index in range(1, len(layers)):
        layer_costs, layer_parents = [], []
        for candidate in layers[layer_index]:
            options = [
                costs[layer_index - 1][index]
                + transition_weight * _transition_penalty(previous, candidate)
                for index, previous in enumerate(layers[layer_index - 1])
            ]
            parent = min(range(len(options)), key=options.__getitem__)
            layer_costs.append(-math.log(max(candidate.score, 1e-9)) + options[parent])
            layer_parents.append(parent)
        costs.append(layer_costs)
        parents.append(layer_parents)
    indices = [0] * len(layers)
    indices[-1] = min(range(len(costs[-1])), key=costs[-1].__getitem__)
    for index in range(len(layers) - 1, 0, -1):
        indices[index - 1] = int(parents[index][indices[index]])
    return [layers[index][candidate] for index, candidate in enumerate(indices)]


def _candidate_segments(
    eligible: Sequence[object],
    bpm: float,
    capo: int,
    templates: Sequence[ChordTemplate],
    root_names: Sequence[str],
    config: ChordInferenceConfig,
) -> list[list[tuple[float, float, list[Candidate]]]]:
    window_seconds = 60.0 / bpm * config.window_beats
    origin = _value(eligible[0], "start")
    last_onset = max(_value(event, "start") for event in eligible)
    final_end = max(_value(event, "end") for event in eligible)
    window_count = max(1, int(math.floor((last_onset - origin) / window_seconds)) + 1)
    segments: list[list[tuple[float, float, list[Candidate]]]] = []
    current: list[tuple[float, float, list[Candidate]]] = []

    def break_segment() -> None:
        nonlocal current
        if current:
            segments.append(current)
            current = []

    for index in range(window_count):
        start = origin + index * window_seconds
        nominal_end = start + window_seconds
        events = [event for event in eligible if start <= _value(event, "start") < nominal_end]
        pitch_classes = {(int(getattr(event, "pitch")) - capo) % 12 for event in events}
        if len(pitch_classes) < config.min_distinct_pitch_classes:
            break_segment()
            continue
        ranked = _rank_window(
            events, capo, templates, root_names, config.include_inversions
        )
        if not ranked or ranked[0].score < config.min_chord_confidence:
            break_segment()
            continue
        candidates = [
            item for item in ranked if item.score >= ranked[0].score - config.candidate_margin
        ][: config.max_candidates_per_window]
        current.append((start, max(start, min(nominal_end, final_end)), candidates))
    break_segment()
    return segments


def _merge_label(labels: list[ChordLabel], label: ChordLabel) -> None:
    if not labels or labels[-1].symbol != label.symbol or not math.isclose(
        labels[-1].end, label.start, abs_tol=1e-6
    ):
        labels.append(label)
        return
    previous = labels[-1]
    strength = round(
        (previous.confidence * previous.windows + label.confidence)
        / (previous.windows + 1),
        4,
    )
    labels[-1] = replace(
        previous,
        end=label.end,
        confidence=strength,
        evidence_pcs=tuple(sorted(set(previous.evidence_pcs) | set(label.evidence_pcs))),
        windows=previous.windows + 1,
        retrieval_score=strength,
    )


def infer_chords(
    events: Sequence[object],
    timing: object,
    capo: int = 0,
    *,
    config: ChordInferenceConfig | None = None,
    templates: Sequence[ChordTemplate] | None = None,
) -> list[ChordLabel]:
    """Infer conservative capo-relative shapes across arpeggio windows."""

    if not 0 <= capo <= 12:
        raise ValueError("capo must be between 0 and 12")
    config = config or ChordInferenceConfig()
    eligible = sorted(
        (
            event
            for event in events
            if 0 <= int(getattr(event, "pitch")) <= 127
            and _value(event, "end") > _value(event, "start")
            and _value(event, "confidence") >= config.min_note_confidence
        ),
        key=lambda event: (_value(event, "start"), int(getattr(event, "pitch"))),
    )
    if not eligible:
        return []
    bpm = _value(timing, "bpm", 120.0)
    bpm = bpm if math.isfinite(bpm) and bpm > 0 else 120.0
    root_names = load_knowledge(str(KNOWLEDGE_PATH.resolve()))[0]
    selected_templates = tuple(templates or load_chord_templates())
    segments = _candidate_segments(
        eligible, bpm, capo, selected_templates, root_names, config
    )
    labels: list[ChordLabel] = []
    for segment in segments:
        selected = decode_candidate_sequence(
            [window[2] for window in segment], config.transition_weight
        )
        for (start, end, _), candidate in zip(segment, selected):
            _merge_label(
                labels,
                ChordLabel(
                    symbol=candidate.symbol,
                    start=start,
                    end=end,
                    confidence=round(candidate.score, 4),
                    root_pc=candidate.root_pc,
                    bass_pc=candidate.bass_pc,
                    quality=candidate.quality,
                    evidence_pcs=candidate.evidence_pcs,
                    sounding_symbol=candidate.sounding_symbol,
                    retrieval_id=candidate.retrieval_id,
                    retrieval_score=round(candidate.score, 4),
                ),
            )
    return labels

