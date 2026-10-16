"""Specialist-agent node implementations.

Each function consumes typed shared state and returns only the artifacts it
owns. Graph topology and CLI concerns live elsewhere.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Optional

from chord_retrieval import infer_chords
from fretwork import (
    assign_positions,
    cluster_events,
    estimate_timing,
    export_results,
    load_note_events,
    parse_tab_hints,
    rank_capos,
    reconcile_note_proposals_with_audit,
    separate_voices,
)

from .state import AGENT_TEAM, AgentState, AttemptSnapshot


def _messages(state: AgentState, message: str) -> list[str]:
    return [*state.get("messages", []), message]


def inspect_recording(state: AgentState) -> dict[str, Any]:
    audio = Path(state["audio_path"])
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio}")
    if audio.stat().st_size == 0:
        raise ValueError("The uploaded audio file is empty")
    hints = parse_tab_hints(state["tab_hint_path"]) if state.get("tab_hint_path") else None
    timing = estimate_timing(str(audio))
    hint_message = f" and loaded {len(hints)} TAB hints" if hints else ""
    return {
        "timing": timing,
        "hints": hints,
        "attempt": 0,
        "profile": "balanced",
        "history": [],
        "messages": _messages(
            state,
            f"Inspected {audio.name}; selected {state['transcription_backend']} backend; "
            f"estimated {timing.bpm:.1f} BPM{hint_message}",
        ),
    }


def _acoustic_proposal(
    state: AgentState,
    *,
    backend: Literal["gaps", "basic_pitch"],
    state_key: Literal["guitar_proposal", "reviewer_proposal"],
    label: str,
) -> dict[str, Any]:
    selected = state["transcription_backend"]
    enabled = selected == "ensemble" or selected == backend
    events = (
        load_note_events(
            state["audio_path"],
            min_confidence=state.get("min_confidence", 0.30),
            profile=state["profile"],
            backend=backend,
        )
        if enabled
        else []
    )
    outcome = f"proposed {len(events)} notes" if enabled else "was skipped"
    return {
        state_key: events,
        "messages": _messages(
            state,
            f"{label} {outcome} for attempt {state['attempt'] + 1} "
            f"({state['profile']})",
        ),
    }


def guitar_crnn_specialist(state: AgentState) -> dict[str, Any]:
    return _acoustic_proposal(
        state,
        backend="gaps",
        state_key="guitar_proposal",
        label="Guitar CRNN specialist",
    )


def basic_pitch_reviewer(state: AgentState) -> dict[str, Any]:
    return _acoustic_proposal(
        state,
        backend="basic_pitch",
        state_key="reviewer_proposal",
        label="Basic Pitch reviewer",
    )


def acoustic_consensus_agent(state: AgentState) -> dict[str, Any]:
    guitar = state.get("guitar_proposal", [])
    reviewer = state.get("reviewer_proposal", [])
    if state["transcription_backend"] == "ensemble":
        events, audit = reconcile_note_proposals_with_audit(guitar, reviewer)
    elif state["transcription_backend"] == "gaps":
        events = copy.deepcopy(guitar)
        audit = None
    else:
        events = copy.deepcopy(reviewer)
        audit = None
    agreements = audit.agreements if audit else 0
    intervention_message = (
        f"; admitted {audit.reviewer_only_admitted} reviewer-only notes and blocked "
        f"{audit.reviewer_octave_conflicts_rejected} octave conflicts"
        if audit
        else ""
    )
    return {
        "events": events,
        "reviewer_audit": audit,
        "messages": _messages(
            state,
            f"Acoustic consensus agent accepted {len(events)} notes; "
            f"{agreements} were supported by both models{intervention_message}",
        ),
    }


def interpret_voices(state: AgentState) -> dict[str, Any]:
    events = separate_voices(state["events"])
    bass_count = sum(event.voice == "bass" for event in events)
    return {
        "events": events,
        "messages": _messages(
            state, f"Voice specialist separated {bass_count} bass attacks from upper notes"
        ),
    }


def optimise_fingering(state: AgentState) -> dict[str, Any]:
    events = state["events"]
    hints = state.get("hints")
    max_span = state.get("max_span", 4)
    requested = state.get("capo_request", "auto")
    if requested == "auto":
        ranking = rank_capos(
            events,
            hints=hints,
            max_span=max_span,
            candidates=state.get("capo_candidates", list(range(8))),
        )
        capo = ranking[0].capo if ranking else 0
    else:
        capo, ranking = int(requested), []
    fingering = assign_positions(events, capo, hints=hints, max_span=max_span)
    return {
        "events": events,
        "selected_capo": capo,
        "capo_ranking": ranking or [fingering],
        "fingering": fingering,
        "messages": _messages(
            state,
            f"Fingering specialist optimised the passage at capo {capo} "
            f"({fingering.assigned_ratio:.0%} assigned)",
        ),
    }


def _quality_metrics(state: AgentState) -> tuple[dict[str, Any], Optional[str]]:
    events, fingering = state["events"], state["fingering"]
    duration = state["timing"].duration
    if duration <= 0 and events:
        duration = max(event.end for event in events)
    density = len(events) / max(duration, 0.001)
    clusters = cluster_events(events)
    polyphony = max((len(cluster) for cluster in clusters), default=0)
    strength = sum(event.confidence for event in events) / len(events) if events else 0.0
    density_score = min(1.0, density / 0.45) if density < 0.45 else 1.0
    if density > 12.0:
        density_score = max(0.0, 1.0 - (density - 12.0) / 12.0)
    polyphony_score = max(0.0, 1.0 - max(0, polyphony - 6) / 6.0)
    stretch_score = max(
        0.0, 1.0 - fingering.stretch_violations / max(1, len(clusters))
    )
    score = (
        strength * 0.34
        + fingering.assigned_ratio * 0.34
        + density_score * 0.12
        + polyphony_score * 0.10
        + stretch_score * 0.10
    )

    issues: list[str] = []
    retry: Optional[str] = None
    if not events or density < 0.45:
        issues.append("the transcription is unusually sparse")
        retry = "sensitive"
    if strength < 0.48:
        issues.append("mean model detection strength is low")
        retry = retry or "sensitive"
    if density > 12.0 or polyphony > 6:
        issues.append("the transcription is unusually dense/noisy")
        retry = "precise"
    if fingering.assigned_ratio < 0.97:
        issues.append("some notes are outside the selected guitar/capo range")
        retry = retry or "precise"
    elif fingering.unassigned_notes:
        issues.append(
            f"{fingering.unassigned_notes} detected notes could not be assigned to strings"
        )
        if fingering.assigned_ratio < 0.995:
            retry = retry or "precise"
    if fingering.stretch_violations:
        issues.append(f"{fingering.stretch_violations} chord shapes exceed the stretch limit")
        retry = "precise"
    if retry == state["profile"] or (score >= 0.78 and not issues):
        retry = None
    return (
        {
            "score": round(score, 4),
            "mean_detection_strength": round(strength, 4),
            "notes_per_second": round(density, 3),
            "maximum_polyphony": polyphony,
            "assigned_ratio": round(fingering.assigned_ratio, 4),
            "stretch_violations": fingering.stretch_violations,
            "issues": issues,
        },
        retry,
    )


def critique_draft(state: AgentState) -> dict[str, Any]:
    quality, retry = _quality_metrics(state)
    snapshot: AttemptSnapshot = {
        "attempt": state["attempt"],
        "profile": state["profile"],
        "events": copy.deepcopy(state["events"]),
        "capo": state["selected_capo"],
        "capo_ranking": copy.deepcopy(state["capo_ranking"]),
        "fingering": copy.deepcopy(state["fingering"]),
        "quality": quality,
        "reviewer_audit": copy.deepcopy(state.get("reviewer_audit")),
    }
    outcome = f"heuristic health {quality['score']:.0%}"
    if quality["issues"]:
        outcome += "; " + "; ".join(quality["issues"])
    return {
        "quality": quality,
        "retry_profile": retry,
        "history": [*state.get("history", []), snapshot],
        "messages": _messages(state, f"Critic supervisor scored attempt: {outcome}"),
    }


def retry_or_finish(state: AgentState) -> Literal["retry", "select_best"]:
    can_retry = state["attempt"] + 1 < state.get("max_attempts", 3)
    return "retry" if can_retry and state.get("retry_profile") else "select_best"


def prepare_retry(state: AgentState) -> dict[str, Any]:
    profile = state.get("retry_profile") or "sensitive"
    return {
        "attempt": state["attempt"] + 1,
        "profile": profile,
        "messages": _messages(
            state, f"Critic requested one bounded retry with the {profile} profile"
        ),
    }


def _artifact_severity(item: AttemptSnapshot) -> float:
    quality = item["quality"]
    return (
        float(quality["stretch_violations"]) * 3.0
        + max(0.0, float(quality["maximum_polyphony"]) - 6.0) * 3.0
        + max(0.0, 1.0 - float(quality["assigned_ratio"])) * 10.0
        + max(0.0, float(quality["notes_per_second"]) - 12.0)
    )


def select_best_attempt(state: AgentState) -> dict[str, Any]:
    baseline = state["history"][0]
    best, reason = baseline, "kept the model's validated balanced decode"
    baseline_count = max(1, len(baseline["events"]))
    baseline_issues = " ".join(baseline["quality"].get("issues", [])).lower()
    for candidate in state["history"][1:]:
        count = len(candidate["events"])
        if candidate["profile"] == "precise":
            if (
                _artifact_severity(candidate) < _artifact_severity(best)
                and count >= baseline_count * 0.72
                and candidate["quality"]["score"] >= best["quality"]["score"] - 0.05
            ):
                best = candidate
                reason = "selected a precise decode that removed physical/noise failures"
        elif candidate["profile"] == "sensitive":
            recall_needed = "sparse" in baseline_issues or "strength" in baseline_issues
            if (
                recall_needed
                and count > len(best["events"])
                and count <= max(baseline_count * 1.75, baseline_count + 20)
                and _artifact_severity(candidate) <= _artifact_severity(best)
            ):
                best = candidate
                reason = "selected a sensitive decode with bounded additional evidence"
    return {
        "events": best["events"],
        "selected_capo": best["capo"],
        "capo_ranking": best["capo_ranking"],
        "fingering": best["fingering"],
        "quality": best["quality"],
        "reviewer_audit": best.get("reviewer_audit"),
        "profile": best["profile"],
        "messages": _messages(
            state,
            f"Attempt selector chose {best['profile']}: {reason} "
            f"(health {best['quality']['score']:.0%})",
        ),
    }


def harmony_rag_specialist(state: AgentState) -> dict[str, Any]:
    chords = infer_chords(state["events"], state["timing"], capo=state["selected_capo"])
    return {
        "chords": chords,
        "messages": _messages(
            state,
            f"Harmony RAG specialist retrieved local templates and grounded "
            f"{len(chords)} capo-relative chord labels",
        ),
    }


def export_draft(state: AgentState) -> dict[str, Any]:
    output_dir = Path(state.get("output_dir", "outputs"))
    stem = re.sub(
        r"[^A-Za-z0-9._-]+", "_", state.get("output_stem", "transcription")
    ).strip("._-") or "transcription"
    outputs = export_results(
        state["events"],
        str(output_dir / stem),
        state["selected_capo"],
        state["timing"],
        chords=state.get("chords", []),
        title=state.get("title", "Fretwork Transcription"),
        render_pdf=state.get("render_pdf", False),
    )
    report_path = output_dir / f"{stem}_report.json"
    final_messages = _messages(
        state, "Publishing specialist exported MusicXML, chord-aware TAB, events, and report"
    )
    report = {
        "audio": Path(state["audio_path"]).name,
        "transcription_backend": state["transcription_backend"],
        "selected_profile": state["profile"],
        "selected_capo": state["selected_capo"],
        "capo_is_inferred": state.get("capo_request", "auto") == "auto",
        "capo_ranking": [asdict(item) for item in state["capo_ranking"][:8]],
        "timing": asdict(state["timing"]),
        "fingering": asdict(state["fingering"]),
        "heuristic_health": state["quality"],
        "reviewer_audit": (
            asdict(state["reviewer_audit"]) | {
                "interventions": state["reviewer_audit"].interventions
            }
            if state.get("reviewer_audit")
            else None
        ),
        "attempts": [
            {
                "attempt": item["attempt"] + 1,
                "profile": item["profile"],
                "capo": item["capo"],
                "heuristic_health": item["quality"],
            }
            for item in state["history"]
        ],
        "chords": [chord.to_dict() for chord in state.get("chords", [])],
        "agent_team": AGENT_TEAM,
        "collaboration_contract": (
            "specialists write typed artifacts to shared LangGraph state; the critic "
            "may request a bounded re-decode; publishing consumes reviewed artifacts"
        ),
        "workflow_log": final_messages,
        "outputs": outputs,
        "warning": (
            "ML-assisted first draft: check pitch, rhythm, and capo by ear before use."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "outputs": outputs,
        "report_path": str(report_path),
        "messages": final_messages,
    }
