#!/usr/bin/env python3
"""Stable public facade for the Fretwork core.

Implementation is split by responsibility under :mod:`fretwork_core`.  This
module keeps the original import and CLI surface so applications and third-
party scripts do not need a migration just because the internals are cleaner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from fretwork_core.fingering import (
    _center_transition_cost,
    _transition_cost,
    assign_positions,
    cluster_events,
    enumerate_cluster_fingerings,
    parse_tab_hints,
    possible_positions,
    rank_capos,
    separate_voices,
)
from fretwork_core.models import (
    BASS_REGISTER_CEILING,
    CHORD_ONSET_WINDOW,
    MAX_FRET,
    STANDARD_TUNING,
    STRING_NAMES,
    FingeringReport,
    FingeringShape,
    NoteEvent,
    ProposalAudit,
    TimingInfo,
)
from fretwork_core.notation import (
    build_score,
    export_results,
    normalize_musicxml_for_musescore,
    render_ascii_tab,
    write_musicxml,
)
from fretwork_core.transcription import (
    GAPS_TRANSCRIPTION_PROFILES,
    TRANSCRIPTION_PROFILES,
    _gaps_activations,
    estimate_timing,
    load_basic_pitch_events,
    load_gaps_events,
    reconcile_note_proposals,
    reconcile_note_proposals_with_audit,
)


# Backward-compatible private names retained for tests and existing callers.
_load_basic_pitch_note_events = load_basic_pitch_events
_load_gaps_note_events = load_gaps_events


def load_note_events(
    audio_path: str,
    min_confidence: float = 0.30,
    profile: str = "balanced",
    backend: str = "gaps",
) -> list[NoteEvent]:
    """Dispatch to one explicitly selected local acoustic backend."""

    if backend == "gaps":
        return _load_gaps_note_events(audio_path, min_confidence, profile)
    if backend == "basic_pitch":
        return _load_basic_pitch_note_events(audio_path, min_confidence, profile)
    raise ValueError(f"Unknown transcription backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe fingerstyle guitar audio.")
    parser.add_argument("audio")
    parser.add_argument("--backend", choices=("gaps", "basic_pitch"), default="gaps")
    parser.add_argument("--capo", type=int, default=0)
    parser.add_argument("--max-span", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--profile", choices=TRANSCRIPTION_PROFILES, default="balanced")
    parser.add_argument("--out-prefix", default="output")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--tab-hints")
    args = parser.parse_args()

    timing = estimate_timing(args.audio)
    events = load_note_events(
        args.audio, args.min_confidence, args.profile, backend=args.backend
    )
    separate_voices(events)
    hints = parse_tab_hints(args.tab_hints) if args.tab_hints else None
    report = assign_positions(events, args.capo, hints=hints, max_span=args.max_span)
    from chord_retrieval import infer_chords

    chords = infer_chords(events, timing, capo=args.capo)
    paths = export_results(
        events,
        args.out_prefix,
        args.capo,
        timing,
        chords=chords,
        render_pdf=args.pdf,
    )
    print(json.dumps({"fingering": asdict(report), "outputs": paths}, indent=2))


if __name__ == "__main__":
    main()
