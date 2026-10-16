"""Local acoustic-model adapters, proposal consensus, and timing analysis."""

from __future__ import annotations

import copy
import logging
import math
import os
import sys
from pathlib import Path
from typing import Sequence

from .models import NoteEvent, ProposalAudit, TimingInfo


TRANSCRIPTION_PROFILES = {
    "balanced": {
        "onset_threshold": 0.50,
        "frame_threshold": 0.30,
        "minimum_note_length": 100.0,
    },
    "sensitive": {
        "onset_threshold": 0.40,
        "frame_threshold": 0.24,
        "minimum_note_length": 75.0,
    },
    "precise": {
        "onset_threshold": 0.62,
        "frame_threshold": 0.38,
        "minimum_note_length": 120.0,
    },
}

GAPS_TRANSCRIPTION_PROFILES = {
    "balanced": {
        "onset_threshold": 0.30,
        "offset_threshold": 0.30,
        "frame_threshold": 0.10,
    },
    "sensitive": {
        "onset_threshold": 0.24,
        "offset_threshold": 0.25,
        "frame_threshold": 0.075,
    },
    "precise": {
        "onset_threshold": 0.40,
        "offset_threshold": 0.38,
        "frame_threshold": 0.16,
    },
}

_BASIC_PITCH_MODEL = None
_GAPS_MODEL = None
_GAPS_ACTIVATIONS_CACHE = {}


def load_basic_pitch_events(
    audio_path: str,
    min_confidence: float = 0.35,
    profile: str = "balanced",
) -> list[NoteEvent]:
    """Transcribe polyphonic notes with Spotify's free Basic Pitch model."""

    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model, predict
    except ImportError as exc:  # pragma: no cover - optional ML stack
        raise RuntimeError(
            "Basic Pitch is not installed. Follow the two-step install in README.md."
        ) from exc
    finally:
        logging.disable(previous_logging_disable)

    if profile not in TRANSCRIPTION_PROFILES:
        raise ValueError(f"Unknown transcription profile: {profile}")
    print(f"Transcribing {audio_path} ({profile} profile)...", file=sys.stderr)
    global _BASIC_PITCH_MODEL
    if _BASIC_PITCH_MODEL is None:
        _BASIC_PITCH_MODEL = Model(ICASSP_2022_MODEL_PATH)
        runtime = getattr(_BASIC_PITCH_MODEL, "model_type", "local model")
        print(f"Loaded Basic Pitch runtime: {runtime}", file=sys.stderr)
    _, _, raw_notes = predict(
        audio_path,
        _BASIC_PITCH_MODEL,
        minimum_frequency=80.0,
        maximum_frequency=2000.0,
        **TRANSCRIPTION_PROFILES[profile],
    )
    events = [
        NoteEvent(
            float(start),
            float(end),
            int(pitch),
            float(strength),
            sources=("basic_pitch",),
        )
        for start, end, pitch, strength, _ in raw_notes
        if float(strength) >= min_confidence
    ]
    return sorted(events, key=lambda event: (event.start, event.pitch))


def _gaps_activations(audio_path: str):
    """Run the GAPS guitar CRNN once and cache its temporal activations."""

    try:
        import librosa
        import torch
        from hf_midi_transcription import MidiTranscriptionModel
    except ImportError as exc:  # pragma: no cover - optional ML stack
        raise RuntimeError(
            "The GAPS backend is not installed. Rebuild Docker or select Basic Pitch."
        ) from exc

    path = Path(audio_path)
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cache_key in _GAPS_ACTIVATIONS_CACHE:
        return _GAPS_ACTIVATIONS_CACHE[cache_key]

    global _GAPS_MODEL
    if _GAPS_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        default_batch = "8" if device == "cuda" else "1"
        batch_size = int(os.getenv("FRETWORK_GAPS_BATCH_SIZE", default_batch))
        from .checkpoints import verified_gaps_checkpoint
        checkpoint_path = str(verified_gaps_checkpoint())
        print(f"Loading GAPS guitar CRNN on {device}...", file=sys.stderr)
        _GAPS_MODEL = MidiTranscriptionModel(
            instrument="guitar",
            device=device,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )

    print(f"Transcribing {audio_path} with the GAPS guitar CRNN...", file=sys.stderr)
    samples, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = _GAPS_MODEL.transcriptor.transcribe(samples, None)
    keys = (
        "reg_onset_output",
        "reg_offset_output",
        "frame_output",
        "velocity_output",
    )
    activations = {key: result["output_dict"][key] for key in keys}
    _GAPS_ACTIVATIONS_CACHE.clear()
    _GAPS_ACTIVATIONS_CACHE[cache_key] = activations
    return activations


def load_gaps_events(
    audio_path: str,
    min_confidence: float = 0.30,
    profile: str = "balanced",
) -> list[NoteEvent]:
    """Decode notes from the guitar-specific recurrent GAPS model."""

    del min_confidence  # GAPS profiles apply their own decoder thresholds.
    if profile not in GAPS_TRANSCRIPTION_PROFILES:
        raise ValueError(f"Unknown transcription profile: {profile}")
    try:
        import numpy as np
        from piano_transcription_inference import config
        from piano_transcription_inference.utilities import RegressionPostProcessor
    except ImportError as exc:  # pragma: no cover - optional ML stack
        raise RuntimeError("The GAPS backend is not installed correctly.") from exc

    activations = _gaps_activations(audio_path)
    params = GAPS_TRANSCRIPTION_PROFILES[profile]
    processor = RegressionPostProcessor(
        config.frames_per_second,
        classes_num=config.classes_num,
        onset_threshold=params["onset_threshold"],
        offset_threshold=params["offset_threshold"],
        frame_threshold=params["frame_threshold"],
        pedal_offset_threshold=0.2,
    )
    decoded, _ = processor.output_dict_to_midi_events(dict(activations))
    frame_count = activations["frame_output"].shape[0]
    events: list[NoteEvent] = []
    for item in decoded:
        pitch = int(item["midi_note"])
        pitch_index = pitch - config.begin_note
        if not 0 <= pitch_index < config.classes_num:
            continue
        start = max(0.0, float(item["onset_time"]))
        end = max(start + 0.01, float(item["offset_time"]))
        onset_frame = min(
            frame_count - 1,
            max(0, int(round(start * config.frames_per_second))),
        )
        offset_frame = min(
            frame_count,
            max(onset_frame + 1, int(round(end * config.frames_per_second))),
        )
        onset_strength = float(
            activations["reg_onset_output"][onset_frame, pitch_index]
        )
        active_frames = activations["frame_output"][onset_frame:offset_frame, pitch_index]
        frame_strength = (
            float(np.percentile(active_frames, 75)) if len(active_frames) else 0.0
        )
        strength = max(0.0, min(1.0, onset_strength * 0.7 + frame_strength * 0.3))
        if pitch >= 40:  # Standard guitar's E2 lower bound.
            events.append(
                NoteEvent(start, end, pitch, strength, sources=("gaps_crnn",))
            )
    return sorted(events, key=lambda event: (event.start, event.pitch))


def reconcile_note_proposals(
    guitar_events: Sequence[NoteEvent],
    reviewer_events: Sequence[NoteEvent],
    *,
    onset_tolerance: float = 0.08,
    reviewer_only_threshold: float = 0.82,
) -> list[NoteEvent]:
    """Reconcile a guitar-model proposal with an independent reviewer."""

    events, _ = reconcile_note_proposals_with_audit(
        guitar_events,
        reviewer_events,
        onset_tolerance=onset_tolerance,
        reviewer_only_threshold=reviewer_only_threshold,
    )
    return events


def reconcile_note_proposals_with_audit(
    guitar_events: Sequence[NoteEvent],
    reviewer_events: Sequence[NoteEvent],
    *,
    onset_tolerance: float = 0.08,
    reviewer_only_threshold: float = 0.82,
) -> tuple[list[NoteEvent], ProposalAudit]:
    """Return the consensus plus reproducible reviewer decision counts."""

    primary = sorted(
        copy.deepcopy(guitar_events), key=lambda event: (event.start, event.pitch)
    )
    reviewer = sorted(reviewer_events, key=lambda event: (event.start, event.pitch))
    used_reviewer: set[int] = set()
    merged: list[NoteEvent] = []
    agreements = 0
    for event in primary:
        matching = [
            (abs(candidate.start - event.start), index, candidate)
            for index, candidate in enumerate(reviewer)
            if index not in used_reviewer
            and candidate.pitch == event.pitch
            and abs(candidate.start - event.start) <= onset_tolerance
        ]
        if matching:
            _, index, candidate = min(matching, key=lambda item: item[0])
            used_reviewer.add(index)
            agreements += 1
            event.sources = ("gaps_crnn", "basic_pitch")
            event.confidence = max(event.confidence, candidate.confidence)
        elif not event.sources:
            event.sources = ("gaps_crnn",)
        merged.append(event)

    admitted = 0
    below_threshold = 0
    octave_rejections = 0
    range_rejections = 0
    for index, event in enumerate(reviewer):
        if index in used_reviewer:
            continue
        if event.confidence < reviewer_only_threshold:
            below_threshold += 1
            continue
        octave_conflict = any(
            abs(primary_event.start - event.start) <= onset_tolerance
            and abs(primary_event.pitch - event.pitch) in (12, 24)
            for primary_event in primary
        )
        if octave_conflict:
            octave_rejections += 1
            continue
        if event.pitch < 40:
            range_rejections += 1
            continue
        accepted = copy.deepcopy(event)
        accepted.sources = ("basic_pitch",)
        merged.append(accepted)
        admitted += 1
    audit = ProposalAudit(
        primary_notes=len(primary),
        reviewer_notes=len(reviewer),
        agreements=agreements,
        primary_only=len(primary) - agreements,
        reviewer_only_admitted=admitted,
        reviewer_below_threshold=below_threshold,
        reviewer_octave_conflicts_rejected=octave_rejections,
        reviewer_out_of_range_rejected=range_rejections,
    )
    return sorted(merged, key=lambda event: (event.start, event.pitch)), audit


def estimate_timing(audio_path: str) -> TimingInfo:
    """Estimate global tempo and duration with a safe fallback."""

    try:
        import librosa

        samples, sample_rate = librosa.load(audio_path, sr=22050, mono=True)
        duration = float(librosa.get_duration(y=samples, sr=sample_rate))
        tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate)
        if hasattr(tempo, "item"):
            tempo = tempo.item()
        bpm = float(tempo) if math.isfinite(float(tempo)) and float(tempo) > 0 else 120.0
        expected_beats = max(1.0, duration * bpm / 60.0)
        confidence = min(1.0, len(beat_frames) / expected_beats)
        return TimingInfo(round(bpm, 2), duration, confidence)
    except Exception as exc:
        print(f"Tempo analysis failed ({exc}); using 120 BPM.", file=sys.stderr)
        return TimingInfo()
