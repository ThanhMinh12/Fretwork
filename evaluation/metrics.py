"""Standard note-transcription metrics and reference loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from fretwork import NoteEvent


@dataclass(frozen=True)
class TranscriptionMetrics:
    onset_precision: float
    onset_recall: float
    onset_f1: float
    offset_precision: float
    offset_recall: float
    offset_f1: float
    average_overlap: float
    reference_notes: int
    estimated_notes: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def load_reference_midi(
    path: str | Path,
    *,
    source_start: float,
    duration: float,
) -> list[NoteEvent]:
    """Load onsets inside one source-audio excerpt and shift them to zero."""

    try:
        import pretty_midi
    except ImportError as exc:  # pragma: no cover - optional evaluation stack
        raise RuntimeError("Evaluation requires pretty_midi from requirements.txt") from exc
    end = source_start + duration
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        NoteEvent(
            note.start - source_start,
            min(note.end, end) - source_start,
            note.pitch,
            1.0,
            sources=("gaps_reference",),
        )
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
        if source_start <= note.start < end
    ]
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def _arrays(events: Sequence[NoteEvent], duration: float):
    import numpy as np

    clipped = [event for event in events if 0.0 <= event.start < duration]
    intervals = np.asarray(
        [[event.start, min(duration, max(event.end, event.start + 0.01))] for event in clipped],
        dtype=float,
    ).reshape((-1, 2))
    pitches = np.asarray(
        [440.0 * 2.0 ** ((event.pitch - 69) / 12.0) for event in clipped],
        dtype=float,
    )
    return intervals, pitches


def score_events(
    reference: Sequence[NoteEvent],
    estimate: Sequence[NoteEvent],
    *,
    duration: float,
) -> TranscriptionMetrics:
    """Score pitch/onset and pitch/onset/offset using mir_eval defaults."""

    try:
        from mir_eval.transcription import precision_recall_f1_overlap
    except ImportError as exc:  # pragma: no cover - optional evaluation stack
        raise RuntimeError("Evaluation requires mir_eval from requirements.txt") from exc
    ref_intervals, ref_pitches = _arrays(reference, duration)
    est_intervals, est_pitches = _arrays(estimate, duration)
    onset_p, onset_r, onset_f1, _ = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        offset_ratio=None,
    )
    offset_p, offset_r, offset_f1, overlap = precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
    )
    return TranscriptionMetrics(
        onset_p,
        onset_r,
        onset_f1,
        offset_p,
        offset_r,
        offset_f1,
        overlap,
        len(reference),
        len(estimate),
    )


def macro_average(rows: Sequence[TranscriptionMetrics]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    keys = (
        "onset_precision",
        "onset_recall",
        "onset_f1",
        "offset_precision",
        "offset_recall",
        "offset_f1",
        "average_overlap",
    )
    result: dict[str, float | int] = {
        key: sum(getattr(row, key) for row in rows) / len(rows) for key in keys
    }
    result.update(
        clips=len(rows),
        reference_notes=sum(row.reference_notes for row in rows),
        estimated_notes=sum(row.estimated_notes for row in rows),
    )
    return result


def macro_average_dicts(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate serialized metric rows without loading model artifacts again."""

    return macro_average([TranscriptionMetrics(**row) for row in rows])
