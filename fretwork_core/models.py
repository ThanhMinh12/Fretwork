"""Shared domain models and guitar constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Standard tuning open-string MIDI pitches, low to high (string 6 -> string 1).
STANDARD_TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
STRING_NAMES = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
MAX_FRET = 19
BASS_REGISTER_CEILING = 55
CHORD_ONSET_WINDOW = 0.06


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int  # Sounding MIDI pitch; capo affects fingering only.
    confidence: float  # Uncalibrated model-strength proxy.
    voice: str = "melody"
    string: Optional[int] = None
    fret: Optional[int] = None  # Relative to the capo, like ordinary TAB.
    sources: tuple[str, ...] = ()


@dataclass
class TimingInfo:
    bpm: float = 120.0
    duration: float = 0.0
    beat_confidence: float = 0.0


@dataclass
class FingeringReport:
    capo: int
    score: float
    assigned_ratio: float
    open_string_ratio: float
    average_fret: float
    average_movement: float
    stretch_violations: int
    unassigned_notes: int
    optimization_cost: float = 0.0
    candidate_counts: list[int] = field(default_factory=list)


@dataclass
class FingeringShape:
    positions: list[tuple[Optional[int], Optional[int]]]
    local_cost: float
    center: Optional[float]
    span: int
    violates_stretch: bool = False


@dataclass(frozen=True)
class ProposalAudit:
    """Count every decision made while reconciling two acoustic proposals."""

    primary_notes: int
    reviewer_notes: int
    agreements: int
    primary_only: int
    reviewer_only_admitted: int
    reviewer_below_threshold: int
    reviewer_octave_conflicts_rejected: int
    reviewer_out_of_range_rejected: int

    @property
    def interventions(self) -> int:
        return (
            self.reviewer_only_admitted
            + self.reviewer_below_threshold
            + self.reviewer_octave_conflicts_rejected
            + self.reviewer_out_of_range_rejected
        )
