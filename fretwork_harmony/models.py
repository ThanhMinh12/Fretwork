"""Harmony knowledge and inference data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ChordTemplate:
    id: str
    quality: str
    suffix: str
    intervals: tuple[int, ...]
    required: tuple[int, ...]
    aliases: tuple[str, ...]
    description: str
    priority: int = 100

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        for key in ("intervals", "required", "aliases"):
            record[key] = list(record[key])
        return record


@dataclass(frozen=True)
class ChordInferenceConfig:
    window_beats: float = 2.0
    min_note_confidence: float = 0.30
    min_chord_confidence: float = 0.78
    min_distinct_pitch_classes: int = 2
    include_inversions: bool = True
    candidate_margin: float = 0.10
    transition_weight: float = 0.22
    max_candidates_per_window: int = 12

    def __post_init__(self) -> None:
        if self.window_beats <= 0:
            raise ValueError("window_beats must be positive")
        if not 0 <= self.min_note_confidence <= 1:
            raise ValueError("min_note_confidence must be between 0 and 1")
        if not 0 <= self.min_chord_confidence <= 1:
            raise ValueError("min_chord_confidence must be between 0 and 1")
        if self.min_distinct_pitch_classes < 2:
            raise ValueError("at least two pitch classes are required")
        if not 0 <= self.candidate_margin <= 1:
            raise ValueError("candidate_margin must be between 0 and 1")
        if self.transition_weight < 0:
            raise ValueError("transition_weight cannot be negative")
        if self.max_candidates_per_window < 1:
            raise ValueError("max_candidates_per_window must be positive")


@dataclass(frozen=True)
class ChordLabel:
    """Capo-relative hand shape aligned to original recording seconds."""

    symbol: str
    start: float
    end: float
    confidence: float
    root_pc: int
    bass_pc: int
    quality: str
    evidence_pcs: tuple[int, ...]
    windows: int = 1
    sounding_symbol: str | None = None
    retrieval_id: str | None = None
    retrieval_score: float | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["evidence_pcs"] = list(self.evidence_pcs)
        result["shape_chord"] = self.symbol
        result["sounding_chord"] = self.sounding_symbol or self.symbol
        return result


@dataclass(frozen=True)
class Candidate:
    symbol: str
    sounding_symbol: str
    score: float
    root_pc: int
    bass_pc: int
    quality: str
    evidence_pcs: tuple[int, ...]
    priority: int
    retrieval_id: str

