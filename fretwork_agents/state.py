"""Shared agent contracts; no orchestration or domain logic lives here."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from chord_retrieval import ChordLabel
from fretwork import FingeringReport, NoteEvent, ProposalAudit, TimingInfo


AGENT_TEAM = [
    {"id": "recording_analyst", "task": "validate audio, timing, and TAB evidence"},
    {"id": "guitar_crnn_specialist", "task": "propose guitar-specific note events"},
    {"id": "basic_pitch_reviewer", "task": "produce an independent acoustic review"},
    {"id": "acoustic_consensus", "task": "reconcile acoustic proposals"},
    {"id": "voice_specialist", "task": "separate bass and upper attacks"},
    {"id": "fingering_specialist", "task": "rank capo and solve the fingering trellis"},
    {"id": "critic_supervisor", "task": "route bounded corrective decodes"},
    {"id": "harmony_rag_specialist", "task": "ground chord-shape labels in local knowledge"},
    {"id": "publishing_specialist", "task": "export notation, TAB, JSON, and audit report"},
]


class AttemptSnapshot(TypedDict):
    attempt: int
    profile: str
    events: list[NoteEvent]
    capo: int
    capo_ranking: list[FingeringReport]
    fingering: FingeringReport
    quality: dict[str, Any]
    reviewer_audit: Optional[ProposalAudit]


class AgentState(TypedDict, total=False):
    audio_path: str
    transcription_backend: Literal["ensemble", "gaps", "basic_pitch"]
    tab_hint_path: Optional[str]
    output_dir: str
    output_stem: str
    title: str
    capo_request: int | Literal["auto"]
    capo_candidates: list[int]
    max_span: int
    min_confidence: float
    max_attempts: int
    render_pdf: bool
    timing: TimingInfo
    hints: Optional[set[tuple[int, int]]]
    attempt: int
    profile: str
    guitar_proposal: list[NoteEvent]
    reviewer_proposal: list[NoteEvent]
    reviewer_audit: Optional[ProposalAudit]
    events: list[NoteEvent]
    chords: list[ChordLabel]
    selected_capo: int
    capo_ranking: list[FingeringReport]
    fingering: FingeringReport
    quality: dict[str, Any]
    retry_profile: Optional[str]
    history: list[AttemptSnapshot]
    messages: list[str]
    outputs: dict[str, Optional[str]]
    report_path: str
