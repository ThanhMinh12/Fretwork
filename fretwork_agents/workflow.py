"""LangGraph topology and public workflow runner."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from .roles import (
    acoustic_consensus_agent,
    basic_pitch_reviewer,
    critique_draft,
    export_draft,
    guitar_crnn_specialist,
    harmony_rag_specialist,
    inspect_recording,
    interpret_voices,
    optimise_fingering,
    prepare_retry,
    retry_or_finish,
    select_best_attempt,
)
from .state import AgentState


def build_graph():
    """Compile the inspectable specialist workflow."""

    try:
        from langchain_core.runnables import RunnableLambda
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - optional integration stack
        raise RuntimeError(
            "LangGraph is not installed. Install the dependencies from README.md."
        ) from exc

    graph = StateGraph(AgentState)
    nodes = {
        "recording_analyst": inspect_recording,
        "guitar_crnn_specialist": guitar_crnn_specialist,
        "basic_pitch_reviewer": basic_pitch_reviewer,
        "acoustic_consensus": acoustic_consensus_agent,
        "voice_specialist": interpret_voices,
        "fingering_specialist": optimise_fingering,
        "critic_supervisor": critique_draft,
        "revision_router": prepare_retry,
        "attempt_selector": select_best_attempt,
        "harmony_rag_specialist": harmony_rag_specialist,
        "publishing_specialist": export_draft,
    }
    for name, function in nodes.items():
        graph.add_node(name, RunnableLambda(function))

    path = [
        "recording_analyst",
        "guitar_crnn_specialist",
        "basic_pitch_reviewer",
        "acoustic_consensus",
        "voice_specialist",
        "fingering_specialist",
        "critic_supervisor",
    ]
    graph.add_edge(START, path[0])
    for source, target in zip(path, path[1:]):
        graph.add_edge(source, target)
    graph.add_conditional_edges(
        "critic_supervisor",
        retry_or_finish,
        {"retry": "revision_router", "select_best": "attempt_selector"},
    )
    graph.add_edge("revision_router", "guitar_crnn_specialist")
    graph.add_edge("attempt_selector", "harmony_rag_specialist")
    graph.add_edge("harmony_rag_specialist", "publishing_specialist")
    graph.add_edge("publishing_specialist", END)
    return graph.compile()


def run_agent(
    audio_path: str,
    *,
    output_dir: str = "outputs",
    output_stem: Optional[str] = None,
    title: str = "Fretwork Transcription",
    transcription_backend: Literal["ensemble", "gaps", "basic_pitch"] = "ensemble",
    capo: int | Literal["auto"] = "auto",
    tab_hint_path: Optional[str] = None,
    max_span: int = 4,
    min_confidence: float = 0.30,
    max_attempts: int = 3,
    render_pdf: bool = False,
) -> AgentState:
    initial: AgentState = {
        "audio_path": audio_path,
        "transcription_backend": transcription_backend,
        "tab_hint_path": tab_hint_path,
        "output_dir": output_dir,
        "output_stem": output_stem or Path(audio_path).stem,
        "title": title,
        "capo_request": capo,
        "capo_candidates": list(range(8)),
        "max_span": max_span,
        "min_confidence": min_confidence,
        "max_attempts": max(1, min(3, max_attempts)),
        "render_pdf": render_pdf,
        "messages": [],
    }
    return build_graph().invoke(initial, {"recursion_limit": 30})

