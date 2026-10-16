#!/usr/bin/env python3
"""CLI and compatibility facade for the local specialist-agent workflow."""

from __future__ import annotations

import argparse
import json
from typing import Literal

from fretwork_agents.roles import (
    _quality_metrics,
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
from fretwork_agents.state import AGENT_TEAM, AgentState, AttemptSnapshot
from fretwork_agents.workflow import build_graph, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local LangGraph fingerstyle transcription agent."
    )
    parser.add_argument("audio")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-stem")
    parser.add_argument("--title", default="Fretwork Transcription")
    parser.add_argument(
        "--backend", choices=("ensemble", "gaps", "basic_pitch"), default="ensemble"
    )
    parser.add_argument("--capo", default="auto", help="0-12 or auto")
    parser.add_argument("--tab-hints")
    parser.add_argument("--max-span", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    capo: int | Literal["auto"]
    if args.capo.lower() == "auto":
        capo = "auto"
    else:
        capo = int(args.capo)
        if not 0 <= capo <= 12:
            parser.error("--capo must be auto or an integer from 0 to 12")
    result = run_agent(
        args.audio,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        title=args.title,
        transcription_backend=args.backend,
        capo=capo,
        tab_hint_path=args.tab_hints,
        max_span=args.max_span,
        min_confidence=args.min_confidence,
        max_attempts=args.max_attempts,
        render_pdf=args.pdf,
    )
    print(
        json.dumps(
            {
                "selected_capo": result["selected_capo"],
                "heuristic_health": result["quality"],
                "outputs": result["outputs"],
                "report": result["report_path"],
                "workflow_log": result["messages"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

