"""Run the held-out GAPS ablation and write an auditable JSON report."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from fretwork import (
    NoteEvent,
    load_note_events,
    reconcile_note_proposals_with_audit,
)

from .metrics import load_reference_midi, macro_average_dicts, score_events


def _event_dict(event: NoteEvent) -> dict[str, object]:
    return {
        "start": event.start,
        "end": event.end,
        "pitch": event.pitch,
        "confidence": event.confidence,
        "sources": list(event.sources),
    }


def _load_events(path: Path) -> list[NoteEvent]:
    return [NoteEvent(**{**row, "sources": tuple(row["sources"])}) for row in json.loads(path.read_text())]


def _predict(audio: Path, backend: str, profile: str, cache: Path) -> tuple[list[NoteEvent], float, bool]:
    cache_path = cache / f"{audio.stem}.{backend}.{profile}.json"
    if cache_path.exists():
        return _load_events(cache_path), 0.0, True
    started = time.perf_counter()
    events = load_note_events(str(audio), profile=profile, backend=backend)
    elapsed = time.perf_counter() - started
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps([_event_dict(event) for event in events]), encoding="utf-8")
    return events, elapsed, False


def run(manifest_path: Path, data_dir: Path, cache_dir: Path, profile: str) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = float(manifest["excerpt_start_seconds"])
    duration = float(manifest["excerpt_duration_seconds"])
    clips = []
    for index, item in enumerate(manifest["items"], start=1):
        identity = item["id"]
        audio = data_dir / "audio" / f"{identity}.wav"
        midi = data_dir / "midi" / f"{identity}.mid"
        if not audio.is_file() or not midi.is_file():
            raise FileNotFoundError(
                f"Missing {identity}; run python -m evaluation.download_gaps first"
            )
        print(f"[{index:02d}/{len(manifest['items'])}] evaluating {identity}")
        reference = load_reference_midi(midi, source_start=start, duration=duration)
        gaps, gaps_seconds, gaps_cached = _predict(audio, "gaps", profile, cache_dir)
        basic, basic_seconds, basic_cached = _predict(audio, "basic_pitch", profile, cache_dir)
        ensemble, audit = reconcile_note_proposals_with_audit(gaps, basic)
        naive_union, naive_audit = reconcile_note_proposals_with_audit(
            gaps, basic, reviewer_only_threshold=0.0
        )
        models = {
            "gaps": score_events(reference, gaps, duration=duration),
            "basic_pitch": score_events(reference, basic, duration=duration),
            "ensemble": score_events(reference, ensemble, duration=duration),
            "naive_union": score_events(reference, naive_union, duration=duration),
        }
        clips.append(
            {
                "id": identity,
                "title": item["title"],
                "metrics": {name: metric.to_dict() for name, metric in models.items()},
                "reviewer_audit": asdict(audit) | {"interventions": audit.interventions},
                "naive_union_audit": asdict(naive_audit)
                | {"interventions": naive_audit.interventions},
                "inference_seconds": {"gaps": gaps_seconds, "basic_pitch": basic_seconds},
                "cache_hits": {"gaps": gaps_cached, "basic_pitch": basic_cached},
            }
        )

    aggregate = {
        name: macro_average_dicts([row["metrics"][name] for row in clips])
        for name in ("gaps", "basic_pitch", "ensemble", "naive_union")
    }
    gaps_f1 = [row["metrics"]["gaps"]["onset_f1"] for row in clips]
    ensemble_f1 = [row["metrics"]["ensemble"]["onset_f1"] for row in clips]
    delta = [ensemble - gaps for ensemble, gaps in zip(ensemble_f1, gaps_f1)]
    return {
        "status": "complete",
        "benchmark": "GAPS official test split / fixed 30-second excerpts",
        "dataset_revision": manifest["dataset_revision"],
        "profile": profile,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "aggregate_macro": aggregate,
        "ensemble_vs_gaps": {
            "onset_f1_percentage_point_delta": (aggregate["ensemble"]["onset_f1"] - aggregate["gaps"]["onset_f1"]) * 100.0,
            "clips_improved": sum(value > 1e-12 for value in delta),
            "clips_unchanged": sum(abs(value) <= 1e-12 for value in delta),
            "clips_degraded": sum(value < -1e-12 for value in delta),
        },
        "conservative_vs_naive_union": {
            "onset_f1_percentage_points_preserved": (
                aggregate["ensemble"]["onset_f1"]
                - aggregate["naive_union"]["onset_f1"]
            )
            * 100.0,
            "offset_f1_percentage_points_preserved": (
                aggregate["ensemble"]["offset_f1"]
                - aggregate["naive_union"]["offset_f1"]
            )
            * 100.0,
        },
        "reviewer_totals": {
            key: sum(row["reviewer_audit"][key] for row in clips)
            for key in clips[0]["reviewer_audit"]
        },
        "naive_union_totals": {
            key: sum(row["naive_union_audit"][key] for row in clips)
            for key in clips[0]["naive_union_audit"]
        },
        "clips": clips,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("gaps_test_15.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("evaluation/data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("evaluation/cache"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/gaps_test_15.json"))
    parser.add_argument("--profile", choices=("balanced", "sensitive", "precise"), default="balanced")
    args = parser.parse_args()
    report = run(args.manifest, args.data_dir, args.cache_dir, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate_macro": report["aggregate_macro"],
                "ensemble_vs_gaps": report["ensemble_vs_gaps"],
                "conservative_vs_naive_union": report[
                    "conservative_vs_naive_union"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
