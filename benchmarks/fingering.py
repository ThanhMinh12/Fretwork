"""Benchmark full-song fingering optimization on deterministic passages."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time

from fretwork import NoteEvent, assign_positions, separate_voices


PATTERN = (
    (48, 55, 60, 64),
    (52,),
    (55,),
    (59, 62),
    (50, 57, 62, 66),
    (54,),
    (57,),
    (61, 64),
)


def make_passage(note_count: int) -> list[NoteEvent]:
    """Build repeatable bass, chord, and upper-note clusters."""

    events: list[NoteEvent] = []
    onset = 0.0
    pattern_index = 0
    while len(events) < note_count:
        pitches = PATTERN[pattern_index % len(PATTERN)]
        for offset, pitch in enumerate(pitches[: note_count - len(events)]):
            events.append(NoteEvent(onset + offset * 0.003, onset + 0.14, pitch, 0.9))
        onset += 0.18
        pattern_index += 1
    separate_voices(events)
    return events


def benchmark(note_count: int, repeats: int, warmups: int) -> dict[str, object]:
    passage = make_passage(note_count)
    for _ in range(warmups):
        assign_positions(copy.deepcopy(passage))
    samples = []
    last_report = None
    for _ in range(repeats):
        events = copy.deepcopy(passage)
        started = time.perf_counter()
        last_report = assign_positions(events)
        samples.append((time.perf_counter() - started) * 1000.0)
    assert last_report is not None
    return {
        "notes": note_count,
        "clusters": len(last_report.candidate_counts),
        "repeats": repeats,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "assigned_ratio": last_report.assigned_ratio,
        "max_candidates_per_cluster": max(last_report.candidate_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, nargs="+", default=(100, 250, 500, 1000))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    result = {
        "benchmark": "deterministic mixed fingerstyle fingering optimization",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "results": [benchmark(count, args.repeats, args.warmups) for count in args.notes],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
