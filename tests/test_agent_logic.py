import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fretwork import FingeringReport, NoteEvent, TimingInfo
from fretwork_agent import _quality_metrics, build_graph, run_agent, select_best_attempt


class CriticRoutingTests(unittest.TestCase):
    def _state(self, profile="balanced"):
        events = [
            NoteEvent(index * 0.2, index * 0.2 + 0.1, 52 + index, 0.7)
            for index in range(10)
        ]
        return {
            "events": events,
            "timing": TimingInfo(bpm=120, duration=2.0, beat_confidence=0.8),
            "profile": profile,
            "fingering": FingeringReport(
                capo=0,
                score=70,
                assigned_ratio=0.98,
                open_string_ratio=0.1,
                average_fret=6,
                average_movement=2,
                stretch_violations=3,
                unassigned_notes=1,
            ),
        }

    def test_physical_violations_request_precise_pass(self):
        metrics, retry_profile = _quality_metrics(self._state())
        self.assertEqual(retry_profile, "precise")
        self.assertTrue(any("stretch" in issue for issue in metrics["issues"]))
        self.assertTrue(any("could not be assigned" in issue for issue in metrics["issues"]))

    def test_precise_pass_does_not_fall_back_to_sensitive(self):
        _, retry_profile = _quality_metrics(self._state(profile="precise"))
        self.assertIsNone(retry_profile)

    def test_attempt_selector_does_not_choose_sparse_pass_for_confidence_alone(self):
        fingering = self._state()["fingering"]
        baseline_events = [NoteEvent(index * 0.1, index * 0.1 + 0.1, 60, 0.7) for index in range(20)]
        sparse_events = [NoteEvent(index * 0.2, index * 0.2 + 0.1, 60, 0.99) for index in range(8)]
        base_quality = {
            "score": 0.82,
            "notes_per_second": 2.0,
            "maximum_polyphony": 2,
            "assigned_ratio": 1.0,
            "stretch_violations": 0,
            "issues": [],
        }
        precise_quality = {**base_quality, "score": 0.95}
        state = {
            "history": [
                {
                    "attempt": 0,
                    "profile": "balanced",
                    "events": baseline_events,
                    "capo": 0,
                    "capo_ranking": [fingering],
                    "fingering": fingering,
                    "quality": base_quality,
                },
                {
                    "attempt": 1,
                    "profile": "precise",
                    "events": sparse_events,
                    "capo": 0,
                    "capo_ranking": [fingering],
                    "fingering": fingering,
                    "quality": precise_quality,
                },
            ],
            "messages": [],
        }
        selected = select_best_attempt(state)
        self.assertEqual(selected["profile"], "balanced")
        self.assertEqual(len(selected["events"]), 20)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph is optional")
    def test_specialist_graph_compiles_with_named_agent_contracts(self):
        nodes = set(build_graph().get_graph().nodes)
        self.assertTrue(
            {
                "guitar_crnn_specialist",
                "basic_pitch_reviewer",
                "acoustic_consensus",
                "critic_supervisor",
                "harmony_rag_specialist",
                "publishing_specialist",
            }.issubset(nodes)
        )

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "LangGraph is optional")
    def test_graph_executes_two_model_review_and_harmony_handoff(self):
        gaps = [
            NoteEvent(0.00, 0.30, 48, 0.92, sources=("gaps_crnn",)),
            NoteEvent(0.25, 0.55, 55, 0.91, sources=("gaps_crnn",)),
            NoteEvent(0.50, 0.80, 64, 0.90, sources=("gaps_crnn",)),
        ]
        reviewer = [
            NoteEvent(event.start + 0.01, event.end, event.pitch, 0.94, sources=("basic_pitch",))
            for event in gaps
        ]

        def proposal(_path, min_confidence, profile, backend):
            self.assertGreaterEqual(min_confidence, 0.0)
            self.assertIn(profile, {"balanced", "sensitive", "precise"})
            return gaps if backend == "gaps" else reviewer

        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "take.wav"
            audio.write_bytes(b"placeholder")
            fake_outputs = {
                "musicxml": str(Path(directory) / "take.musicxml"),
                "tab": str(Path(directory) / "take_tab.txt"),
                "events": str(Path(directory) / "take_events.json"),
                "chords": str(Path(directory) / "take_chords.json"),
                "pdf": None,
            }
            with (
                patch(
                    "fretwork_agents.roles.estimate_timing",
                    return_value=TimingInfo(120, 1, 0.9),
                ),
                patch("fretwork_agents.roles.load_note_events", side_effect=proposal),
                patch("fretwork_agents.roles.export_results", return_value=fake_outputs),
            ):
                result = run_agent(
                    str(audio), output_dir=directory, max_attempts=1, capo=0
                )
        self.assertEqual(result["transcription_backend"], "ensemble")
        self.assertTrue(all(len(event.sources) == 2 for event in result["events"]))
        self.assertEqual(result["reviewer_audit"].agreements, 3)
        self.assertEqual([chord.symbol for chord in result["chords"]], ["C"])
        trace = " ".join(result["messages"])
        self.assertIn("Guitar CRNN specialist proposed", trace)
        self.assertIn("Basic Pitch reviewer proposed", trace)
        self.assertIn("Acoustic consensus agent accepted", trace)


if __name__ == "__main__":
    unittest.main()
