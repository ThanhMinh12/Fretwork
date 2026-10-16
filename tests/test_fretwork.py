import importlib.util
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

from chord_retrieval import ChordLabel
from fretwork import (
    NoteEvent,
    TimingInfo,
    _transition_cost,
    assign_positions,
    cluster_events,
    enumerate_cluster_fingerings,
    export_results,
    load_note_events,
    parse_tab_hints,
    possible_positions,
    reconcile_note_proposals,
    reconcile_note_proposals_with_audit,
    rank_capos,
    render_ascii_tab,
    separate_voices,
    normalize_musicxml_for_musescore,
)


class FretboardTests(unittest.TestCase):
    def test_transcription_backend_dispatch_is_explicit(self):
        sentinel = [NoteEvent(0.0, 0.2, 64, 0.9)]
        with patch("fretwork._load_gaps_note_events", return_value=sentinel) as gaps:
            self.assertIs(load_note_events("take.wav", backend="gaps"), sentinel)
            gaps.assert_called_once_with("take.wav", 0.30, "balanced")
        with patch("fretwork._load_basic_pitch_note_events", return_value=sentinel) as basic:
            self.assertIs(load_note_events("take.wav", backend="basic_pitch"), sentinel)
            basic.assert_called_once_with("take.wav", 0.30, "balanced")
        with self.assertRaises(ValueError):
            load_note_events("take.wav", backend="pretend_lstm")

    def test_acoustic_agents_reconcile_agreement_and_strong_reviewer_notes(self):
        guitar = [
            NoteEvent(0.00, 0.30, 64, 0.60, sources=("gaps_crnn",)),
            NoteEvent(0.50, 0.80, 67, 0.70, sources=("gaps_crnn",)),
        ]
        reviewer = [
            NoteEvent(0.03, 0.31, 64, 0.90, sources=("basic_pitch",)),
            NoteEvent(0.52, 0.79, 79, 0.99, sources=("basic_pitch",)),  # octave conflict
            NoteEvent(1.00, 1.20, 69, 0.90, sources=("basic_pitch",)),
            NoteEvent(1.50, 1.70, 71, 0.60, sources=("basic_pitch",)),
        ]
        result = reconcile_note_proposals(guitar, reviewer)
        self.assertEqual([event.pitch for event in result], [64, 67, 69])
        self.assertEqual(result[0].sources, ("gaps_crnn", "basic_pitch"))
        self.assertAlmostEqual(result[0].confidence, 0.90)
        self.assertEqual(result[-1].sources, ("basic_pitch",))

        _, audit = reconcile_note_proposals_with_audit(guitar, reviewer)
        self.assertEqual(audit.agreements, 1)
        self.assertEqual(audit.primary_only, 1)
        self.assertEqual(audit.reviewer_only_admitted, 1)
        self.assertEqual(audit.reviewer_octave_conflicts_rejected, 1)
        self.assertEqual(audit.reviewer_below_threshold, 1)
        self.assertEqual(audit.interventions, 3)

    def test_capo_positions_are_relative(self):
        # F#4 is open first string with capo 2, but fret 2 without it.
        self.assertIn((1, 0), possible_positions(66, capo=2))
        self.assertIn((1, 2), possible_positions(66, capo=0))

    def test_nearby_onsets_use_one_chord_shape(self):
        events = [
            NoteEvent(0.000, 0.5, 50, 0.9),
            NoteEvent(0.018, 0.5, 57, 0.9),
            NoteEvent(0.031, 0.5, 62, 0.9),
            NoteEvent(0.045, 0.5, 66, 0.9),
        ]
        separate_voices(events)
        report = assign_positions(events, max_span=4)
        self.assertEqual(report.stretch_violations, 0)
        self.assertEqual({event.string for event in events}, {1, 2, 3, 4})
        frets = [event.fret for event in events if event.fret]
        self.assertLessEqual(max(frets) - min(frets), 4)

    def test_passage_optimiser_avoids_unnecessary_position_jump(self):
        events = [
            NoteEvent(0.0, 0.2, 64, 0.95),
            NoteEvent(0.3, 0.5, 66, 0.95),
            NoteEvent(0.6, 0.8, 67, 0.95),
        ]
        separate_voices(events)
        report = assign_positions(events)
        self.assertLess(report.average_movement, 2.0)
        self.assertEqual([(event.string, event.fret) for event in events], [(1, 0), (1, 2), (1, 3)])

    def test_ringing_note_avoids_impossible_string_reuse(self):
        events = [
            NoteEvent(0.0, 1.0, 52, 0.95),
            NoteEvent(0.3, 0.7, 54, 0.95),
        ]
        separate_voices(events)
        assign_positions(events)
        self.assertNotEqual(events[0].string, events[1].string)

    def test_viterbi_sacrifices_early_local_cost_for_better_future(self):
        # Greedy starts both E3 notes at D-string fret 2 and then jumps to fret
        # 6. The global path deliberately starts at A-string fret 7 because it
        # sees that the final A#4 is coming.
        events = [
            NoteEvent(0.0, 0.2, 52, 0.95),
            NoteEvent(0.3, 0.5, 52, 0.95),
            NoteEvent(0.6, 0.8, 70, 0.95),
        ]
        separate_voices(events)
        report = assign_positions(events)
        self.assertEqual(
            [(event.string, event.fret) for event in events],
            [(5, 7), (5, 7), (1, 6)],
        )
        self.assertAlmostEqual(report.optimization_cost, 1.7)

    def test_dp_cost_equals_exhaustive_search_over_full_trellis(self):
        events = [
            NoteEvent(0.00, 0.2, 52, 0.9),
            NoteEvent(0.02, 0.2, 59, 0.9),
            NoteEvent(0.30, 0.5, 55, 0.9),
            NoteEvent(0.32, 0.5, 62, 0.9),
            NoteEvent(0.60, 0.8, 57, 0.9),
            NoteEvent(0.62, 0.8, 64, 0.9),
        ]
        separate_voices(events)
        clusters = cluster_events(events)
        layers = [enumerate_cluster_fingerings(cluster, 0) for cluster in clusters]

        def path_cost(path):
            return sum(shape.local_cost for shape in path) + sum(
                _transition_cost(left, right) for left, right in zip(path, path[1:])
            )

        exhaustive_minimum = min(path_cost(path) for path in product(*layers))
        report = assign_positions(events)
        self.assertEqual(report.candidate_counts, [len(layer) for layer in layers])
        self.assertAlmostEqual(report.optimization_cost, exhaustive_minimum, places=6)

    def test_complete_shape_layer_is_not_beam_pruned(self):
        events = [
            NoteEvent(index * 0.005, 0.4, pitch, 0.9)
            for index, pitch in enumerate([52, 57, 61, 64])
        ]
        separate_voices(events)
        shapes = enumerate_cluster_fingerings(events, capo=0)
        self.assertGreater(len(shapes), 48)
        self.assertTrue(all(all(string is not None for string, _ in shape.positions) for shape in shapes))

    def test_dense_impossible_cluster_is_flagged_not_crashed(self):
        events = [
            NoteEvent(index * 0.002, 0.4, 40 + index * 3, 0.6 + index * 0.02)
            for index in range(9)
        ]
        separate_voices(events)
        report = assign_positions(events)
        self.assertGreater(report.unassigned_notes, 0)
        self.assertLessEqual(sum(event.string is not None for event in events), 6)

    def test_auto_capo_ranking_finds_shifted_open_tuning(self):
        events = [
            NoteEvent(index * 0.2, index * 0.2 + 0.1, pitch + 2, 0.9)
            for index, pitch in enumerate([40, 45, 50, 55, 59, 64])
        ]
        self.assertEqual(rank_capos(events)[0].capo, 2)

    def test_tab_hints_parse_double_digit_frets(self):
        with tempfile.TemporaryDirectory() as directory:
            tab = Path(directory) / "hint.tab"
            tab.write_text("e|--10--12--|\nB|--8-------|\n", encoding="utf-8")
            self.assertEqual(parse_tab_hints(str(tab)), {(1, 10), (1, 12), (2, 8)})

    def test_tab_has_tempo_and_barlines(self):
        event = NoteEvent(5.0, 5.5, 64, 0.9, string=1, fret=0)
        chord = ChordLabel("Dm", 5.0, 6.0, 0.9, 2, 2, "minor", (2, 5, 9))
        rendered = render_ascii_tab([event], capo=3, bpm=100, chords=[chord])
        self.assertIn("Capo: 3", rendered)
        self.assertIn("Estimated tempo: 100.0 BPM", rendered)
        self.assertIn("Recording offset removed: 5.00 seconds", rendered)
        self.assertTrue(rendered.splitlines()[4].startswith("Ch|Dm"))
        self.assertTrue(rendered.splitlines()[5].startswith("e|-0-"))
        self.assertGreaterEqual(rendered.count("|"), 12)

    def test_musescore_voice_ids_are_one_based(self):
        with tempfile.TemporaryDirectory() as directory:
            xml = Path(directory) / "score.musicxml"
            xml.write_text(
                "<score-partwise><voice>0</voice><voice>1</voice>"
                "<voice>10</voice></score-partwise>",
                encoding="utf-8",
            )
            normalize_musicxml_for_musescore(str(xml))
            result = xml.read_text(encoding="utf-8")
            self.assertIn("<voice>1</voice>", result)
            self.assertIn("<voice>2</voice>", result)
            self.assertIn("<voice>11</voice>", result)
            self.assertNotIn("<voice>0</voice>", result)
            normalize_musicxml_for_musescore(str(xml))
            self.assertEqual(xml.read_text(encoding="utf-8"), result)

    @unittest.skipUnless(importlib.util.find_spec("music21"), "music21 is optional")
    def test_musicxml_preserves_flat_inversion_and_extended_chord_labels(self):
        import xml.etree.ElementTree as ET

        symbols = ["Bbmaj7", "Bbsus2", "Ddim/Ab", "Am(add9)/B", "Bmaj7/Bb", "Gaug/Eb", "E5"]
        chords = [
            ChordLabel(symbol, index * 0.5, index * 0.5 + 0.5, 0.9, 0, 0, "test", (0, 4, 7))
            for index, symbol in enumerate(symbols)
        ]
        events = [NoteEvent(0.0, 3.0, 64, 0.9, string=1, fret=0)]
        with tempfile.TemporaryDirectory() as directory:
            paths = export_results(events, str(Path(directory) / "labels"), 0,
                                   TimingInfo(bpm=120.0, duration=3.0, beat_confidence=0.8), chords)
            score = ET.parse(paths["musicxml"])
        harmonies = score.findall(".//harmony")
        words = score.findall(".//words")
        self.assertEqual(len(harmonies) + len(words), len(symbols))
        self.assertEqual(harmonies[0].findtext("root/root-alter"), "-1")
        self.assertEqual(harmonies[2].findtext("bass/bass-alter"), "-1")
        self.assertIn("Am(add9)/B", [word.text for word in words])
        self.assertIn("E5", [word.text for word in words])
        self.assertNotIn("none", [item.findtext("kind") for item in harmonies])

    @unittest.skipUnless(importlib.util.find_spec("music21"), "music21 is optional")
    def test_musicxml_contains_capo_relative_harmony_and_real_notes(self):
        events = [
            NoteEvent(2.00, 2.45, 50, 0.95),
            NoteEvent(2.22, 2.60, 57, 0.94),
            NoteEvent(2.44, 2.80, 66, 0.93),
        ]
        separate_voices(events)
        assign_positions(events, capo=2)
        timing = TimingInfo(bpm=120.0, duration=3.0, beat_confidence=0.8)
        chord = ChordLabel(
            "C", 2.0, 2.8, 0.95, 0, 0, "major", (0, 4, 7),
            sounding_symbol="D", retrieval_id="major-triad", retrieval_score=0.95,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = export_results(
                events,
                str(Path(directory) / "capo_check"),
                2,
                timing,
                chords=[chord],
            )
            xml = Path(paths["musicxml"]).read_text(encoding="utf-8")
            chord_json = json.loads(Path(paths["chords"]).read_text(encoding="utf-8"))
        self.assertIn("<harmony", xml)
        self.assertIn("<root-step>C</root-step>", xml)
        self.assertGreater(xml.count("<note"), 0)
        self.assertNotIn("<voice>0</voice>", xml)
        self.assertEqual(chord_json[0]["shape_chord"], "C")
        self.assertEqual(chord_json[0]["sounding_chord"], "D")


class VoiceTests(unittest.TestCase):
    def test_voice_split_uses_same_onset_window_as_fingering(self):
        events = [
            NoteEvent(0.0, 0.5, 45, 0.9),
            NoteEvent(0.05, 0.5, 64, 0.9),
            NoteEvent(0.20, 0.5, 67, 0.9),
        ]
        separate_voices(events)
        self.assertEqual(len(cluster_events(events)), 2)
        self.assertEqual(events[0].voice, "bass")
        self.assertEqual(events[1].voice, "melody")


if __name__ == "__main__":
    unittest.main()
