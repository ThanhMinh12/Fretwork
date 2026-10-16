import unittest

from chord_retrieval import (
    ChordInferenceConfig,
    _Candidate,
    _decode_candidate_sequence,
    infer_chords,
    load_chord_templates,
    search_chord_knowledge,
)
from fretwork import NoteEvent, TimingInfo


def note(start, pitch, confidence=0.9):
    return NoteEvent(start, start + 0.45, pitch, confidence)


class ChordKnowledgeTests(unittest.TestCase):
    def test_local_knowledge_contains_extended_shapes(self):
        suffixes = {template.suffix for template in load_chord_templates()}
        self.assertTrue({"", "m", "7", "sus4", "add9"}.issubset(suffixes))

    def test_mcp_search_is_ranked_and_json_serialisable(self):
        results = search_chord_knowledge("Dm7", limit=2)
        self.assertEqual(results[0]["suffix"], "m7")
        self.assertEqual(results[0]["id"], "minor-seventh")
        self.assertIn("retrieval_score", results[0])
        self.assertIsInstance(results[0]["intervals"], list)
        self.assertEqual(search_chord_knowledge("minor seventh", limit=0), [])


class ChordInferenceTests(unittest.TestCase):
    timing = TimingInfo(bpm=120.0, duration=4.0, beat_confidence=0.9)

    def test_arpeggio_across_half_measure_is_one_chord(self):
        # C, E and G never start simultaneously, but all fall in a two-beat
        # (one-second at 120 BPM) analysis window.
        events = [note(0.00, 48), note(0.32, 55), note(0.68, 64)]
        labels = infer_chords(events, self.timing)
        self.assertEqual([label.symbol for label in labels], ["C"])
        self.assertEqual(labels[0].evidence_pcs, (0, 4, 7))

    def test_symbols_name_capo_relative_hand_shapes(self):
        # Sounding D-F#-A becomes the C-E-G hand shape under capo 2.
        events = [note(0.00, 50), note(0.30, 57), note(0.60, 66)]
        labels = infer_chords(events, self.timing, capo=2)
        self.assertEqual(labels[0].symbol, "C")
        self.assertEqual(labels[0].root_pc, 0)
        self.assertEqual(labels[0].sounding_symbol, "D")
        self.assertEqual(labels[0].to_dict()["shape_chord"], "C")
        self.assertEqual(labels[0].to_dict()["sounding_chord"], "D")
        self.assertEqual(labels[0].retrieval_id, "major-triad")

    def test_minor_and_extended_chords_have_readable_suffixes(self):
        d_minor = [note(0.00, 50), note(0.25, 57), note(0.55, 65)]
        c_add9 = [note(1.00, 48), note(1.20, 55), note(1.45, 62), note(1.70, 64)]
        labels = infer_chords(d_minor + c_add9, self.timing)
        self.assertEqual([label.symbol for label in labels], ["Dm", "Cadd9"])

    def test_repeated_adjacent_labels_are_merged(self):
        events = [
            note(0.00, 48), note(0.25, 55), note(0.55, 64),
            note(1.00, 48), note(1.25, 55), note(1.55, 64),
        ]
        labels = infer_chords(events, self.timing)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].symbol, "C")
        self.assertEqual(labels[0].windows, 2)
        self.assertGreater(labels[0].end, 1.5)

    def test_weak_or_single_note_windows_are_suppressed(self):
        weak = [note(0.00, 48, 0.15), note(0.25, 55, 0.18), note(0.50, 64, 0.20)]
        single = [note(1.00, 60, 0.95)]
        self.assertEqual(infer_chords(weak + single, self.timing), [])

    def test_one_beat_windows_can_follow_faster_changes(self):
        config = ChordInferenceConfig(window_beats=1.0, include_inversions=False)
        events = [
            note(0.00, 48), note(0.15, 52), note(0.30, 55),
            note(0.50, 50), note(0.65, 57), note(0.80, 65),
        ]
        labels = infer_chords(events, self.timing, config=config)
        self.assertEqual([label.symbol for label in labels], ["C", "Dm"])

    def test_inversion_is_named_when_bass_is_not_root(self):
        events = [note(0.00, 52), note(0.25, 55), note(0.50, 60)]
        labels = infer_chords(events, self.timing)
        self.assertEqual(labels[0].symbol, "C/E")

    def test_viterbi_sequence_prior_resolves_a_one_window_flicker(self):
        def candidate(symbol, score, root):
            return _Candidate(
                symbol=symbol,
                sounding_symbol=symbol,
                score=score,
                root_pc=root,
                bass_pc=root,
                quality="major",
                evidence_pcs=(root,),
                priority=10,
                retrieval_id="major-triad",
            )

        layers = [
            [candidate("C", 0.95, 0), candidate("G", 0.80, 7)],
            [candidate("G", 0.91, 7), candidate("C", 0.90, 0)],
            [candidate("C", 0.95, 0), candidate("G", 0.80, 7)],
        ]
        local = _decode_candidate_sequence(layers, transition_weight=0.0)
        contextual = _decode_candidate_sequence(layers, transition_weight=0.22)
        self.assertEqual([item.symbol for item in local], ["C", "G", "C"])
        self.assertEqual([item.symbol for item in contextual], ["C", "C", "C"])


if __name__ == "__main__":
    unittest.main()
