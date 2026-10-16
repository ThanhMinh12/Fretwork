import importlib.util
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import macro_average_dicts, score_events
from fretwork import NoteEvent


class EvaluationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mir_eval"), "mir_eval is optional")
    def test_identical_notes_score_one(self):
        notes = [NoteEvent(0.1, 0.5, 64, 1.0), NoteEvent(0.6, 1.0, 67, 1.0)]
        score = score_events(notes, notes, duration=2.0)
        self.assertEqual(score.onset_f1, 1.0)
        self.assertEqual(score.offset_f1, 1.0)

    @unittest.skipUnless(importlib.util.find_spec("mir_eval"), "mir_eval is optional")
    def test_octave_error_is_not_a_pitch_match(self):
        reference = [NoteEvent(0.1, 0.5, 64, 1.0)]
        estimate = [NoteEvent(0.1, 0.5, 76, 1.0)]
        self.assertEqual(score_events(reference, estimate, duration=1.0).onset_f1, 0.0)

    def test_macro_average_preserves_counts(self):
        row = {
            "onset_precision": 0.5,
            "onset_recall": 0.4,
            "onset_f1": 0.44,
            "offset_precision": 0.3,
            "offset_recall": 0.2,
            "offset_f1": 0.24,
            "average_overlap": 0.7,
            "reference_notes": 10,
            "estimated_notes": 12,
        }
        result = macro_average_dicts([row, row])
        self.assertEqual(result["clips"], 2)
        self.assertEqual(result["reference_notes"], 20)
        self.assertEqual(result["onset_f1"], 0.44)


if __name__ == "__main__":
    unittest.main()
