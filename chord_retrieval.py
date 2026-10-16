"""Compatibility facade for local chord retrieval and inference."""

from fretwork_harmony.inference import (
    decode_candidate_sequence as _decode_candidate_sequence,
    infer_chords,
)
from fretwork_harmony.knowledge import (
    KNOWLEDGE_PATH,
    load_chord_templates,
    search_chord_knowledge,
)
from fretwork_harmony.models import (
    Candidate as _Candidate,
    ChordInferenceConfig,
    ChordLabel,
    ChordTemplate,
)

__all__ = [
    "ChordInferenceConfig",
    "ChordLabel",
    "ChordTemplate",
    "infer_chords",
    "load_chord_templates",
    "search_chord_knowledge",
]

