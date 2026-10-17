import numpy as np
import pytest
import soundfile as sf

from hf_space.demo import DemoWorkspaces, validate_audio, render_score


def test_rejects_path_traversal(tmp_path):
    store = DemoWorkspaces(tmp_path)
    with pytest.raises(ValueError):
        store.delete('../other-files')


def test_workspaces_isolated_and_bounded(tmp_path):
    store = DemoWorkspaces(tmp_path, max_sessions=2)
    a, b = store.new(), store.new()
    first, second = store.prepare(a), store.prepare(b)
    with pytest.raises(ValueError, match='busy'):
        store.prepare(store.new())
    store.delete(a)
    assert not first.exists()
    assert second.exists()


@pytest.mark.parametrize('seconds,valid', [(1, False), (2, True), (30, True), (31, False)])
def test_duration_limits_before_decoding(tmp_path, seconds, valid):
    audio = tmp_path / 'recording.wav'
    sf.write(audio, np.zeros(16000 * seconds), 16000)
    if valid:
        assert validate_audio(str(audio)) == audio
    else:
        with pytest.raises(ValueError, match='2–30'):
            validate_audio(str(audio))


def test_actual_musicxml_engraving(tmp_path):
    from music21 import stream, note, harmony
    score = stream.Score()
    part = stream.Part()
    part.append(harmony.ChordSymbol('C'))
    part.append(note.Note('C4', quarterLength=1))
    score.append(part)
    path = score.write('musicxml', fp=tmp_path / 'score.musicxml')
    rendered = render_score(path)
    assert '<svg' in rendered
    assert 'note' in rendered
