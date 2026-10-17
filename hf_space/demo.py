"""Input limits, ephemeral workspaces, and real MusicXML preview rendering."""
from __future__ import annotations

import re
import shutil
import threading
import uuid
from pathlib import Path

import soundfile as sf


class DemoWorkspaces:
    def __init__(self, root: Path, max_sessions: int = 12):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions
        self.lock = threading.RLock()

    @staticmethod
    def new():
        return uuid.uuid4().hex

    def path(self, session):
        if not isinstance(session, str) or not re.fullmatch(r'[a-f0-9]{32}', session):
            raise ValueError('This demo session expired. Reload the page.')
        return self.root / session

    def delete(self, session):
        with self.lock:
            path = self.path(session)
            if path.exists():
                shutil.rmtree(path)

    def prepare(self, session):
        with self.lock:
            path = self.path(session)
            if not path.exists() and sum(p.is_dir() for p in self.root.iterdir()) >= self.max_sessions:
                raise ValueError('The free demo is busy. Please try again later.')
            self.delete(session)
            path.mkdir(mode=0o700)
            return path


def validate_audio(filename):
    if not filename:
        raise ValueError('Choose a guitar recording first.')
    path = Path(filename)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024**2:
        raise ValueError('Choose a non-empty recording smaller than 16 MB.')
    try:
        info = sf.info(path)
    except (RuntimeError, ValueError) as exc:
        raise ValueError('The recording could not be decoded. Try WAV, FLAC, or MP3.') from exc
    if not 2 <= info.duration <= 30.05 or info.channels > 2:
        raise ValueError('Use 2–30 seconds of mono or stereo guitar audio.')
    return path


def render_score(path):
    import verovio
    toolkit = verovio.toolkit()
    toolkit.setOptions({'pageWidth': 2100, 'pageHeight': 2800, 'scale': 40, 'adjustPageHeight': True})
    if not toolkit.loadData(Path(path).read_text(encoding='utf-8')):
        raise RuntimeError('The score could not be engraved. Download the MusicXML to open it in a notation editor.')
    return '<div style="overflow:auto;background:white;color:black">' + ''.join(
        toolkit.renderToSVG(page) for page in range(1, min(toolkit.getPageCount(), 6) + 1)
    ) + '</div>'
