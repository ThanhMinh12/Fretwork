"""Use only the pinned, integrity-checked upstream guitar checkpoint."""
import hashlib
import os
from pathlib import Path

GAPS_REVISION = '689e773723bcafd8c81015b10c03f12675ce16ec'
GAPS_SHA256 = '65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513'


def verify_checkpoint(path: Path, expected: str = GAPS_SHA256) -> Path:
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != expected:
        raise RuntimeError('Guitar checkpoint failed its SHA-256 integrity check; use the pinned upstream weights')
    return path


def verified_gaps_checkpoint() -> Path:
    configured = os.getenv('FRETWORK_GAPS_CHECKPOINT','').strip()
    if configured:
        return verify_checkpoint(Path(configured))
    from huggingface_hub import hf_hub_download
    return verify_checkpoint(Path(hf_hub_download(
        'xavriley/midi-transcription-models', 'guitar-gaps.pth', revision=GAPS_REVISION,
    )))
