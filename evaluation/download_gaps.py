"""Download and crop the pinned held-out GAPS evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import wave
from pathlib import Path


DEFAULT_MANIFEST = Path(__file__).with_name("gaps_test_15.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Fretwork-eval/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def _crop_wav(source: Path, destination: Path, start: float, duration: float) -> None:
    temporary = destination.with_suffix(".part.wav")
    with wave.open(str(source), "rb") as input_wav:
        rate = input_wav.getframerate()
        start_frame = round(start * rate)
        frame_count = round(duration * rate)
        if start_frame + frame_count > input_wav.getnframes():
            raise ValueError(f"Requested excerpt exceeds {source.name}")
        input_wav.setpos(start_frame)
        frames = input_wav.readframes(frame_count)
        with wave.open(str(temporary), "wb") as output_wav:
            output_wav.setparams(input_wav.getparams())
            output_wav.setnframes(0)
            output_wav.writeframes(frames)
    temporary.replace(destination)


def prepare(manifest_path: Path, data_dir: Path, keep_sources: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest["dataset_revision"]
    base = f"https://huggingface.co/datasets/xavriley/GAPS/resolve/{revision}"
    audio_dir, midi_dir = data_dir / "audio", data_dir / "midi"
    source_dir = data_dir / "sources"
    for directory in (audio_dir, midi_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    start = float(manifest["excerpt_start_seconds"])
    duration = float(manifest["excerpt_duration_seconds"])
    for index, item in enumerate(manifest["items"], start=1):
        identity = item["id"]
        midi = midi_dir / f"{identity}.mid"
        if not midi.exists():
            _download(f"{base}/{item['midi']}?download=true", midi)
        if midi.stat().st_size != item["midi_bytes"] or _git_blob_oid(midi) != item["midi_git_oid"]:
            raise ValueError(f"MIDI integrity check failed for {identity}")

        excerpt = audio_dir / f"{identity}.wav"
        if not excerpt.exists():
            source = source_dir / f"{identity}.wav"
            if not source.exists():
                print(f"[{index:02d}/{len(manifest['items'])}] downloading {identity}")
                _download(f"{base}/{item['audio']}?download=true", source)
            if source.stat().st_size != item["audio_bytes"] or _sha256(source) != item["audio_sha256"]:
                raise ValueError(f"Audio integrity check failed for {identity}")
            _crop_wav(source, excerpt, start, duration)
            if not keep_sources:
                source.unlink()
        print(f"[{index:02d}/{len(manifest['items'])}] ready {identity}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=Path("evaluation/data"))
    parser.add_argument("--keep-sources", action="store_true")
    args = parser.parse_args()
    prepare(args.manifest, args.data_dir, args.keep_sources)


if __name__ == "__main__":
    main()
