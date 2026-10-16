"""ASCII TAB, MusicXML construction, and artifact export."""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from .models import STRING_NAMES, NoteEvent, TimingInfo


def render_ascii_tab(
    events: Sequence[NoteEvent],
    capo: int,
    bpm: float = 120.0,
    chords: Sequence[object] = (),
    subdivisions_per_beat: int = 4,
    bars_per_block: int = 4,
) -> str:
    """Render tempo-aware ASCII TAB with chord and bar lines."""

    if not events:
        return "(no notes detected)"
    bpm = bpm if bpm > 0 else 120.0
    time_origin = min(event.start for event in events)
    seconds_per_slot = 60.0 / bpm / subdivisions_per_beat
    slots_per_bar = subdivisions_per_beat * 4
    last_slot = max(
        int(round((event.end - time_origin) / seconds_per_slot)) for event in events
    ) + 1
    total_slots = max(slots_per_bar, last_slot)
    cells = {string: ["---"] * total_slots for string in range(1, 7)}
    chord_characters = [" "] * (total_slots * 3)
    for event in events:
        if event.string is None or event.fret is None:
            continue
        slot = min(
            total_slots - 1,
            max(0, int(round((event.start - time_origin) / seconds_per_slot))),
        )
        cells[event.string][slot] = str(event.fret).center(3, "-")
    for chord in chords:
        chord_start = float(getattr(chord, "start", time_origin))
        slot = min(
            total_slots - 1,
            max(0, int(round((chord_start - time_origin) / seconds_per_slot))),
        )
        label = str(getattr(chord, "symbol", "?"))
        for index, character in enumerate(label):
            target = slot * 3 + index
            if target >= len(chord_characters):
                break
            chord_characters[target] = character

    output = [
        f"Capo: {capo if capo else 'none'}",
        f"Estimated tempo: {bpm:.1f} BPM",
        f"Recording offset removed: {time_origin:.2f} seconds",
        "",
    ]
    slots_per_block = slots_per_bar * bars_per_block
    for block_start in range(0, total_slots, slots_per_block):
        block_end = min(total_slots, block_start + slots_per_block)
        if chords:
            chord_chunks = []
            for bar_start in range(block_start, block_end, slots_per_bar):
                bar_end = min(block_end, bar_start + slots_per_bar)
                chord_chunks.append(
                    "".join(chord_characters[bar_start * 3 : bar_end * 3])
                )
            output.append("Ch|" + "|".join(chord_chunks) + "|")
        for string in range(1, 7):
            chunks = []
            for bar_start in range(block_start, block_end, slots_per_bar):
                bar_end = min(block_end, bar_start + slots_per_bar)
                chunks.append("".join(cells[string][bar_start:bar_end]))
            output.append(f"{STRING_NAMES[string]}|" + "|".join(chunks) + "|")
        output.append("")
    return "\n".join(output)


def _insert_chord_symbols(part, chords: Sequence[object], origin: float, scale: float) -> None:
    from music21 import expressions, harmony

    for label in chords:
        offset = max(0.0, _quantize((float(getattr(label, "start")) - origin) * scale))
        symbol = str(getattr(label, "symbol"))
        # music21 spells flats with '-', unlike the public chord vocabulary.
        normalized = re.sub(r"^([A-G])b", r"\1-", symbol)
        normalized = re.sub(r"/([A-G])b", r"/\1-", normalized)
        try:
            annotation = harmony.ChordSymbol(normalized)
        except (ValueError, harmony.HarmonyException):
            # Preserve unsupported extensions verbatim rather than dropping
            # a model prediction or guessing different harmonic semantics.
            annotation = expressions.TextExpression(symbol)
            annotation.placement = "above"
        part.insert(offset, annotation)


def _quantize(value: float, grid: float = 0.25) -> float:
    return round(value / grid) * grid


def _group_written_onsets(
    events: Sequence[NoteEvent], origin: float, scale: float
) -> dict[tuple[str, float], list[NoteEvent]]:
    grouped: dict[tuple[str, float], list[NoteEvent]] = {}
    for event in events:
        voice = "bass" if event.voice == "bass" else "melody"
        offset = _quantize((event.start - origin) * scale)
        grouped.setdefault((voice, offset), []).append(event)
    return grouped


def _populate_voice(voice, voice_name: str, grouped, scale: float) -> None:
    from music21 import articulations, chord as m21chord, note as m21note

    onsets = sorted(
        (offset, events)
        for (group_voice, offset), events in grouped.items()
        if group_voice == voice_name
    )
    for index, (offset, events) in enumerate(onsets):
        durations = [
            max(0.25, _quantize((event.end - event.start) * scale))
            for event in events
        ]
        duration = max(0.25, min(durations))
        if index + 1 < len(onsets):
            duration = max(0.25, min(duration, onsets[index + 1][0] - offset))
        if len(events) == 1:
            musical_object = m21note.Note(events[0].pitch)
            target_notes = [musical_object]
        else:
            musical_object = m21chord.Chord([event.pitch for event in events])
            target_notes = list(musical_object.notes)
        musical_object.quarterLength = duration
        musical_object.stemDirection = "down" if voice_name == "bass" else "up"
        for target_note, event in zip(target_notes, events):
            if event.string is not None:
                target_note.articulations.append(
                    articulations.StringIndication(event.string)
                )
            if event.fret is not None:
                target_note.articulations.append(articulations.FretIndication(event.fret))
        voice.insert(offset, musical_object)


def build_score(
    events: Sequence[NoteEvent],
    title: str = "Fretwork Transcription",
    bpm: float = 120.0,
    capo: int = 0,
    chords: Sequence[object] = (),
):
    """Build a two-voice MusicXML-ready score with chord and fret marks."""

    try:
        from music21 import clef, instrument, metadata, meter, stream, tempo
    except ImportError as exc:  # pragma: no cover - optional notation stack
        raise RuntimeError("music21 is not installed; see README.md") from exc

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    part = stream.Part(id="FingerstyleGuitar")
    part.partName = f"Guitar (capo {capo})" if capo else "Guitar"
    part.insert(0, instrument.Guitar())
    part.insert(0, clef.TrebleClef())
    part.insert(0, tempo.MetronomeMark(number=float(bpm)))
    part.insert(0, meter.TimeSignature("4/4"))

    melody_voice = stream.Voice(id="melody")
    bass_voice = stream.Voice(id="bass")
    scale = float(bpm) / 60.0
    origin = min((event.start for event in events), default=0.0)
    grouped = _group_written_onsets(events, origin, scale)
    _insert_chord_symbols(part, chords, origin, scale)
    _populate_voice(melody_voice, "melody", grouped, scale)
    _populate_voice(bass_voice, "bass", grouped, scale)

    part.insert(0, melody_voice)
    part.insert(0, bass_voice)
    part.makeMeasures(inPlace=True)
    # Avoid makeNotation(): music21's auto-beamer can emit malformed partial
    # beam pairs for irregular ML timing.
    part.makeTies(inPlace=True)
    part.makeRests(fillGaps=True, timeRangeFromBarDuration=True, inPlace=True)
    part.makeAccidentals(inPlace=True, cautionaryNotImmediateRepeat=False)
    score.insert(0, part)
    return score


def normalize_musicxml_for_musescore(xml_path: str) -> None:
    """Shift music21's zero-based voice IDs to MuseScore-safe IDs."""

    path = Path(xml_path)
    document = path.read_text(encoding="utf-8")
    if "<voice>0</voice>" not in document:
        return
    document = re.sub(
        r"<voice>(\d+)</voice>",
        lambda match: f"<voice>{int(match.group(1)) + 1}</voice>",
        document,
    )
    path.write_text(document, encoding="utf-8")


def write_musicxml(score, xml_path: str) -> str:
    generated_path = str(score.write("musicxml", fp=xml_path))
    normalize_musicxml_for_musescore(generated_path)
    return generated_path


def export_results(
    events: list[NoteEvent],
    output_prefix: str,
    capo: int,
    timing: TimingInfo,
    chords: Sequence[object] = (),
    title: str = "Fretwork Transcription",
    render_pdf: bool = False,
) -> dict[str, Optional[str]]:
    """Write MusicXML, TAB, event/chord JSON, and an optional PDF."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Optional[str]] = {
        "musicxml": str(prefix.with_suffix(".musicxml")),
        "tab": str(prefix.parent / f"{prefix.name}_tab.txt"),
        "events": str(prefix.parent / f"{prefix.name}_events.json"),
        "chords": str(prefix.parent / f"{prefix.name}_chords.json"),
        "pdf": None,
    }
    Path(paths["tab"]).write_text(
        render_ascii_tab(events, capo, timing.bpm, chords), encoding="utf-8"
    )
    Path(paths["events"]).write_text(
        json.dumps([asdict(event) for event in events], indent=2), encoding="utf-8"
    )
    chord_records = [
        chord.to_dict() if hasattr(chord, "to_dict") else dict(chord)  # type: ignore[arg-type]
        for chord in chords
    ]
    Path(paths["chords"]).write_text(
        json.dumps(chord_records, indent=2), encoding="utf-8"
    )
    score = build_score(events, title, timing.bpm, capo, chords)
    write_musicxml(score, paths["musicxml"])
    if render_pdf and shutil.which("lilypond"):
        try:
            pdf_prefix = str(prefix.parent / f"{prefix.name}_score")
            paths["pdf"] = str(score.write("lily.pdf", fp=pdf_prefix))
        except Exception as exc:
            print(f"PDF rendering failed: {exc}", file=sys.stderr)
    return paths

