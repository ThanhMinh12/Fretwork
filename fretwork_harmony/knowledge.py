"""Versioned local chord corpus loading and lexical retrieval."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from .models import ChordTemplate


KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "chords.json"


def _coerce_template(record: Mapping[str, object]) -> ChordTemplate:
    quality = str(record["quality"])
    identifier = record.get("id") or re.sub(
        r"[^a-z0-9]+", "-", quality.casefold()
    ).strip("-")
    return ChordTemplate(
        id=str(identifier),
        quality=quality,
        suffix=str(record["suffix"]),
        intervals=tuple(int(value) % 12 for value in record["intervals"]),  # type: ignore[arg-type]
        required=tuple(int(value) % 12 for value in record["required"]),  # type: ignore[arg-type]
        aliases=tuple(str(value) for value in record.get("aliases", ())),  # type: ignore[arg-type]
        description=str(record.get("description", "")),
        priority=int(record.get("priority", 100)),
    )


@lru_cache(maxsize=4)
def load_knowledge(
    path_string: str,
) -> tuple[tuple[str, ...], tuple[ChordTemplate, ...]]:
    with Path(path_string).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    root_names = tuple(str(name) for name in document["root_names"])
    if len(root_names) != 12:
        raise ValueError("chord knowledge must define exactly 12 root names")
    templates = tuple(_coerce_template(record) for record in document["templates"])
    if not templates:
        raise ValueError("chord knowledge contains no templates")
    return root_names, templates


def load_chord_templates(path: str | Path | None = None) -> tuple[ChordTemplate, ...]:
    return load_knowledge(str(Path(path or KNOWLEDGE_PATH).resolve()))[1]


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#ø]+", value.casefold()))


def _query_without_root(query: str) -> str:
    compact = query.strip().replace(" ", "")
    match = re.fullmatch(r"[A-Ga-g](?:#|b)?(.+)", compact)
    return match.group(1) if match else query


def search_chord_knowledge(query: str, limit: int = 5) -> list[dict[str, object]]:
    """Return deterministic ranked records from the bundled local corpus."""

    if limit <= 0:
        return []
    searchable_query = _query_without_root(query)
    phrase = _normalise(searchable_query)
    tokens = set(phrase.split())
    ranked: list[tuple[float, ChordTemplate]] = []
    for template in load_chord_templates():
        exact_fields = {
            _normalise(template.quality),
            _normalise(template.suffix),
            *(_normalise(alias) for alias in template.aliases),
        }
        words = set(
            _normalise(
                " ".join(
                    (template.quality, template.suffix, *template.aliases, template.description)
                )
            ).split()
        )
        if not phrase:
            score = 1.0
        else:
            score = len(tokens & words) / max(1, len(tokens)) * 10.0
            if searchable_query == template.suffix:
                score += 40.0
            if phrase in exact_fields:
                score += 20.0
            elif any(phrase in field for field in exact_fields):
                score += 4.0
        if score > 0:
            ranked.append((score, template))
    ranked.sort(key=lambda item: (-item[0], item[1].priority, item[1].quality))
    results = []
    for score, template in ranked[:limit]:
        record = template.to_dict()
        record["retrieval_score"] = round(score, 4)
        results.append(record)
    return results

