#!/usr/bin/env python3
"""Local Model Context Protocol tools for the Fretwork agent.

The server deliberately accepts filesystem paths instead of audio bytes or
URLs.  Audio never leaves the machine: an MCP host asks Fretwork to process a
file that is already inside one of the configured local input roots.

This module targets the official MCP Python SDK v2 API (``mcp>=2,<3``).  The
non-MCP helper functions remain importable when that optional dependency is
absent, which keeps path validation and adapter code independently testable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_ROOTS = (PROJECT_ROOT, Path("/tmp"))
SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}
SUPPORTED_TAB_SUFFIXES = {".tab", ".txt"}
MAX_AUDIO_BYTES = 1_000_000_000
MAX_TAB_BYTES = 10_000_000
MAX_REPORT_BYTES = 25_000_000


try:  # MCP is an optional integration dependency, not a core ML dependency.
    from mcp.server import MCPServer
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - environment specific
    MCPServer = None  # type: ignore[assignment,misc]
    _MCP_IMPORT_ERROR: Exception | None = exc
else:
    _MCP_IMPORT_ERROR = None


def _configured_roots(variable: str, defaults: Sequence[Path]) -> tuple[Path, ...]:
    """Resolve an OS-path-separated allowlist from the environment."""
    configured = os.getenv(variable, "").strip()
    raw_roots = [Path(item).expanduser() for item in configured.split(os.pathsep) if item]
    roots = raw_roots or list(defaults)
    return tuple(root.resolve() for root in roots)


def _input_roots() -> tuple[Path, ...]:
    return _configured_roots("FRETWORK_MCP_INPUT_ROOTS", DEFAULT_INPUT_ROOTS)


def _output_root() -> Path:
    configured = os.getenv("FRETWORK_MCP_OUTPUT_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_OUTPUT_ROOT.resolve()


def _within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _reject_nonlocal_path(raw_path: str, label: str) -> str:
    value = str(raw_path).strip()
    if not value or "\x00" in value:
        raise ValueError(f"{label} path is empty or invalid")
    # urlsplit also catches file:, data:, ftp:, and cloud-provider schemes.
    # file:// is intentionally rejected: callers must provide an ordinary
    # local path so there is no ambiguity about host names or URL decoding.
    if urlsplit(value).scheme or "://" in value:
        raise ValueError(f"{label} must be an ordinary local filesystem path, not a URL")
    return value


def _resolve_input_file(
    raw_path: str,
    *,
    label: str,
    suffixes: set[str],
    max_bytes: int,
    roots: Sequence[Path] | None = None,
) -> Path:
    """Validate a regular local file and resolve symlinks before allowlisting."""
    value = _reject_nonlocal_path(raw_path, label)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc

    allowed_roots = tuple(roots) if roots is not None else _input_roots()
    if not _within(resolved, allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise PermissionError(f"{label} is outside the allowed local roots: {allowed}")
    file_stat = resolved.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} must be a regular file: {resolved}")
    if resolved.suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise ValueError(f"Unsupported {label} extension {resolved.suffix!r}; expected {expected}")
    if file_stat.st_size == 0:
        raise ValueError(f"{label} is empty: {resolved}")
    if file_stat.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes:,}-byte local processing limit")
    return resolved


def _resolve_output_directory(subdirectory: str = "") -> Path:
    """Return an output directory confined beneath the configured output root."""
    output_root = _output_root()
    value = str(subdirectory).strip()
    if not value or value == ".":
        return output_root
    _reject_nonlocal_path(value, "Output directory")
    requested = Path(value)
    if requested.is_absolute():
        resolved = requested.resolve()
    else:
        resolved = (output_root / requested).resolve()
    if not _within(resolved, (output_root,)):
        raise PermissionError(f"Output directory must remain beneath {output_root}")
    return resolved


def _safe_stem(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not stem:
        raise ValueError("Output stem must contain at least one letter or number")
    return stem[:120]


def _parse_capo(value: str) -> int | Literal["auto"]:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        capo = int(text)
    except ValueError as exc:
        raise ValueError("capo must be 'auto' or an integer from 0 through 12") from exc
    if not 0 <= capo <= 12:
        raise ValueError("capo must be 'auto' or an integer from 0 through 12")
    return capo


def _json_compatible(value: Any) -> Any:
    """Convert project results into values accepted as MCP structured output."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    return str(value)


def transcribe_local_audio(
    audio_path: str,
    title: str = "Fretwork Transcription",
    backend: Literal["ensemble", "gaps", "basic_pitch"] = "ensemble",
    capo: str = "auto",
    tab_hint_path: str | None = None,
    output_subdirectory: str = "",
    output_stem: str | None = None,
    max_span: int = 4,
    min_confidence: float = 0.30,
    max_attempts: int = 3,
    render_pdf: bool = False,
) -> dict[str, Any]:
    """Transcribe a local solo-guitar recording through the LangGraph agent.

    The input is a local path, never a URL or uploaded payload. ``ensemble``
    asks the guitar CRNN specialist and Basic Pitch reviewer to reconcile their
    proposals; either model can also be selected alone. The result contains
    generated file paths, diagnostics, selected capo, and the inspectable
    multi-agent workflow log.
    """
    if backend not in {"ensemble", "gaps", "basic_pitch"}:
        raise ValueError("backend must be 'ensemble', 'gaps', or 'basic_pitch'")
    if not 1 <= int(max_span) <= 12:
        raise ValueError("max_span must be from 1 through 12 frets")
    if not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if not 1 <= int(max_attempts) <= 3:
        raise ValueError("max_attempts must be from 1 through 3")
    clean_title = str(title).strip()
    if not clean_title or len(clean_title) > 200 or any(ord(char) < 32 for char in clean_title):
        raise ValueError("title must be 1-200 printable characters")

    audio = _resolve_input_file(
        audio_path,
        label="Audio file",
        suffixes=SUPPORTED_AUDIO_SUFFIXES,
        max_bytes=MAX_AUDIO_BYTES,
    )
    tab_hint: Path | None = None
    if tab_hint_path:
        tab_hint = _resolve_input_file(
            tab_hint_path,
            label="TAB hint file",
            suffixes=SUPPORTED_TAB_SUFFIXES,
            max_bytes=MAX_TAB_BYTES,
        )
    output_dir = _resolve_output_directory(output_subdirectory)
    stem = _safe_stem(output_stem, audio.stem)

    # Import lazily so listing MCP tools stays cheap and does not initialize
    # Torch/ONNX, librosa, music21, or LangGraph.
    from fretwork_agent import run_agent

    result = run_agent(
        str(audio),
        output_dir=str(output_dir),
        output_stem=stem,
        title=clean_title,
        transcription_backend=backend,
        capo=_parse_capo(capo),
        tab_hint_path=str(tab_hint) if tab_hint else None,
        max_span=int(max_span),
        min_confidence=float(min_confidence),
        max_attempts=int(max_attempts),
        render_pdf=bool(render_pdf),
    )
    return _json_compatible(
        {
            "selected_capo": result.get("selected_capo"),
            "heuristic_health": result.get("quality", {}),
            "outputs": result.get("outputs", {}),
            "report_path": result.get("report_path"),
            "workflow_log": result.get("messages", []),
            "backend": backend,
            "local_only": True,
        }
    )


_CHORD_MODULE: ModuleType | None = None


def _load_chord_retrieval() -> ModuleType:
    """Load the in-repository chord knowledge adapter, if it is installed."""
    global _CHORD_MODULE
    if _CHORD_MODULE is not None:
        return _CHORD_MODULE
    module_path = PROJECT_ROOT / "chord_retrieval.py"
    if not module_path.is_file():
        raise RuntimeError(
            "Chord retrieval is not available in this checkout; chord_retrieval.py is missing"
        )
    spec = importlib.util.spec_from_file_location("_fretwork_chord_retrieval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the local chord retrieval module")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses and recursive imports can find
    # their defining module in the normal Python way.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CHORD_MODULE = module
    return module


def search_local_chord_knowledge(query: str, limit: int = 5) -> dict[str, Any]:
    """Retrieve chord spellings and guitar-shape knowledge from the local index.

    This is the RAG retrieval boundary: it searches the bundled deterministic
    chord corpus and does not call a web search, hosted vector database, or LLM.
    """
    clean_query = str(query).strip()
    if not clean_query or len(clean_query) > 200:
        raise ValueError("query must contain 1-200 characters")
    if not 1 <= int(limit) <= 20:
        raise ValueError("limit must be from 1 through 20")
    module = _load_chord_retrieval()
    search = getattr(module, "search_chord_knowledge", None)
    if not callable(search):
        raise RuntimeError("chord_retrieval.py does not expose search_chord_knowledge")
    matches = search(clean_query, limit=int(limit))
    return {
        "query": clean_query,
        "matches": _json_compatible(matches),
        "retrieval": "local_chord_knowledge",
        "local_only": True,
    }


def inspect_transcription_report(
    report_path: str,
    include_workflow_log: bool = True,
) -> dict[str, Any]:
    """Read and validate a Fretwork report JSON from the local output root."""
    report = _resolve_input_file(
        report_path,
        label="Report file",
        suffixes={".json"},
        max_bytes=MAX_REPORT_BYTES,
        roots=(_output_root(),),
    )
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Report is not valid UTF-8 JSON: {report}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Report JSON must contain an object at its top level")
    if not include_workflow_log:
        payload = dict(payload)
        payload.pop("workflow_log", None)
    return {
        "report_path": str(report),
        "report": _json_compatible(payload),
        "local_only": True,
    }


if MCPServer is not None:
    mcp = MCPServer(
        "Fretwork",
        instructions=(
            "Local-only fingerstyle guitar transcription. Use transcribe_local_audio "
            "for a path already on this machine, search_local_chord_knowledge for "
            "grounded chord information, and inspect_transcription_report to review "
            "heuristic health and agent decisions. Never invent paths or pass remote URLs."
        ),
    )
    mcp.tool()(transcribe_local_audio)
    mcp.tool()(search_local_chord_knowledge)
    mcp.tool()(inspect_transcription_report)
else:  # A clear sentinel is friendlier to import-time probes and unit tests.
    mcp = None


def main() -> None:
    """Run the local MCP server over stdio."""
    if mcp is None:
        raise SystemExit(
            "The Fretwork MCP server requires the official Python SDK v2. "
            "Install it with: python -m pip install 'mcp>=2,<3'\n"
            f"Import error: {_MCP_IMPORT_ERROR}"
        )
    # Stdio keeps the default deployment local and lets the MCP host own the
    # process. There is intentionally no public HTTP listener in this module.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
