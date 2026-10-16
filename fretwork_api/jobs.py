from __future__ import annotations

import json
import shutil
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .config import Settings


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self.settings.prepare()
        self._fail_interrupted()

    def run_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError(job_id)
        return self.settings.runs_dir / job_id

    def _job_path(self, job_id: str) -> Path:
        return self.run_dir(job_id) / "job.json"

    def create(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        record = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "phase": "Waiting for the transcription worker",
            "error": None,
            "result": None,
            "created_at": now,
            "updated_at": now,
            **payload,
        }
        self.run_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._write(record)
        return record

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        if not path.is_file():
            raise KeyError(job_id)
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        records = []
        for path in self.settings.runs_dir.glob("*/job.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            record = self.get(job_id)
            record.update(changes)
            record["updated_at"] = _now()
            self._write(record)
            return record

    def delete(self, job_id: str) -> None:
        if not self._job_path(job_id).is_file():
            raise KeyError(job_id)
        shutil.rmtree(self.run_dir(job_id))

    def _write(self, record: dict[str, Any]) -> None:
        destination = self._job_path(record["id"])
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(destination)

    def _fail_interrupted(self) -> None:
        for record in self.list():
            if record["status"] in {"queued", "processing"}:
                self.update(
                    record["id"],
                    status="failed",
                    progress=0.0,
                    phase="Transcription was interrupted",
                    error="The server restarted while this job was running. Submit it again.",
                )


class JobRunner:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        transcribe: Callable[..., dict[str, Any]],
    ):
        self.settings = settings
        self.store = store
        self.transcribe = transcribe
        self.slots = threading.BoundedSemaphore(settings.max_pending_jobs)
        self.executor = ThreadPoolExecutor(
            max_workers=settings.workers, thread_name_prefix="fretwork"
        )

    def submit(self, job_id: str) -> None:
        self.executor.submit(self._run, job_id)

    @staticmethod
    def _plain(value: Any) -> dict[str, Any]:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, dict):
            return value
        return {}

    def _run(self, job_id: str) -> None:
        record = self.store.get(job_id)
        run_dir = self.store.run_dir(job_id)
        try:
            self.store.update(
                job_id,
                status="processing",
                progress=0.08,
                phase="Processing audio, model predictions, and fingering",
            )
            options = record["options"]
            result = self.transcribe(
                record["input"]["audio_path"],
                output_dir=str(run_dir),
                output_stem="score",
                title=record["title"],
                transcription_backend=options["backend"],
                capo=options["capo"],
                tab_hint_path=record["input"].get("tab_hint_path"),
                max_span=options["max_span"],
                max_attempts=options["max_attempts"],
                render_pdf=options["render_pdf"],
            )
            self.store.update(
                job_id,
                progress=0.94,
                phase="Packaging notation and review artifacts",
            )
            output_paths = dict(result.get("outputs", {}))
            output_paths["report"] = result.get("report_path")
            output_paths["source"] = record["input"]["audio_path"]
            artifacts: dict[str, dict[str, str]] = {}
            for name, raw_path in output_paths.items():
                if not raw_path:
                    continue
                path = Path(raw_path).resolve()
                if path.is_file() and (run_dir.resolve() == path.parent or run_dir.resolve() in path.parents):
                    artifacts[name] = {
                        "filename": path.name,
                        "url": f"/api/jobs/{job_id}/artifacts/{name}",
                    }

            tab_preview = ""
            if "tab" in artifacts:
                tab_preview = (run_dir / artifacts["tab"]["filename"]).read_text(
                    encoding="utf-8"
                )[:30000]
            chords: list[dict[str, Any]] = []
            if "chords" in artifacts:
                try:
                    chords = json.loads(
                        (run_dir / artifacts["chords"]["filename"]).read_text(
                            encoding="utf-8"
                        )
                    )[:200]
                except (json.JSONDecodeError, OSError):
                    chords = []

            quality = dict(result.get("quality", {}))
            fingering = self._plain(result.get("fingering"))
            timing = self._plain(result.get("timing"))
            summary = {
                "metrics": {
                    "heuristic_health": float(quality.get("score", 0)),
                    "selected_capo": int(result.get("selected_capo", 0)),
                    "assigned_ratio": float(fingering.get("assigned_ratio", 0)),
                    "note_count": len(result.get("events", [])),
                    "chord_count": len(result.get("chords", [])),
                    "bpm": float(timing.get("bpm", 0)),
                    "duration_seconds": float(timing.get("duration", 0)),
                },
                "profile": result.get("profile", "balanced"),
                "backend": result.get("transcription_backend", options["backend"]),
                "issues": quality.get("issues", []),
                "messages": result.get("messages", []),
                "tab_preview": tab_preview,
                "chords": chords,
                "artifacts": artifacts,
            }
            self.store.update(
                job_id,
                status="complete",
                progress=1.0,
                phase="First draft ready for review",
                result=summary,
                error=None,
            )
        except Exception as exc:
            self.store.update(
                job_id,
                status="failed",
                progress=0.0,
                phase="Transcription failed",
                error=str(exc) or exc.__class__.__name__,
            )
        finally:
            self.slots.release()
