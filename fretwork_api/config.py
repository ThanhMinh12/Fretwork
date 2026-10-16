from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    frontend_dir: Path | None
    max_upload_bytes: int
    cors_origins: tuple[str, ...]
    workers: int
    max_pending_jobs: int = 4
    max_audio_seconds: float = 180
    max_storage_bytes: int = 2 * 1024 ** 3

    @classmethod
    def from_env(cls) -> "Settings":
        frontend = os.getenv("FRETWORK_FRONTEND_DIR")
        origins = tuple(
            value.strip()
            for value in os.getenv("FRETWORK_CORS_ORIGINS", "http://localhost:5173").split(",")
            if value.strip()
        )
        return cls(
            data_dir=Path(os.getenv("FRETWORK_DATA_DIR", "data")).resolve(),
            frontend_dir=Path(frontend).resolve() if frontend else None,
            max_upload_bytes=int(os.getenv("FRETWORK_MAX_UPLOAD_MB", "200"))
            * 1024
            * 1024,
            cors_origins=origins,
            workers=max(1, int(os.getenv("FRETWORK_JOB_WORKERS", "1"))),
            max_pending_jobs=max(1, int(os.getenv("FRETWORK_MAX_PENDING_JOBS", "4"))),
            max_audio_seconds=float(os.getenv("FRETWORK_MAX_AUDIO_SECONDS", "180")),
            max_storage_bytes=int(os.getenv("FRETWORK_MAX_STORAGE_MB", "2048")) * 1024 ** 2,
        )

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def prepare(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
