from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from fretwork_agent import run_agent

from . import __version__
from .config import Settings
from .jobs import JobRunner, JobStore
from .security import install_security, owner_id
from .dispatch import DispatchRegistry
from pydantic import BaseModel, Field


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TAB_EXTENSIONS = {".txt", ".tab"}
ARTIFACT_MEDIA_TYPES = {
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "tab": "text/plain; charset=utf-8",
    "events": "application/json",
    "chords": "application/json",
    "report": "application/json",
    "pdf": "application/pdf",
}


async def _save_upload(upload: UploadFile, destination: Path, limit: int) -> None:
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _public(record: dict[str, Any]) -> dict[str, Any]:
    visible = {key: value for key, value in record.items() if key not in {"input", "owner_id"}}
    visible["source_filename"] = record["input"]["original_filename"]
    return visible


class DispatchCancellation(BaseModel):
    token: str = Field(max_length=128)


def create_app(
    settings: Settings | None = None,
    transcribe=run_agent,
    *, app_factory=FastAPI, queued_inference: bool = False,
) -> FastAPI:
    config = settings or Settings.from_env()
    store = JobStore(config)
    runner = JobRunner(config, store, transcribe)
    app = app_factory(
        title="Fretwork API",
        version=__version__,
        description="Asynchronous chord-aware fingerstyle transcription service.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = config
    app.state.store = store
    app.state.runner = runner
    dispatches = DispatchRegistry(max_pending=config.max_pending_jobs)

    if queued_inference:
        @app.api(name="run_job", concurrency_id="inference", concurrency_limit=1)
        def run_job(token: str) -> dict:
            job_id = dispatches.claim(token)
            runner._run(job_id)
            return {"id": job_id, "status": store.get(job_id)["status"]}

    @app.get("/api/runtime")
    async def runtime(request: Request):
        owner_id(request)
        return {"queued_inference": queued_inference, "max_upload_mb": config.max_upload_bytes // 1024**2,
                "max_audio_seconds": config.max_audio_seconds, "temporary": queued_inference}

    @app.post("/api/dispatch/cancel", status_code=204, response_class=Response)
    async def cancel_dispatch(request: Request, cancellation: DispatchCancellation):
        dispatches.cancel(cancellation.token, owner_id(request))
        return Response(status_code=204)
    install_security(app, config.data_dir, "FRETWORK", config.cors_origins, config.max_upload_bytes)

    def owned_job(request: Request, job_id: str) -> dict:
        try:
            record = store.get(job_id)
            if record.get("owner_id") != owner_id(request):
                raise KeyError(job_id)
            return record
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        active = sum(
            record["status"] in {"queued", "processing"} for record in store.list()
        )
        return {
            "status": "ok",
            "version": __version__,
            "worker_count": config.workers,
            "active_jobs": active,
            "processing": "local",
        }

    @app.get("/api/jobs")
    async def list_jobs(request: Request, limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[dict]:
        mine = [record for record in store.list() if record.get("owner_id") == owner_id(request)]
        return [_public(record) for record in mine[:limit]]

    @app.post("/api/jobs", status_code=202)
    async def create_job(
        request: Request,
        audio: Annotated[UploadFile, File(description="Solo-guitar recording")],
        title: Annotated[str, Form(max_length=160)] = "Fretwork Transcription",
        backend: Annotated[Literal["ensemble", "gaps", "basic_pitch"], Form()] = "ensemble",
        capo: Annotated[str, Form(pattern=r"^(auto|[0-9]|1[0-2])$")] = "auto",
        max_span: Annotated[int, Form(ge=2, le=7)] = 4,
        max_attempts: Annotated[int, Form(ge=1, le=3)] = 3,
        render_pdf: Annotated[bool, Form()] = False,
        tab_hint: Annotated[UploadFile | None, File()] = None,
    ) -> dict:
        original_name = audio.filename or "recording.wav"
        extension = Path(original_name).suffix.lower()
        if extension not in AUDIO_EXTENSIONS:
            await audio.close()
            if tab_hint:
                await tab_hint.close()
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported audio type. Use: {', '.join(sorted(AUDIO_EXTENSIONS))}",
            )
        hint_extension = Path(tab_hint.filename or "").suffix.lower() if tab_hint else ""
        if tab_hint and hint_extension not in TAB_EXTENSIONS:
            await audio.close()
            await tab_hint.close()
            raise HTTPException(status_code=415, detail="TAB hint must be .txt or .tab")

        if sum(path.stat().st_size for path in config.runs_dir.rglob("*") if path.is_file()) >= config.max_storage_bytes:
            raise HTTPException(status_code=507, detail="Storage is full; delete old drafts first")
        if not runner.slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="The worker queue is full. Try again shortly.")
        job_id = uuid.uuid4().hex
        run_dir = config.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=False)
        audio_path = run_dir / f"source{extension}"
        hint_path = run_dir / f"hint{hint_extension}" if tab_hint else None
        try:
            await _save_upload(audio, audio_path, config.max_upload_bytes)
            if tab_hint and hint_path:
                await _save_upload(tab_hint, hint_path, 2 * 1024 * 1024)
            # Tests can inject a runner; real requests must validate decoded
            # audio before committing a job or loading either model.
            if transcribe is run_agent or queued_inference:
                import soundfile as sf
                try:
                    try:
                        duration = sf.info(audio_path).duration
                    except sf.LibsndfileError:
                        import json
                        import shutil
                        import subprocess
                        executable = shutil.which("ffprobe")
                        if not executable:
                            raise ValueError("This audio format needs FFmpeg/ffprobe. Use WAV or install FFmpeg.")
                        probe = subprocess.run([executable, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(audio_path)], capture_output=True, check=True, timeout=15)
                        duration = float(json.loads(probe.stdout)["format"]["duration"])
                    if duration <= 0 or duration > config.max_audio_seconds:
                        raise ValueError(f"Recording must be at most {config.max_audio_seconds:g} seconds")
                except Exception as exc:
                    detail = str(exc) if isinstance(exc, ValueError) else "The recording could not be decoded. Try WAV or MP3."
                    raise HTTPException(status_code=422, detail=detail) from exc
            record = store.create(
                job_id,
                {
                    "title": title.strip() or Path(original_name).stem,
                    "owner_id": owner_id(request),
                    "input": {
                        "original_filename": original_name,
                        "audio_path": str(audio_path),
                        "tab_hint_path": str(hint_path) if hint_path else None,
                    },
                    "options": {
                        "backend": backend,
                        "capo": "auto" if capo == "auto" else int(capo),
                        "max_span": max_span,
                        "max_attempts": max_attempts,
                        "render_pdf": render_pdf,
                    },
                },
            )
        except Exception:
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)
            runner.slots.release()
            raise
        result = _public(record)
        if queued_inference:
            def expired():
                store.update(job_id, status="failed", phase="Inference request expired or could not start",
                             error="The browser queue could not start this job. Submit it again.")
                runner.slots.release()
            try:
                token = dispatches.register(owner_id(request), job_id, expired)
            except ValueError as exc:
                expired()
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            result["dispatch"] = {"endpoint": "/run_job", "token": token}
        else:
            runner.submit(job_id)
        return result

    @app.get("/api/jobs/{job_id}")
    async def get_job(request: Request, job_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            return _public(owned_job(request, job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.delete("/api/jobs/{job_id}", status_code=204, response_class=Response)
    async def delete_job(request: Request, job_id: str) -> Response:
        try:
            record = owned_job(request, job_id)
            if record["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail="A running job cannot be deleted")
            store.delete(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/artifacts/{artifact}")
    async def artifact(request: Request, job_id: str, artifact: str) -> FileResponse:
        try:
            record = owned_job(request, job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        details = (record.get("result") or {}).get("artifacts", {}).get(artifact)
        if not details:
            raise HTTPException(status_code=404, detail="Artifact not found")
        path = store.run_dir(job_id) / details["filename"]
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact file is missing")
        inline = artifact == "source"
        return FileResponse(
            path,
            filename=details["filename"],
            media_type=ARTIFACT_MEDIA_TYPES.get(artifact),
            content_disposition_type="inline" if inline else "attachment",
        )

    if config.frontend_dir and (config.frontend_dir / "index.html").is_file():
        assets = config.frontend_dir / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/" if queued_inference else "/{requested_path:path}", include_in_schema=False)
        async def frontend(requested_path: str = "") -> FileResponse:
            if requested_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Endpoint not found")
            candidate = (config.frontend_dir / requested_path).resolve()
            if requested_path and candidate.is_file() and config.frontend_dir in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(config.frontend_dir / "index.html")

    return app


app = create_app()

