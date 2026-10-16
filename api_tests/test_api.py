import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from fretwork_api.config import Settings
from fretwork_api.main import create_app


@dataclass
class Fingering:
    assigned_ratio: float = 0.98


@dataclass
class Timing:
    bpm: float = 112.0
    duration: float = 4.0


def fake_agent(audio_path: str, *, output_dir: str, **kwargs):
    output = Path(output_dir)
    (output / "score.musicxml").write_text("<score-partwise/>", encoding="utf-8")
    (output / "score_tab.txt").write_text("Capo: 2\ne|--0--|", encoding="utf-8")
    (output / "score_events.json").write_text("[]", encoding="utf-8")
    (output / "score_chords.json").write_text(
        json.dumps([{"shape_chord": "C", "start": 0.0}]), encoding="utf-8"
    )
    report = output / "score_report.json"
    report.write_text("{}", encoding="utf-8")
    return {
        "outputs": {
            "musicxml": str(output / "score.musicxml"),
            "tab": str(output / "score_tab.txt"),
            "events": str(output / "score_events.json"),
            "chords": str(output / "score_chords.json"),
            "pdf": None,
        },
        "report_path": str(report),
        "quality": {"score": 0.91, "issues": []},
        "fingering": Fingering(),
        "timing": Timing(),
        "selected_capo": 2,
        "events": [object(), object()],
        "chords": [object()],
        "messages": ["Acoustic consensus agent accepted 2 notes"],
        "profile": "balanced",
        "transcription_backend": kwargs["transcription_backend"],
    }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_job_lifecycle_and_artifact_download(tmp_path: Path) -> None:
    settings = Settings(tmp_path, None, 1024 * 1024, (), 1)
    transport = httpx.ASGITransport(app=create_app(settings, fake_agent))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/jobs",
            files={"audio": ("take.wav", b"RIFF-fake", "audio/wav")},
            data={"title": "My take", "backend": "ensemble", "capo": "auto"},
        )
        assert created.status_code == 202
        job = created.json()
        for _ in range(100):
            response = await client.get(f"/api/jobs/{job['id']}")
            job = response.json()
            if job["status"] in {"complete", "failed"}:
                break
            await __import__("asyncio").sleep(0.01)

        assert job["status"] == "complete"
        assert job["result"]["metrics"]["heuristic_health"] == 0.91
        assert "Capo: 2" in job["result"]["tab_preview"]
        assert job["result"]["artifacts"]["musicxml"]["url"].endswith(
            "/artifacts/musicxml"
        )
        assert b"score-partwise" in (tmp_path / "runs" / job["id"] / "score.musicxml").read_bytes()
        assert (await client.delete(f"/api/jobs/{job['id']}")).status_code == 204
        assert (await client.get(f"/api/jobs/{job['id']}")).status_code == 404
        assert not (tmp_path / "runs" / job['id']).exists()


@pytest.mark.anyio
async def test_rejects_unsupported_audio(tmp_path: Path) -> None:
    settings = Settings(tmp_path, None, 1024 * 1024, (), 1)
    transport = httpx.ASGITransport(app=create_app(settings, fake_agent))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/jobs", files={"audio": ("notes.txt", b"nope", "text/plain")}
        )
    assert response.status_code == 415


@pytest.mark.anyio
async def test_sessions_cannot_read_or_delete_other_users_jobs(tmp_path: Path):
    transport = httpx.ASGITransport(app=create_app(Settings(tmp_path, None, 1024 * 1024, (), 1), fake_agent))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as owner, httpx.AsyncClient(transport=transport, base_url="http://test") as stranger:
        response = await owner.post('/api/jobs', files={'audio': ('take.wav', b'RIFF-fake', 'audio/wav')})
        assert response.status_code == 202
        job_id = response.json()['id']
        assert len((await owner.get('/api/jobs')).json()) == 1
        assert (await stranger.get('/api/jobs')).json() == []
        for path in [f'/api/jobs/{job_id}', f'/api/jobs/{job_id}/artifacts/source']:
            assert (await stranger.get(path)).status_code == 404
        assert (await stranger.delete(f'/api/jobs/{job_id}')).status_code == 404
        assert 'owner_id' not in (await owner.get(f'/api/jobs/{job_id}')).json()


@pytest.mark.anyio
async def test_rejects_cross_origin_writes(tmp_path: Path):
    transport = httpx.ASGITransport(app=create_app(Settings(tmp_path, None, 1024, (), 1), fake_agent))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.post('/api/jobs', headers={'Origin':'https://evil.example'}, files={'audio':('take.wav', b'RIFF', 'audio/wav')})
        assert result.status_code == 403


@pytest.mark.anyio
async def test_full_queue_is_rejected_without_creating_a_job(tmp_path: Path):
    app = create_app(Settings(tmp_path, None, 1024, (), 1, max_pending_jobs=1), fake_agent)
    assert app.state.runner.slots.acquire(blocking=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post('/api/jobs', files={'audio':('take.wav', b'RIFF', 'audio/wav')})
        assert response.status_code == 429
        assert (await client.get('/api/jobs')).json() == []
    app.state.runner.slots.release()
