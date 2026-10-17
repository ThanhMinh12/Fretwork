import io
import wave

import gradio as gr
import pytest
from fastapi.testclient import TestClient

from api_tests.test_api import fake_agent
from fretwork_api.config import Settings
from fretwork_api.main import create_app


def audio():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b'\0\0'*32000)
    return buffer.getvalue()


def test_private_api_and_single_use_queued_workflow(tmp_path):
    app = create_app(Settings(tmp_path, None, 1024*1024, (), 1), fake_agent,
                     app_factory=gr.Server, queued_inference=True)
    handler = app._deferred_apis[0][0]
    with TestClient(app) as owner, TestClient(app) as foreign:
        created = owner.post('/api/jobs', files={'audio': ('guitar.wav', audio())})
        assert created.status_code == 202
        job = created.json()
        token = job['dispatch']['token']
        assert job['status'] == 'queued'
        assert 'dispatch' not in owner.get('/api/jobs').json()[0]
        assert foreign.get('/api/jobs/'+job['id']).status_code == 404
        foreign.post('/api/dispatch/cancel', json={'token': token})
        assert handler(token)['status'] == 'complete'
        with pytest.raises(ValueError):
            handler(token)
        artifact = '/api/jobs/'+job['id']+'/artifacts/musicxml'
        assert owner.get(artifact).status_code == 200
        assert foreign.get(artifact).status_code == 404


def test_hf_embedding_uses_secure_partitioned_cookies(tmp_path, monkeypatch):
    monkeypatch.setenv('FRETWORK_HOSTED','true')
    monkeypatch.setenv('FRETWORK_COOKIE_SECURE','true')
    app = create_app(Settings(tmp_path,None,1024*1024,(),1), fake_agent,
                     app_factory=gr.Server, queued_inference=True)
    with TestClient(app, base_url='https://test') as client:
        response = client.get('/api/runtime')
        cookie = response.headers['set-cookie']
        assert 'samesite=none' in cookie.lower() and 'secure' in cookie.lower() and 'Partitioned' in cookie
        assert "frame-ancestors 'self' https://huggingface.co" in response.headers['content-security-policy']
        assert 'x-frame-options' not in response.headers
        assert "connect-src 'self' https://huggingface.co" in response.headers['content-security-policy']
        assert client.post('/gradio_api/upload', files={'file': ('unused.wav', b'no')}).status_code == 404
        assert client.get('/gradio_api/file=/tmp/anything').status_code == 404
