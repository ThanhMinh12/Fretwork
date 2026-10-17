"""Real React + private FastAPI jobs, with browser-dispatched ZeroGPU work."""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'hf_space')]
import spaces  # Import before the existing model initializer.
import gradio as gr
from hf_space.app import initialize_models, cached_activations


def main():
    hosted = os.getenv('FRETWORK_SPACE_DEVICE', 'cuda') != 'cpu'
    os.environ['FRETWORK_HOSTED'] = str(hosted).lower()
    os.environ['FRETWORK_COOKIE_SECURE'] = str(hosted).lower()
    initialize_models()
    from fretwork_api.config import Settings
    from fretwork_api.main import create_app
    from fretwork_agents.workflow import run_agent
    from fretwork_core import transcription

    def transcribe(*args, **kwargs):
        cached_activations.cache_clear()
        transcription._GAPS_ACTIVATIONS_CACHE.clear()
        try:
            return run_agent(*args, **kwargs)
        finally:
            cached_activations.cache_clear()
            transcription._GAPS_ACTIVATIONS_CACHE.clear()

    config = replace(Settings.from_env(), data_dir=Path(os.getenv('FRETWORK_DATA_DIR', '/tmp/fretwork-fullstack')),
                     frontend_dir=ROOT/'web/dist', max_audio_seconds=30, max_upload_bytes=16*1024**2)
    server = create_app(config, transcribe, app_factory=gr.Server, queued_inference=True)
    def cleanup():
        while True:
            cutoff = datetime.now(UTC)-timedelta(hours=1)
            for job in server.state.store.list():
                if job['status'] not in {'queued', 'processing'} and datetime.fromisoformat(job['created_at']) < cutoff:
                    try:
                        server.state.store.delete(job['id'])
                    except (KeyError, FileNotFoundError):
                        pass  # The owner may have deleted this draft meanwhile.
            threading.Event().wait(300)
    threading.Thread(target=cleanup, daemon=True).start()
    server.launch(server_name='0.0.0.0', server_port=int(os.getenv('PORT', '7860')),
                  max_file_size='16mb', show_error=False, ssr_mode=False, run_history=False, footer_links=[],
                  blocked_paths=[str(config.data_dir.resolve())])


if __name__ == '__main__':
    main()
