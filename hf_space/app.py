"""Free ZeroGPU demo; retains the independent React/FastAPI deployment."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'hf_space'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'false')

import spaces  # Initialize ZeroGPU before torch is imported.
import gradio as gr
from demo import DemoWorkspaces, render_score, validate_audio

ORIGINAL_GAPS_ACTIVATIONS = None


@spaces.GPU(duration=45)
def guitar_activations(path):
    return ORIGINAL_GAPS_ACTIVATIONS(path)


@lru_cache(maxsize=1)
def cached_activations(path):
    # Parent-process cache: ZeroGPU child-process global mutations do not persist.
    return guitar_activations(path)


def initialize_models():
    global ORIGINAL_GAPS_ACTIVATIONS
    from bootstrap import install_adapters
    install_adapters()
    import torch
    from hf_midi_transcription import MidiTranscriptionModel
    from fretwork_core import transcription
    from fretwork_core.checkpoints import verified_gaps_checkpoint
    torch.set_num_threads(2)
    model = MidiTranscriptionModel(instrument='guitar', device='cpu', batch_size=8,
                                   checkpoint_path=str(verified_gaps_checkpoint()))
    # Avoid upstream DataParallel; place weights at root startup for ZeroGPU.
    model.transcriptor.model.to(os.getenv('FRETWORK_SPACE_DEVICE', 'cuda')).eval()
    transcription._GAPS_MODEL = model
    ORIGINAL_GAPS_ACTIVATIONS = transcription._gaps_activations
    transcription._gaps_activations = cached_activations


def transcribe(store, session, filename):
    from fretwork_agents.workflow import run_agent
    from fretwork_core import transcription
    source = validate_audio(filename)
    work = store.prepare(session)
    cached_activations.cache_clear()
    transcription._GAPS_ACTIVATIONS_CACHE.clear()
    try:
        state = run_agent(str(source), output_dir=str(work), output_stem='fretwork',
                          title='Fretwork transcription', transcription_backend='ensemble', max_attempts=2)
        outputs = state['outputs']
        preview = render_score(outputs['musicxml'])
        tab = Path(outputs['tab']).read_text(encoding='utf-8')
        files = [value for value in outputs.values() if value] + [state['report_path']]
        message = (f"{len(state['events'])} notes · {len(state.get('chords', []))} chords · "
                   f"{state['timing'].bpm:.0f} BPM · capo {state['selected_capo']}\n\n"
                   'ML-assisted first draft. Check notes, rhythm, and capo by ear.')
        return message, preview, tab, files
    finally:
        cached_activations.cache_clear()
        transcription._GAPS_ACTIVATIONS_CACHE.clear()


def build_app(store):
    def run(session, filename):
        try:
            return transcribe(store, session, filename)
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc
    with gr.Blocks(title='Fretwork', delete_cache=(300, 3600)) as demo:
        session = gr.State(value=store.new, time_to_live=3600, delete_callback=store.delete)
        gr.Markdown('# Fretwork\nGuitar audio to sheet music and TAB.')
        gr.Markdown('Free GPU demo · Queues and daily limits apply. Use non-sensitive audio only; download links are shareable. Files expire after an hour or restart.')
        # File avoids decoding an arbitrarily long compressed file in Gradio
        # before our metadata-based duration check.
        audio = gr.File(label='Guitar recording · 2–30 seconds / 16 MB', file_types=['audio'])
        with gr.Row():
            submit = gr.Button('Generate sheet music', variant='primary')
            clear = gr.Button('Clear workspace')
        status = gr.Markdown()
        preview = gr.HTML(label='Sheet music preview')
        tab = gr.Textbox(label='Chord-aware TAB', lines=10, interactive=False)
        files = gr.File(label='Download MusicXML, TAB, events, chords, and report', file_count='multiple', interactive=False)
        gr.Markdown('[Full React/FastAPI source](https://github.com/ThanhMinh12/Fretwork) · [Thanh’s portfolio](https://thanhminh12.github.io/)')
        opts = dict(concurrency_id='fretwork-demo', concurrency_limit=1)
        submit.click(run, [session, audio], [status, preview, tab, files], api_name='transcribe', **opts)
        def reset(s):
            store.delete(s)
            return store.new(), None, '', '', '', None
        clear.click(reset, [session], [session, audio, status, preview, tab, files], api_name='clear_workspace', **opts)
    return demo.queue(max_size=12, default_concurrency_limit=1)


if __name__ == '__main__':
    initialize_models()
    store = DemoWorkspaces(Path('/tmp/fretwork-demo-sessions'))
    build_app(store).launch(server_name='0.0.0.0', server_port=int(os.getenv('PORT', '7860')), max_file_size='16mb', show_error=False)
