---
title: Fretwork
emoji: 🎸
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: hf_space/fullstack.py
startup_duration_timeout: 30m
suggested_hardware: zero-a10g
short_description: Guitar audio to chord-aware sheet music and tablature.
---

# Fretwork · free demo

Upload a 2–30 second guitar recording (16 MB maximum). The actual GAPS guitar
CRNN and Basic Pitch ONNX models propose notes; the existing consensus,
fingering, and harmony pipeline produces MusicXML, TAB, and an inspectable
report. React engraves the generated MusicXML with OpenSheetMusicDisplay.

The real **React/TypeScript interface and FastAPI job/artifact routes** run
here. Gradio Server mode supplies queued GPU transport, not a replacement UI.
React creates a private job over REST, then submits a single-use dispatch token
through `@gradio/client` to retain visitor quota handling. Downloads and polling
use the signed-session API. The frozen Python 3.11 CPU runtime remains in
[the source repository](https://github.com/ThanhMinh12/Fretwork).

The GPU runs only the GAPS activation pass. CPU decoding, Basic Pitch ONNX,
consensus, harmony, and engraving run after the GPU is released. The guitar
model is loaded on CPU, integrity checked, and moved to CUDA at root startup
without upstream's multi-GPU DataParallel wrapper. Per-recording activation
reuse stays in the parent process and is cleared after each job.

ZeroGPU has queues and daily visitor quotas. No paid hardware or credits are
required. Finished/failed drafts expire after an hour (checked every five
minutes) or are lost on restart. Use only non-sensitive audio. Artifact routes
enforce signed browser-session ownership, including inside the HF iframe with
secure partitioned cookies. Delete a draft to remove its files immediately.
Browser-session isolation is not account login. Scores are editable drafts.

## Reproduce

Upload this README, requirements, and packages file to the Space root as
`README.md`, `requirements.txt`, and `packages.txt`; include `hf_space/*.py`,
`fretwork.py`, `fretwork_agent.py`, `chord_retrieval.py`, `fretwork_core/`, `fretwork_agents/`,
`fretwork_harmony/`, `fretwork_api/`, `knowledge/`, and compiled `web/dist/`.
Build the frontend with `npm ci` and `npm run build` in `web/` before uploading.
Never upload recordings, credentials,
model caches, outputs, or evaluation data. Select **ZeroGPU**, not paid hardware.

`bootstrap.py` installs three explicitly pinned upstream distributions with
`--no-deps` before model imports. This is intentional: Basic Pitch's metadata
unconditionally requests unused TensorFlow and an incompatible resampy, and
the guitar adapter references an unpinned Git dependency. Their actual runtime
dependencies are listed in this separate hosted requirements file; no
TensorFlow or paid external API is used. Code and checkpoint revisions match
the audited full-stack app. Weights download from their original public hosts,
not from this repository.

For a local CPU smoke test, install the hosted requirements with a CPU PyTorch
wheel and run `FRETWORK_SPACE_DEVICE=cpu python hf_space/fullstack.py` from the root.

The former widget UI is retained as `hf_space/app.py` for rollback.
