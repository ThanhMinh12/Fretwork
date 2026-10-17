---
title: Fretwork
emoji: 🎸
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: 3.12.12
app_file: hf_space/app.py
startup_duration_timeout: 30m
suggested_hardware: zero-a10g
short_description: Guitar audio to chord-aware sheet music and tablature.
---

# Fretwork · free demo

Upload a 2–30 second guitar recording (16 MB maximum). The actual GAPS guitar
CRNN and Basic Pitch ONNX models propose notes; the existing consensus,
fingering, and harmony pipeline produces MusicXML, TAB, and an inspectable
report. The preview is engraved from the generated MusicXML with Verovio.

This is a **Gradio demonstration**, not the full React/FastAPI deployment.
The responsive React interface, asynchronous job API, private owner routes,
and frozen Python 3.11 CPU runtime remain in
[the source repository](https://github.com/ThanhMinh12/Fretwork).

The GPU runs only the GAPS activation pass. CPU decoding, Basic Pitch ONNX,
consensus, harmony, and engraving run after the GPU is released. The guitar
model is loaded on CPU, integrity checked, and moved to CUDA at root startup
without upstream's multi-GPU DataParallel wrapper. Per-recording activation
reuse stays in the parent process and is cleared after each job.

ZeroGPU has queues and daily visitor quotas. No paid hardware or credits are
required. Sessions expire after an hour or restart. Use only non-sensitive
audio: Gradio download URLs are shareable capability links, not authenticated
artifact routes. Clearing the workspace removes working files; cached outputs
may remain for up to an hour. Transcriptions are first drafts, not ground truth.

## Reproduce

Upload this README, requirements, and packages file to the Space root as
`README.md`, `requirements.txt`, and `packages.txt`; include `hf_space/*.py`,
`fretwork.py`, `chord_retrieval.py`, `fretwork_core/`, `fretwork_agents/`,
`fretwork_harmony/`, and `knowledge/`. Never upload recordings, credentials,
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
wheel and run `FRETWORK_SPACE_DEVICE=cpu python hf_space/app.py` from the root.
