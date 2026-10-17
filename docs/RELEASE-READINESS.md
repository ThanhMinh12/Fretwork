# Release verification — October 16, 2026

The application is verified for local/portfolio use, not certified as an
unrestricted multi-tenant service. Deployment is a separate milestone.

## Fixed and verified

- Modern compatible FastAPI, Pydantic, AnyIO and **MCP 2.3.0**; the MCP client
  discovery/tool-call test now runs rather than skipping.
- CPU runtime lock, pinned guitar-adapter Git commits, and a supported installer
  which installs that frozen closure. A fresh Python 3.11 virtual environment
  completed installation and runtime validation. Two Basic Pitch upstream
  metadata exceptions are documented: unused TensorFlow and resampy (the ONNX
  path is used and resampy 0.4.3 is required by the guitar adapter).
- Both the Docker image and the native runtime pin the guitar checkpoint's
  upstream revision and verify its SHA-256 before model loading. A mismatched
  custom checkpoint is rejected without deleting the user's file.
- **41 core tests + 5 API tests passed, zero skipped.** API tests include private
  sessions, foreign-job/artifact denial, deletion, cross-origin write rejection,
  and a full-queue response. Logic tests use controlled inputs; they are separate
  from real inference.
- A real 30-second `019_Vpswc.wav` recording went through HTTP upload, the GAPS
  model, Basic Pitch, consensus, fingering, harmony, and artifact export in
  **182.85 seconds** on CPU including cold setup. Output: **99 note events,
  29 chord labels**, nonempty MusicXML/TAB/events/chords/report/audio downloads.
- Chromium displayed the completed real job and **engraved MusicXML notation**
  with OpenSheetMusicDisplay, plus TAB/chords. A stranger session could not read
  the job or original recording.
- Flat and slash-chord spelling is translated for music21. Unsupported chord
  extensions remain visible text annotations instead of disappearing from the
  score. A regression test covers seven flat, inversion, extension and power
  chord examples, including a power chord which otherwise exports as no-chord.
  The final genuine two-model HTTP/browser replay verified **29 score
  annotations for all 29 predicted chords** (99 notes), including the extended
  chord text fallback.
- Comparison of that fresh output with the clip's reference MIDI: onset F1
  **0.9447**, offset F1 **0.6231** (100 reference notes, 99 predictions). This is
  one in-domain classical-guitar excerpt, not a general accuracy claim or a
  chord-label benchmark. The internal 96% health is **not accuracy**.
- Upload length/size checks, a bounded worker queue, persistent signed sessions,
  safe artifact paths, a storage threshold, and deletion are implemented.

## Operational contract

Run one API process and retain the complete `/data` volume, including the signing
secret. A server restart marks interrupted work as failed, with an explicit retry
message; it does not pretend the job is still running. Use HTTPS and secure
cookies when hosted. Session isolation is not account-based authentication;
use an authenticated reverse proxy for private/sensitive deployments.

Audio is processed on the server. A hosted deployment is not browser-only or
confined to the visitor's computer. Scores remain editable drafts; PDF engraving
is optional and requires LilyPond. External-user identity, automated retention,
object storage and a distributed durable queue are future production work.

## Deployment status / authorization

No new AWS resources were provisioned. The existing Detour App Runner has
512 MB RAM and ephemeral H2 storage, and the configured region has no RDS/EC2
host to reuse. Real ML workers exceeded that memory budget during testing.
The full React/FastAPI stack still needs suitable hosting and persistent
storage; its deployment must not be confused with the separate demo below.

Source is public at [ThanhMinh12/Fretwork](https://github.com/ThanhMinh12/Fretwork).
Recordings, model weights and generated user data are excluded from Git.
The user subsequently authorized free non-AWS hosting. A separate
[live Gradio demo](https://huggingface.co/spaces/ThanhMinh12/Fretwork) now runs
on free ZeroGPU (confirmed requested and actual hardware `zero-a10g`, no paid
plan/hardware/credits enabled). The original React/FastAPI application and
Python 3.11 CPU lock are unchanged. The hosted profile uses Python 3.12 /
PyTorch 2.8 and shares the real transcription, consensus and harmony code.

The hosted API processed a real 30-second guitar excerpt with both actual
models: **99 notes, 29 chord labels**, two engraved SVG score pages, nonempty
TAB, and five downloadable artifacts (MusicXML, TAB, notes, chords and report).
The observed run took **25.95 s** including the hosted workflow/transfer; this
is a single run, not a latency guarantee. The test workspace was cleared.
Seven hosted-adapter tests cover duration limits, bounded isolated workspaces,
targeted cleanup, and actual MusicXML-to-SVG engraving.

Only GAPS activation inference allocates a GPU; decoding, Basic Pitch ONNX,
the agent workflow, harmony and engraving run on CPU. CUDA placement happens
at startup without upstream DataParallel; the parent caches activations only
for the current recording and clears them after each job.

The hosted demo accepts 2–30 second recordings, at most 16 MB, with bounded
queues and 12 working sessions. Working sessions and registered Gradio cache
files expire after an hour or restart. Download links are shareable capability
links, **not** the original API's private artifact routes. The UI explicitly
asks for non-sensitive uploads. See [hosted setup](../hf_space/README.md).

Core/API tests, the React build, the full container build and non-root startup
checks pass in [GitHub Actions](https://github.com/ThanhMinh12/Fretwork/actions/runs/38979949440).
The image serves both API and frontend, runs as UID 10001, and creates a private
session-secret file. Actual model inference is separately verified by the
native real-audio regression above; the container startup test is not an
end-to-end inference benchmark. Docker Desktop remains unavailable locally.

## Reproduce

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m pytest api_tests -q
# With the actual API running, and a real guitar recording:
python scripts/smoke_real.py recording.wav --url http://127.0.0.1:8000 --browser
```

The optional browser check requires Playwright's Chromium installation.
