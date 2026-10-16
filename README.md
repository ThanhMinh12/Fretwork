# Fretwork

[![Verify](https://github.com/ThanhMinh12/Fretwork/actions/workflows/verify.yml/badge.svg)](https://github.com/ThanhMinh12/Fretwork/actions/workflows/verify.yml)

Fretwork is a free, local-first multi-agent system that turns a solo
fingerstyle-guitar recording into a playable first-draft score:

- standard notation with separate bass and melody/chord voices (`.musicxml`)
- capo-relative chord shapes such as `A`, `Dm`, `Cadd9`, and `G/B`
- capo-relative, tempo-aware ASCII TAB (`.txt`)
- editable note/fingering and chord-timeline JSON
- an audit report with agent decisions, warnings, and ranked capo candidates
- an optional PDF when LilyPond is installed

No third-party AI API key, subscription, or paid model is involved. Audio stays
on the machine running Fretwork. When self-hosted remotely, recordings are
uploaded to that server—not processed inside the visitor's browser.

The application is verified locally with real audio. Public hosting is pending
suitable existing AWS capacity; see the [release report](docs/RELEASE-READINESS.md).

## How it works

Two independent models propose notes; the workflow reconciles their predictions,
assigns playable fingerings, infers chord labels, and exports an editable score:

```text
React + TypeScript upload, job tracking, and artifact review UI
      ↓ REST
FastAPI job service
      ↓
LangGraph supervisor + LangChain specialist-agent contracts
      ↓
GAPS guitar CRNN proposal + Basic Pitch independent review
      ↓
acoustic consensus agent → bass / melody interpretation
      ↓
capo search + exact full-trellis Viterbi fingering agent
      ↓
local chord-template retrieval → grounded harmony-label agent
      ↓
critic supervisor ── concrete issue ──↺ bounded specialist retry
      ↓
publishing agent → MusicXML + chord-aware TAB + JSON + report
```

This is an artifact/review/reconciliation workflow. The guitar-trained
recurrent model proposes notes; an independent general-purpose model reviews
the same recording; the consensus agent records agreements and admits only
very strong reviewer-only notes. Other agents own voice interpretation,
harmony retrieval, fingering, criticism, and publishing through typed shared
LangGraph state. The critic autonomously routes bounded corrective decoding.

The agents are local policy/ML agents, not hosted chat models. Adding an LLM
would not recover missing audio evidence and would violate the free/offline
constraint. LangChain Core supplies inspectable runnable contracts; LangGraph
owns shared state, supervision, conditional routing, and retries.

The harmony stage is local RAG. It retrieves scored records from the versioned
`knowledge/chords.json` corpus, grounds them against pitch-class and bass
evidence across arpeggio-aware beat windows, and Viterbi-decodes close
candidates with a small adjacent-chord transition prior. Every JSON label
records its retrieved template ID and score. The musical prior annotates the
result; it never rewrites detected pitches.

## Quick start with Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>. The React client uploads recordings to the
FastAPI service, polls asynchronous job progress, previews TAB/chords/agent
decisions, and downloads every generated artifact. Finished files persist in
the `fretwork-data` Docker volume.

The first build downloads Python packages and the pinned 99 MB guitar
checkpoint, then verifies its published SHA-256. Transcription is offline
after the build. CPU is supported, although the recurrent model is slower than
the fast Basic Pitch-only option.

## Native install

The frozen CPU runtime is verified on Linux x86-64 with Python 3.11.
On Ubuntu/Debian:

```bash
sudo apt-get install ffmpeg git libsndfile1 libgomp1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
bash scripts/install.sh
```

Start the FastAPI service:

```bash
python app.py
```

Open <http://localhost:8000/api/docs> for the REST API. For frontend
development, install Node 22+ and run `npm ci && npm run dev` from `web/`;
Vite proxies `/api` to the Python server.

See [the API and job lifecycle](docs/API.md) for endpoint details.

The installer consumes a complete, frozen CPU dependency lock, including the
real MCP v2 SDK. Frozen installation avoids an upstream unpinned Git dependency
conflicting with the exact same revision pinned by this project. Basic Pitch is
installed separately because its upstream metadata unnecessarily requires
TensorFlow and an older resampy; Fretwork uses its bundled ONNX model alongside
the resampy version required by the guitar model. `scripts/check_runtime.py`
validates the complete installed dependency closure with only those two explicit
metadata exceptions. Do not replace the supported installer with a bare
`pip install -r requirements.txt`; that file is the input to lock generation.

Jobs, downloads, and history are isolated by a signed browser-session cookie.
Keep `/data/.session-secret` and the data volume together across restarts. This
is browser-session isolation, not a user-account/login system; clearing cookies
loses access to old runs. For a private hosted service, put an authenticated
HTTPS reverse proxy in front and set `FRETWORK_COOKIE_SECURE=true`.

Recordings are limited to three minutes by default, with a bounded four-job
queue and a 2 GB storage threshold. Delete finished drafts to free space.
The React client previews engraved MusicXML notation as well as TAB and chords.
The generated score remains a draft to review by ear, not guaranteed ground truth.

Run `python scripts/smoke_real.py recording.wav --url http://127.0.0.1:8000`
for an actual two-model HTTP workflow. Add `--browser` with Playwright installed
to verify the rendered notation preview. See `docs/RELEASE-READINESS.md` for
measured verification and deployment limits.

Or use the agent directly:

```bash
python fretwork_agent.py my_anime_cover.mp3 --backend ensemble --capo auto
python fretwork_agent.py my_take.wav --capo 3 --tab-hints rough.tab --pdf
```

Useful controls:

- `--backend ensemble` is the default CRNN + Basic Pitch review; `gaps` and
  `basic_pitch` run one acoustic specialist alone.
- `--capo auto` ranks capo 0–7 by open strings, reachability, stretch, fret
  height, and hand movement. Supply the known number whenever possible.
- `--max-span 4` sets the comfortable fretted chord span.
- `--max-attempts 1|2|3` bounds supervisor retries.
- `--tab-hints rough.tab` biases ambiguous positions toward a partial TAB.

## Chord and capo semantics

Chord symbols describe the shape your hand forms. A sounding D-major harmony
with capo 2 is printed as `C`, because `C` is the shape you play. Standard
notation retains sounding pitches. The chord JSON records both `shape_chord`
and `sounding_chord`, timing, supporting pitch classes, the retrieved template
ID, and an uncalibrated retrieval score.

Capo placement cannot be uniquely recovered from sound. Auto mode returns the
most playable hypothesis and records the complete ranking; it is not ground
truth. Enter the known capo whenever possible.

## Exact fingering optimization

A sounding pitch can occur in several places. Fretwork groups attacks within
60 ms, enumerates every distinct-string chord shape, and runs exact Viterbi
dynamic programming over the full trellis:

```text
dp[i][s] = intra(s) + min(dp[i-1][p] + transition(p, s))
```

Transition cost includes hand-center shift, per-string fret changes,
released/added strings, and a strong penalty for stealing a still-ringing
string for a different pitch. No beam pruning or center-only predecessor
collapse is used; the algorithm is the literal `O(n·k²)` recurrence.
Impossible ML clusters are marked partially unassigned instead of being hidden.

## MCP server

`mcp_server.py` is a real local stdio Model Context Protocol v2 server. It
exposes tools that run the multi-agent transcription graph on an allowlisted
local path, retrieve chord knowledge, and inspect report JSON:

```bash
python mcp_server.py
```

Point any MCP v2 host at that command. Add input roots with
`FRETWORK_MCP_INPUT_ROOTS` (OS-path-separated) and set
`FRETWORK_MCP_OUTPUT_ROOT` if needed. URLs are rejected, and recurrent tensors
never cross MCP—only small structured artifacts and local paths do.

## Accuracy and scope

Best input is a clean solo-guitar performance. The default guitar CRNN uses
recurrent onset/offset/frame decoding, while Basic Pitch provides independent
disagreement evidence. WAV, MP3, FLAC, OGG, and M4A are supported.

This version does **not** turn a full anime opening with vocals, drums, synths,
and orchestra into a new fingerstyle arrangement. Use an isolated solo-guitar
cover or stem.

Treat every result as a draft to check by ear. The UI's “heuristic health” is
not an accuracy percentage; measured accuracy requires an annotated reference.
Fast ornaments, muted bass, harmonics, rubato, alternate tunings, local tempo
changes, and dense voicings remain difficult. MusicXML opens in MuseScore for
correction and final engraving.

### Reproducible evaluation

The repository includes a deterministic 15-piece benchmark drawn only from
the official GAPS `test` split. Its manifest pins the dataset revision and
every source checksum; the downloader keeps 30-second excerpts locally and
does not commit the audio. The evaluator reports standard pitch/onset and
pitch/onset/offset metrics for GAPS, Basic Pitch, their conservative consensus,
and a naïve union ablation, plus
reviewer agreement, admissions, threshold rejections, and octave-conflict
rejections. It caches raw predictions so interrupted CPU runs resume safely.

```bash
python -m evaluation.download_gaps
python -m evaluation.run_transcription
python -m benchmarks.fingering --notes 100 250 500 1000
```

Do not tune on this test subset. Add a separate validation manifest for policy
or threshold changes. GAPS is classical-guitar data, so its results validate
the evaluation plumbing and in-domain transcription—not anime-cover accuracy.

The pinned run in `evaluation/results/gaps_test_15_summary.json` measured
**94.22% pitch/onset F1** and **63.01% pitch/onset/offset F1** over 2,150
reference notes. A naïve union with the reviewer's unmatched proposals fell to
83.22% onset F1; the conservative gate blocked 847 unsupported decisions and
preserved 11.00 F1 points without degrading any of the 15 clips. It matched the
stronger GAPS proposal rather than improving it, which is the honest result.

The committed CPU fingering benchmark maps a mixed 1,000-note passage across
500 attack clusters in a 1.11-second median with every note assigned. Its
largest cluster layer contains 54 complete candidate shapes.

## Model and protocol provenance

- Guitar acoustic model and checkpoint: [GAPS paper](https://arxiv.org/abs/2408.08653)
  and [official model repository](https://huggingface.co/xavriley/midi-transcription-models).
  The Docker build pins a repository revision and verifies checkpoint SHA-256
  `65483e7c0e340a90415b15b520687587698c8c728f5fa470a205f13ee45c6513`.
- Independent reviewer: [Spotify Basic Pitch](https://github.com/spotify/basic-pitch).
- Protocol server: [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).

The GAPS benchmark is for classical-guitar data; Fretwork has not yet been
measured on a labeled anime fingerstyle-cover test set. “Recommended” means
the most evidence-rich local mode in this app, not a published accuracy claim.

## Development

```bash
python -m unittest discover -s tests -v
python -m pytest api_tests -q
cd web && npm install && npm run build
```

Key files:

- `fretwork.py` — stable core facade and direct CLI
- `fretwork_core/models.py` — shared musical data contracts
- `fretwork_core/transcription.py` — acoustic backends, consensus, and timing
- `fretwork_core/fingering.py` — voice hints, capo ranking, and exact Viterbi
- `fretwork_core/notation.py` — MusicXML, chord-aware TAB, and export
- `fretwork_agent.py` — stable agent facade and CLI
- `fretwork_agents/state.py` — typed agent/artifact contracts
- `fretwork_agents/roles.py` — specialist policies and publishing
- `fretwork_agents/workflow.py` — LangGraph topology and runner
- `chord_retrieval.py` — stable harmony facade
- `fretwork_harmony/knowledge.py` + `knowledge/chords.json` — local RAG index
- `fretwork_harmony/inference.py` — chord evidence and sequence Viterbi
- `mcp_server.py` — MCP v2 tools and filesystem security boundary
- `fretwork_api/` — FastAPI uploads, asynchronous jobs, artifact downloads, and history
- `web/` — responsive React + TypeScript job and output-review interface
- `app.py` — local FastAPI development entrypoint
- `docs/architecture.md` — contracts and design details
- `evaluation/` — pinned held-out manifest, downloader, metrics, and ablation
- `benchmarks/fingering.py` — deterministic whole-song optimization benchmark

## License

Fretwork is MIT-licensed. Dependencies and model weights retain their own
licenses.
