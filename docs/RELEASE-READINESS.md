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
Do not label this application publicly deployed until suitable authorized
compute and persistent storage are available.

Source is public at [ThanhMinh12/Fretwork](https://github.com/ThanhMinh12/Fretwork).
Recordings, model weights and generated user data are excluded from Git.
The portfolio links to the source and accurately labels hosting as pending.

Core/API tests and the React build pass in GitHub Actions. Container builds and
non-root startup checks run there because Docker Desktop integration is
unavailable in this WSL environment; inspect the linked workflow before
claiming a particular container check has passed.

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
