# FastAPI service

The web application and third-party clients use the same REST interface. Open
`/api/docs` on a running server for interactive OpenAPI documentation.

## Job lifecycle

1. `POST /api/jobs` accepts multipart audio, options, and an optional TAB hint.
2. The service stores the input under a random run ID and returns `202` with a
   queued job immediately.
3. One local worker runs the existing LangGraph transcription pipeline. Poll
   `GET /api/jobs/{id}` for `progress`, `phase`, and terminal status.
4. A completed job contains metrics, critic notes, the agent trace, a TAB
   preview, chord labels, and allowlisted artifact URLs.

Jobs and outputs are written atomically beneath `FRETWORK_DATA_DIR/runs`. A job
that was active during a server restart is marked failed instead of being left
permanently in a processing state.

## Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Worker and service status |
| `POST` | `/api/jobs` | Upload and enqueue a transcription |
| `GET` | `/api/jobs` | List recent local jobs |
| `GET` | `/api/jobs/{id}` | Poll progress or read results |
| `DELETE` | `/api/jobs/{id}` | Delete a finished run and its artifacts |
| `GET` | `/api/jobs/{id}/artifacts/{name}` | Download an allowlisted output |

Supported artifact names are `source`, `musicxml`, `tab`, `events`, `chords`,
`report`, and `pdf` when LilyPond successfully rendered one.

## Example

```bash
curl -X POST http://localhost:8000/api/jobs \
  -c /tmp/fretwork-cookies.txt -b /tmp/fretwork-cookies.txt \
  -F audio=@take.wav \
  -F title='Sunday arrangement' \
  -F backend=ensemble \
  -F capo=auto \
  -F max_span=4 \
  -F max_attempts=3
```

The server intentionally defaults to one worker. Each transcription uses large
local models and CPU/GPU memory; increasing concurrency without measuring that
resource pressure can make every job slower or exhaust memory.

Retain the session cookie for polling and downloads. History, job reads,
deletion, and artifacts are scoped to that browser session; another session gets
404 for your job IDs. Cross-origin writes are rejected unless explicitly
allowlisted. A full queue returns 429; oversize requests return 413; invalid or
overlong audio returns 422; exhausted storage returns 507. Uploads require
Content-Length. The session secret is generated in the persistent data directory
if no stable environment secret is configured.

