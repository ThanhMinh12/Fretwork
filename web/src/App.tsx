import { FormEvent, useEffect, useRef, useState } from "react";
import { api, type Runtime } from "./api";
import type { Job } from "./types";
import ScorePreview from "./ScorePreview";

const audioAccept = ".wav,.mp3,.m4a,.flac,.ogg";

function Icon({ name }: { name: "upload" | "download" | "check" | "clock" }) {
  const paths = {
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M4 17v3h16v-3" /></>,
    download: <><path d="M12 4v12m0 0 5-5m-5 5-5-5" /><path d="M4 20h16" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  };
  return <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function Status({ job }: { job: Job }) {
  const failed = job.status === "failed";
  const progress = Math.min(100, Math.max(0, Math.round(job.progress * 100)));
  return <section className={`run-status panel ${failed ? "failed" : ""}`} aria-live="polite">
    <div className="status-heading"><Icon name="clock" /><h2>{failed ? "Couldn’t create this score" : job.status === "queued" ? "Your recording is queued" : "Creating your sheet music"}</h2>{!failed && <span>{progress}%</span>}</div>
    <p>{failed ? job.error ?? "Please try again with a clean solo-guitar recording." : job.phase}</p>
    {!failed && <><div className="progress" role="progressbar" aria-label="Transcription progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div><p className="muted">You can leave this page open. Processing and GPU queues may take a few minutes.</p></>}
  </section>;
}

function Results({ job }: { job: Job }) {
  const [preview, setPreview] = useState<"score" | "tab">("score");
  const result = job.result!;
  const { metrics, artifacts } = result;
  const labels: Record<string, string> = { chords: "Chord timeline", events: "Note events", report: "Transcription report", pdf: "PDF score" };
  const extraArtifacts = Object.entries(artifacts).filter(([name]) => !["source", "musicxml", "tab"].includes(name));
  return <section className="result-section" aria-label="Transcription result">
    <div className="result-heading">
      <div><p className="eyebrow"><Icon name="check" />Your score is ready</p><h2>{job.title}</h2><p className="muted">{metrics.note_count} notes · {metrics.chord_count} chord labels · {Math.round(metrics.bpm)} BPM · {formatDuration(metrics.duration_seconds)}</p></div>
      {artifacts.musicxml && <a className="button primary" href={artifacts.musicxml.url}><Icon name="download" />Download score</a>}
    </div>
    <div className="panel preview-panel">
      <div className="preview-toolbar"><div className="segmented" role="group" aria-label="Preview format"><button type="button" aria-pressed={preview === "score"} onClick={() => setPreview("score")}>Sheet music</button><button type="button" aria-pressed={preview === "tab"} onClick={() => setPreview("tab")}>Guitar TAB</button></div>{artifacts.tab && <a className="text-link" href={artifacts.tab.url}>Download TAB</a>}</div>
      {preview === "score" && artifacts.musicxml ? <ScorePreview url={artifacts.musicxml.url} /> : <pre className="tab-preview">{result.tab_preview || "TAB preview unavailable."}</pre>}
      <p className="draft-note">AI-generated draft. Check the notes and chords by ear before using.</p>
    </div>
    {artifacts.source && <details className="disclosure panel"><summary>Listen to your recording</summary><audio aria-label="Original recording" controls preload="none" src={artifacts.source.url} /></details>}
    <details className="disclosure panel"><summary>Chords & transcription details</summary><div className="detail-content">
      <h3>Chord timeline</h3><div className="chords">{result.chords.length ? result.chords.map((chord, index) => <div key={`${chord.start}-${index}`}><strong>{chord.shape_chord ?? chord.symbol ?? "?"}</strong><span>{Number(chord.start ?? 0).toFixed(1)}s</span></div>) : <p className="muted">No chords identified in this draft.</p>}</div>
      <h3>Automatic checks</h3><p className="muted">Capo: {metrics.selected_capo === 0 ? "none" : metrics.selected_capo}{job.options.capo === "auto" ? " (estimated)" : ""} · Notes assigned to TAB: {Math.round(metrics.assigned_ratio * 100)}%</p><p className="muted">Playability signal: {Math.round(metrics.heuristic_health * 100)}%. This is a heuristic, not measured transcription accuracy.</p>
      {result.issues.length ? <ul className="issue-list">{result.issues.map((issue, index) => <li key={`${index}-${issue}`}>{issue}</li>)}</ul> : <p className="muted">No automatic quality warnings.</p>}
      <details className="processing-log"><summary>View processing log</summary><ol>{result.messages.map((message, index) => <li key={index}>{message}</li>)}</ol></details>
    </div></details>
    {extraArtifacts.length > 0 && <details className="disclosure panel"><summary>More downloads</summary><div className="download-list">{extraArtifacts.map(([name, artifact]) => <a href={artifact.url} key={name}><Icon name="download" /><span>{labels[name] ?? name}</span></a>)}</div></details>}
  </section>;
}

export default function App() {
  const [audio, setAudio] = useState<File | null>(null);
  const [tabHint, setTabHint] = useState<File | null>(null);
  const [title, setTitle] = useState("Fretwork Transcription");
  const [backend, setBackend] = useState("ensemble");
  const [capo, setCapo] = useState("auto");
  const [maxSpan, setMaxSpan] = useState(4);
  const [maxAttempts, setMaxAttempts] = useState(3);
  const [renderPdf, setRenderPdf] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [history, setHistory] = useState<Job[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const hintInput = useRef<HTMLInputElement>(null);
  const resultTarget = useRef<HTMLDivElement>(null);
  const busy = submitting || Boolean(job && ["queued", "processing"].includes(job.status));

  const loadHistory = async () => {
    try { setHistory(await api.jobs()); } catch { /* A submission displays connection failures. */ }
  };
  useEffect(() => { void loadHistory(); void api.runtime().then(setRuntime).catch(() => undefined); }, []);
  useEffect(() => {
    if (job) resultTarget.current?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  }, [job?.id]);
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const updated = await api.job(job.id); setJob(updated);
        if (["complete", "failed"].includes(updated.status)) void loadHistory();
      } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not refresh the job"); }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  const chooseAudio = (file: File | undefined) => {
    if (!file || busy) return;
    if (!/\.(wav|mp3|m4a|flac|ogg)$/i.test(file.name)) { setError("Choose a WAV, MP3, M4A, FLAC, or OGG recording."); return; }
    if (!file.size) { setError("This recording is empty. Choose another file."); return; }
    if (runtime && file.size > runtime.max_upload_mb * 1024 ** 2) { setError(`Choose a recording smaller than ${runtime.max_upload_mb} MB.`); return; }
    if (title === "Fretwork Transcription" || title === audio?.name.replace(/\.[^.]+$/, "").slice(0, 160)) setTitle(file.name.replace(/\.[^.]+$/, "").slice(0, 160));
    setAudio(file); setError(null);
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (busy) return;
    if (!audio) { setError("Choose a solo-guitar recording first."); return; }
    setSubmitting(true); setError(null);
    try { const created = await api.submit({ audio, tabHint, title: title.trim() || "Fretwork Transcription", backend, capo, maxSpan, maxAttempts, renderPdf }); setJob(created); void loadHistory(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Submission failed"); }
    finally { setSubmitting(false); }
  };
  const remove = async (item: Job) => {
    if (!window.confirm(`Delete “${item.title}” and its generated files?`)) return;
    setDeleting(item.id);
    try { await api.remove(item.id); if (job?.id === item.id) setJob(null); await loadHistory(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Could not delete the draft"); }
    finally { setDeleting(null); }
  };

  return <div className="app">
    <a className="skip-link" href="#create">Skip to transcription</a>
    <header className="topbar"><a className="brand" href="#top"><span>F</span>Fretwork</a><a className="text-link" href="https://github.com/ThanhMinh12/Fretwork" target="_blank" rel="noreferrer">GitHub ↗</a></header>
    <main id="top">
      <section className="intro"><h1>Guitar audio to sheet music.</h1><p>Upload a solo-guitar recording. Get an editable score and TAB.</p></section>
      {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" onClick={() => setError(null)} aria-label="Dismiss error">×</button></div>}
      <section className="create-section panel" id="create" aria-label="New transcription"><form onSubmit={(event) => void submit(event)}>
        <button className={`dropzone ${audio ? "selected" : ""} ${dragging ? "dragging" : ""}`} type="button" disabled={busy} onClick={() => audioInput.current?.click()} onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseAudio(event.dataTransfer.files[0]); }}>
          <span className="upload-icon"><Icon name={audio ? "check" : "upload"} /></span><strong>{audio ? audio.name : "Choose a guitar recording"}</strong><span>{audio ? `${(audio.size / 1024 ** 2).toFixed(1)} MB · Click to change` : "or drag and drop it here"}</span><small>WAV, MP3, M4A, FLAC, OGG{runtime ? ` · Up to ${runtime.max_audio_seconds} seconds / ${runtime.max_upload_mb} MB` : ""}</small>
        </button>
        <input ref={audioInput} className="sr-only" aria-label="Upload guitar recording" type="file" accept={audioAccept} disabled={busy} onChange={(event) => { chooseAudio(event.target.files?.[0]); event.target.value = ""; }} />
        <details className="advanced"><summary>Advanced options</summary><fieldset className="fields" disabled={busy}>
          <label><span>Score title</span><input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={160} /></label>
          <label><span>Transcription model</span><select value={backend} onChange={(event) => setBackend(event.target.value)}><option value="ensemble">Ensemble (recommended)</option><option value="gaps">Guitar model only</option><option value="basic_pitch">Basic Pitch only</option></select></label>
          <label><span>Capo</span><select value={capo} onChange={(event) => setCapo(event.target.value)}><option value="auto">Detect automatically</option><option value="0">No capo</option>{Array.from({ length: 12 }, (_, i) => <option key={i + 1} value={i + 1}>Capo {i + 1}</option>)}</select></label>
          <label><span>Maximum stretch: {maxSpan} frets</span><input type="range" min="2" max="7" value={maxSpan} onChange={(event) => setMaxSpan(Number(event.target.value))} /></label>
          <label><span>Review passes</span><select value={maxAttempts} onChange={(event) => setMaxAttempts(Number(event.target.value))}><option value="1">1 — faster</option><option value="2">2 — balanced</option><option value="3">3 — thorough</option></select></label>
          <div className="hint-field"><span>TAB hint (optional)</span><button className="button secondary" type="button" onClick={() => hintInput.current?.click()}>{tabHint?.name ?? "Choose .txt or .tab"}</button>{tabHint && <button className="text-link" type="button" onClick={() => setTabHint(null)}>Remove hint</button>}<input ref={hintInput} className="sr-only" aria-label="Upload TAB hint" type="file" accept=".txt,.tab" onChange={(event) => { setTabHint(event.target.files?.[0] ?? null); event.target.value = ""; }} /></div>
          <label className="checkbox"><input type="checkbox" checked={renderPdf} onChange={(event) => setRenderPdf(event.target.checked)} /><span>Also create a PDF (requires LilyPond on the server)</span></label>
        </fieldset></details>
        <button className="button primary submit" disabled={!audio || busy}>{submitting ? "Uploading…" : busy ? "Transcribing…" : "Create sheet music"}</button>
        {runtime?.queued_inference && <p className="demo-note">Free demo · GPU queues and daily limits apply. Files expire after an hour. Use non-sensitive audio.</p>}
      </form></section>
      <div ref={resultTarget} id="result" className="result-target">{job && (job.status === "complete" && job.result ? <Results key={job.id} job={job} /> : <Status job={job} />)}</div>
      {history.length > 0 && <details className="history disclosure panel"><summary>Recent drafts <span className="count">{history.length}</span></summary><div className="history-list">{history.map((item) => <div className="history-row" key={item.id}><button className="history-open" onClick={() => { setJob(item); setError(null); }}><Icon name={item.status === "complete" ? "check" : "clock"} /><span><strong>{item.title}</strong><small>{new Date(item.created_at).toLocaleDateString()} · {item.status === "complete" ? "Ready" : item.status}</small></span></button><button className="text-link remove" disabled={["queued", "processing"].includes(item.status) || deleting === item.id} aria-label={`Delete ${item.title}`} onClick={() => void remove(item)}>{deleting === item.id ? "Deleting…" : "Delete"}</button></div>)}</div></details>}
    </main>
    <footer><span>Fretwork · Solo-guitar transcription</span><a href="/api/docs">API docs ↗</a></footer>
  </div>;
}
