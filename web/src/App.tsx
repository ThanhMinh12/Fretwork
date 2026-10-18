import { FormEvent, useEffect, useRef, useState } from "react";
import { api, type Runtime } from "./api";
import type { Job } from "./types";
import ScorePreview from "./ScorePreview";
import WelcomeStory from "./WelcomeStory";

const audioAccept = ".wav,.mp3,.m4a,.flac,.ogg";

function Icon({ name }: { name: "upload" | "download" | "check" | "clock" | "plus" | "trash" }) {
  const paths = {
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M4 17v3h16v-3" /></>,
    download: <><path d="M12 4v12m0 0 5-5m-5 5-5-5" /><path d="M4 20h16" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    trash: <><path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7" /></>,
  };
  return <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function Status({ job, onRetry }: { job: Job; onRetry: () => void }) {
  const failed = job.status === "failed";
  const progress = Math.min(100, Math.max(0, Math.round(job.progress * 100)));
  return <section className={`run-status ${failed ? "failed" : ""}`} aria-live="polite">
    <span className="status-icon"><Icon name="clock" /></span>
    <h2>{failed ? "Couldn’t create this score" : job.status === "queued" ? "Your recording is queued" : "Creating your sheet music"}</h2>
    <p>{failed ? job.error ?? "Try again with a clean solo-guitar recording." : job.status === "queued" ? "Waiting for a worker." : "Listening to your recording and writing the score."}</p>
    {failed ? <button className="button primary" onClick={onRetry}>Try again</button> : <>
      <div className="progress-label"><span>Transcription progress</span><span>{progress}%</span></div>
      <div className="progress" role="progressbar" aria-label="Transcription progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div>
      <p className="muted">Keep this page open. Processing and GPU queues may take a few minutes.</p>
    </>}
  </section>;
}

function Results({ job }: { job: Job }) {
  const [preview, setPreview] = useState<"score" | "tab" | "audio" | "details">("score");
  const result = job.result!;
  const { metrics, artifacts } = result;
  const labels: Record<string, string> = { tab: "Guitar TAB (.txt)", chords: "Chord timeline", events: "Note events", report: "Transcription report", pdf: "PDF score" };
  const extraArtifacts = Object.entries(artifacts).filter(([name]) => !["source", "musicxml"].includes(name));

  return <section className="result-section" aria-label="Transcription result">
    <div className="workspace-heading">
      <div><h1>{job.title}</h1><p className="muted">Your score is ready.</p></div>
      <div className="download-actions">
        {artifacts.musicxml && <a className="button primary" href={artifacts.musicxml.url}><Icon name="download" />Download score</a>}
        {extraArtifacts.length > 0 && <details className="download-menu" onBlur={event => { if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) event.currentTarget.open = false; }} onKeyDown={event => { if (event.key === "Escape") { event.currentTarget.open = false; event.currentTarget.querySelector("summary")?.focus(); } }}><summary>More formats</summary><div className="download-list">{extraArtifacts.map(([name, artifact]) => <a href={artifact.url} key={name}>{labels[name] ?? name}</a>)}</div></details>}
      </div>
    </div>
    <div className="preview-tabs" role="group" aria-label="Preview format">
      <button type="button" aria-pressed={preview === "score"} onClick={() => setPreview("score")}>Sheet music</button>
      <button type="button" aria-pressed={preview === "tab"} onClick={() => setPreview("tab")}>Guitar TAB</button>
      {artifacts.source && <button type="button" aria-pressed={preview === "audio"} onClick={() => setPreview("audio")}>Audio</button>}
      <button type="button" aria-pressed={preview === "details"} onClick={() => setPreview("details")}>Details</button>
    </div>
    <div className="preview-surface">
      {preview === "score" && (artifacts.musicxml ? <ScorePreview url={artifacts.musicxml.url} /> : <p className="empty-preview">No sheet music is available for this draft.</p>)}
      {preview === "tab" && <pre className="tab-preview">{result.tab_preview || "TAB preview unavailable."}</pre>}
      {preview === "audio" && artifacts.source && <div className="audio-preview"><h2>Original recording</h2><audio aria-label="Original recording" controls preload="none" src={artifacts.source.url} /><p className="muted">{formatDuration(metrics.duration_seconds)}</p></div>}
      {preview === "details" && <div className="detail-content">
        <p className="muted">{metrics.note_count} notes · {metrics.chord_count} chord labels · {Math.round(metrics.bpm)} BPM</p>
        <h2>Chord timeline</h2><div className="chords">{result.chords.length ? result.chords.map((chord, index) => <div key={`${chord.start}-${index}`}><strong>{chord.shape_chord ?? chord.symbol ?? "?"}</strong><span>{Number(chord.start ?? 0).toFixed(1)}s</span></div>) : <p className="muted">No chords identified in this draft.</p>}</div>
        <h2>Automatic checks</h2><p className="muted">Capo: {metrics.selected_capo === 0 ? "none" : metrics.selected_capo}{job.options.capo === "auto" ? " (estimated)" : ""} · Notes assigned to TAB: {Math.round(metrics.assigned_ratio * 100)}%</p><p className="muted">Playability signal: {Math.round(metrics.heuristic_health * 100)}%. This is a heuristic, not measured transcription accuracy.</p>
        {result.issues.length ? <ul className="issue-list">{result.issues.map((issue, index) => <li key={`${index}-${issue}`}>{issue}</li>)}</ul> : <p className="muted">No automatic quality warnings.</p>}
        <details className="processing-log"><summary>Processing log</summary><ol>{result.messages.map((message, index) => <li key={index}>{message}</li>)}</ol></details>
      </div>}
    </div>
    <p className="draft-note">AI-generated draft. Review the notes and chords by ear.</p>
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
  const [view, setView] = useState<"new" | "job">("new");
  const [navOpen, setNavOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const hintInput = useRef<HTMLInputElement>(null);
  const workspace = useRef<HTMLElement>(null);
  const busy = submitting || Boolean(job && ["queued", "processing"].includes(job.status));
  const welcome = view === "new" && !audio && !navOpen;

  const loadHistory = async () => {
    try { setHistory(await api.jobs()); } catch { /* Submissions display connection failures. */ }
  };
  useEffect(() => { void loadHistory(); void api.runtime().then(setRuntime).catch(() => undefined); }, []);
  useEffect(() => {
    if (view === "job") { workspace.current?.focus({ preventScroll: true }); window.scrollTo(0, 0); }
  }, [view, job?.id]);
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
    if (busy || !audio) return;
    setSubmitting(true); setError(null);
    try {
      const created = await api.submit({ audio, tabHint, title: title.trim() || "Fretwork Transcription", backend, capo, maxSpan, maxAttempts, renderPdf });
      setJob(created); setView("job"); setNavOpen(false); void loadHistory();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Submission failed"); }
    finally { setSubmitting(false); }
  };
  const remove = async (item: Job) => {
    if (!window.confirm(`Delete “${item.title}” and its generated files?`)) return;
    setDeleting(item.id);
    try { await api.remove(item.id); if (job?.id === item.id) { setJob(null); setView("new"); } setNavOpen(false); await loadHistory(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Could not delete the draft"); }
    finally { setDeleting(null); }
  };

  return <div className="app">
    <a className="skip-link" href="#workspace">Skip to workspace</a>
    <header className="topbar"><a className="brand" href="#workspace" aria-label="Fretwork workspace"><img className="brand-logo" src="/brandmark.svg" width="36" height="36" alt="" /><span>Fretwork</span></a><div className="top-actions">
      {history.length > 0 && <button className="quiet-button" aria-label="Recent scores" aria-expanded={navOpen} onClick={() => setNavOpen(!navOpen)}>Scores</button>}
      {view === "job" && <button className="button secondary" aria-label="New transcription" disabled={busy} onClick={() => { setView("new"); setAudio(null); setTitle("Fretwork Transcription"); setNavOpen(false); setError(null); }}><Icon name="plus" /><span className="desktop-label">New transcription</span><span className="mobile-label">New</span></button>}
      <details className="help-menu"><summary>Help</summary><div><p>Choose a clean solo-guitar recording, then create your score. Model and capo settings are optional.</p>{runtime && <p>WAV, MP3, M4A, FLAC, OGG · Up to {runtime.max_audio_seconds} seconds / {runtime.max_upload_mb} MB.</p>}{runtime?.queued_inference && <p>Free GPU demo. Queues and daily limits apply. Files expire after an hour; use non-sensitive audio.</p>}<a href="https://github.com/ThanhMinh12/Fretwork" target="_blank" rel="noreferrer">Source code ↗</a><a href="/api/docs">API documentation ↗</a></div></details>
    </div></header>
    <div className={`workspace-layout ${navOpen ? "with-sidebar" : "single-pane"}`}>
      {navOpen && <aside className="sidebar is-open" aria-label="Score navigation">
        <h2 className="sidebar-title">Your scores <span>{history.length}</span></h2>
        <button className="sidebar-toggle" aria-expanded={navOpen} aria-controls="score-list" onClick={() => setNavOpen(!navOpen)}>Your scores <span>{history.length}</span><b>{navOpen ? "−" : "+"}</b></button>
        <div className="sidebar-content" id="score-list">
          <div className="history-list">{history.length ? history.map((item) => {
            const displayed = job?.id === item.id ? job : item;
            return <div className={`history-row ${view === "job" && job?.id === item.id ? "active" : ""}`} key={item.id}>
              <button className="history-open" aria-current={view === "job" && job?.id === item.id ? "page" : undefined} title={item.title} onClick={() => { setJob(displayed); setView("job"); setNavOpen(false); setError(null); }}><Icon name={displayed.status === "complete" ? "check" : "clock"} /><span><strong>{item.title}</strong><small>{displayed.status === "complete" ? "Ready" : displayed.status}</small></span></button>
              <button className="icon-button remove" disabled={["queued", "processing"].includes(displayed.status) || deleting === item.id} aria-label={`Delete ${item.title}`} title="Delete score" onClick={() => void remove(item)}><Icon name="trash" /></button>
            </div>;
          }) : <p className="sidebar-empty">Your transcriptions will appear here.</p>}</div>
          <div className="sidebar-footer"><p>Scores are private to this browser.</p><a href="/api/docs">API documentation ↗</a></div>
        </div>
      </aside>}
      <main id="workspace" className={welcome ? "welcome-main" : undefined} ref={workspace} tabIndex={-1}>
        <div className={`workspace-inner ${view === "new" ? "upload-workspace" : ""} ${welcome ? "welcome-workspace" : ""}`}>
          {error && <div className="error-banner" role="alert"><span>{error}</span><button type="button" onClick={() => setError(null)} aria-label="Dismiss error">×</button></div>}
          {view === "new" ? <section className={`create-section ${welcome ? "welcome-layout" : ""}`} aria-label="New transcription">
            {welcome && <WelcomeStory />}
            <div className={welcome ? "welcome-action" : undefined}>
            <div className="workspace-heading"><div>{welcome ? <h2>Audio to sheet music</h2> : <h1>Audio to sheet music</h1>}<p className="muted">{audio ? "Your recording is ready to transcribe." : "Choose a solo-guitar recording to begin."}</p></div></div>
            <form onSubmit={(event) => void submit(event)}>
              <button className={`dropzone ${audio ? "selected" : ""} ${dragging ? "dragging" : ""}`} type="button" disabled={busy} onClick={() => audioInput.current?.click()} onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); chooseAudio(event.dataTransfer.files[0]); }}>
                <span className="upload-icon"><Icon name={audio ? "check" : "upload"} /></span><strong>{audio ? audio.name : "Choose a guitar recording"}</strong><span>{audio ? `${(audio.size / 1024 ** 2).toFixed(1)} MB · Click to change` : "or drag and drop an audio file"}</span>
              </button>
              <input ref={audioInput} className="sr-only" aria-label="Upload guitar recording" type="file" accept={audioAccept} disabled={busy} onChange={(event) => { chooseAudio(event.target.files?.[0]); event.target.value = ""; }} />
              <p className="file-help">{runtime ? `Up to ${runtime.max_audio_seconds} seconds / ${runtime.max_upload_mb} MB` : "Solo-guitar audio"}</p>
              {audio && <details className="advanced"><summary>Settings (optional)</summary><fieldset className="fields" disabled={busy}>
                <label><span>Score title</span><input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={160} /></label>
                <label><span>Transcription model</span><select value={backend} onChange={(event) => setBackend(event.target.value)}><option value="ensemble">Ensemble (recommended)</option><option value="gaps">Guitar model only</option><option value="basic_pitch">Basic Pitch only</option></select></label>
                <label><span>Capo</span><select value={capo} onChange={(event) => setCapo(event.target.value)}><option value="auto">Detect automatically</option><option value="0">No capo</option>{Array.from({ length: 12 }, (_, i) => <option key={i + 1} value={i + 1}>Capo {i + 1}</option>)}</select></label>
                <label><span>Maximum stretch: {maxSpan} frets</span><input type="range" min="2" max="7" value={maxSpan} onChange={(event) => setMaxSpan(Number(event.target.value))} /></label>
                <label><span>Review passes</span><select value={maxAttempts} onChange={(event) => setMaxAttempts(Number(event.target.value))}><option value="1">1 — faster</option><option value="2">2 — balanced</option><option value="3">3 — thorough</option></select></label>
                <div className="hint-field"><span>TAB hint (optional)</span><button className="button secondary" type="button" onClick={() => hintInput.current?.click()}>{tabHint?.name ?? "Choose .txt or .tab"}</button>{tabHint && <button className="text-link" type="button" onClick={() => setTabHint(null)}>Remove hint</button>}<input ref={hintInput} className="sr-only" aria-label="Upload TAB hint" type="file" accept=".txt,.tab" onChange={(event) => { setTabHint(event.target.files?.[0] ?? null); event.target.value = ""; }} /></div>
                <label className="checkbox"><input type="checkbox" checked={renderPdf} onChange={(event) => setRenderPdf(event.target.checked)} /><span>Also create a PDF (requires LilyPond on the server)</span></label>
              </fieldset></details>}
              {audio && <div className="form-actions"><button className="button primary" disabled={busy}>{submitting ? "Uploading…" : busy ? "Transcribing…" : "Create sheet music"}</button></div>}
            </form>
            {welcome && runtime?.queued_inference && <p className="privacy-note">Use non-sensitive audio. Files expire after an hour.</p>}
            </div>
          </section> : job && <div id="result">{job.status === "complete" && job.result ? <Results key={job.id} job={job} /> : <><div className="workspace-heading"><div><p className="eyebrow">Transcription</p><h1>{job.title}</h1></div></div><Status job={job} onRetry={() => { setView("new"); setError(null); }} /></>}</div>}
          {!welcome && runtime?.queued_inference && <p className="privacy-note">Use non-sensitive audio. Files expire after an hour.</p>}
        </div>
      </main>
    </div>
  </div>;
}
