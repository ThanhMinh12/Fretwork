import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Job } from "./types";
import ScorePreview from './ScorePreview';

const audioAccept = ".wav,.mp3,.m4a,.flac,.ogg";

function Icon({ name }: { name: "wave" | "upload" | "download" | "check" | "clock" | "spark" }) {
  const path = {
    wave: <path d="M3 12h2l2-7 3 14 3-11 2 8 2-4h4" />,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M4 17v3h16v-3"/></>,
    download: <><path d="M12 4v12m0 0 5-5m-5 5-5-5"/><path d="M4 20h16"/></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    spark: <path d="m12 2 1.7 5.3L19 9l-5.3 1.7L12 16l-1.7-5.3L5 9l5.3-1.7Z" />,
  };
  return <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">{path[name]}</svg>;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function Status({ job }: { job: Job }) {
  return (
    <section className="run-status" aria-live="polite">
      <div className={`status-orb ${job.status}`}><Icon name={job.status === "complete" ? "check" : "wave"} /></div>
      <div className="status-copy">
        <span>{job.status.toUpperCase()}</span>
        <h2>{job.phase}</h2>
        <p>{job.error ?? (job.status === "processing" ? "The guitar model and independent reviewer are reconciling the recording. CPU runs can take several minutes." : job.source_filename)}</p>
      </div>
      <strong>{Math.round(job.progress * 100)}%</strong>
      <div className="progress"><i style={{ width: `${job.progress * 100}%` }} /></div>
    </section>
  );
}

function Results({ job }: { job: Job }) {
  const result = job.result!;
  const metrics = result.metrics;
  const artifactLabels: Record<string, string> = {
    musicxml: "MusicXML score", tab: "ASCII TAB", chords: "Chord timeline",
    events: "Editable events", report: "Agent report", pdf: "PDF score",
  };
  return (
    <section className="result-section">
      <div className="result-title"><div><span>TRANSCRIPTION COMPLETE</span><h2>{job.title}</h2></div><a className="primary-download" href={result.artifacts.musicxml?.url}><Icon name="download" />Open score</a></div>
      <div className="metrics">
        <div><span>HEURISTIC HEALTH</span><strong>{Math.round(metrics.heuristic_health * 100)}%</strong><small>playability signal</small></div>
        <div><span>SELECTED CAPO</span><strong>{metrics.selected_capo || "—"}</strong><small>{job.options.capo === "auto" ? "inferred candidate" : "provided"}</small></div>
        <div><span>NOTES ASSIGNED</span><strong>{Math.round(metrics.assigned_ratio * 100)}%</strong><small>{metrics.note_count} note events</small></div>
        <div><span>ESTIMATED TEMPO</span><strong>{Math.round(metrics.bpm)}</strong><small>BPM · {formatDuration(metrics.duration_seconds)}</small></div>
      </div>

      <div className="review-grid">
        <article className="tab-card">
          <div className="card-heading"><div><span>PLAYABLE DRAFT</span><h3>Chord-aware TAB</h3></div><a href={result.artifacts.tab?.url}>Download .txt ↗</a></div>
          <pre>{result.tab_preview || "TAB preview unavailable"}</pre>
        </article>
        <aside>
          <article className="audio-card">
            <span>REFERENCE RECORDING</span>
            <audio controls src={result.artifacts.source?.url} />
          </article>
          <article className="critic-card">
            <span>CRITIC NOTES</span>
            {result.issues.length ? <ul>{result.issues.map((issue) => <li key={issue}>{issue}</li>)}</ul> : <p><Icon name="check" />No automatic quality warnings</p>}
            <small>This health score is not measured transcription accuracy. Review the draft by ear.</small>
          </article>
        </aside>
      </div>

      {result.artifacts.musicxml && <ScorePreview url={result.artifacts.musicxml.url} />}

      <article className="chord-card">
        <div className="card-heading"><div><span>HARMONY RAG</span><h3>Retrieved chord timeline</h3></div><b>{metrics.chord_count} labels</b></div>
        <div className="chords">{result.chords.length ? result.chords.slice(0, 16).map((chord, index) => <div key={`${chord.start}-${index}`}><strong>{chord.shape_chord ?? chord.symbol ?? "?"}</strong><span>{Number(chord.start ?? 0).toFixed(1)}s</span></div>) : <p>No chord labels in this draft.</p>}</div>
      </article>

      <div className="bottom-grid">
        <article className="trace-card"><div className="card-heading"><div><span>AGENT TRACE</span><h3>How the draft was made</h3></div><b>{result.profile}</b></div><ol>{result.messages.map((message, index) => <li key={`${index}-${message}`}>{message}</li>)}</ol></article>
        <article className="downloads-card"><span>ALL ARTIFACTS</span>{Object.entries(result.artifacts).filter(([name]) => name !== "source").map(([name, artifact]) => <a href={artifact.url} key={name}><div><Icon name="download" /><span><strong>{artifactLabels[name] ?? name}</strong><small>{artifact.filename}</small></span></div><b>↗</b></a>)}</article>
      </div>
    </section>
  );
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
  const [error, setError] = useState<string | null>(null);
  const [hosted, setHosted] = useState(false);
  const audioInput = useRef<HTMLInputElement>(null);
  const hintInput = useRef<HTMLInputElement>(null);

  const loadHistory = async () => {
    try { setHistory(await api.jobs()); } catch { /* Health is also reflected by submissions. */ }
  };

  useEffect(() => { void loadHistory(); }, []);
  useEffect(() => { void api.runtime().then(r => setHosted(Boolean(r.queued_inference))).catch(() => undefined); }, []);
  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const updated = await api.job(job.id);
        setJob(updated);
        if (["complete", "failed"].includes(updated.status)) void loadHistory();
      } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not refresh the job"); }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!audio) { setError("Choose a solo-guitar recording first."); return; }
    setSubmitting(true); setError(null);
    try {
      const created = await api.submit({ audio, tabHint, title, backend, capo, maxSpan, maxAttempts, renderPdf });
      setJob(created); void loadHistory();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Submission failed"); }
    finally { setSubmitting(false); }
  };

  return (
    <div className="app">
      <header><a className="brand" href="#top"><span>F</span>Fretwork</a><nav><a href="#create">Transcribe</a><a href="#process">Process</a><a href="/api/docs">API</a></nav><p><i />{hosted ? 'FREE GPU DEMO' : 'LOCAL PROCESSING'}</p></header>
      <main id="top">
        {hosted && <p style={{padding:'16px 0',fontSize:13}}>Free hosted demo · 30 seconds / 16 MB per recording. GPU queues and daily limits apply. Use non-sensitive audio; drafts expire after an hour.</p>}
        <section className="hero">
          <div className="hero-copy"><p className="eyebrow">OPEN MODELS · SELF-HOSTED · NO THIRD-PARTY AI API</p><h1>Audio in.<br/><em>Fingerstyle</em> out.</h1><p>Turn a solo-guitar recording into an editable score, playable TAB, and grounded chord timeline—then inspect every decision the specialist agents made.</p><a href="#create">Start a transcription <span>↓</span></a></div>
          <div className="score-art" aria-hidden="true"><span className="clef">𝄞</span><div className="staff">{Array.from({length: 5}).map((_, i) => <i key={i} />)}<b className="n1">♪</b><b className="n2">♩</b><b className="n3">♫</b></div><div className="fretboard">{Array.from({length: 6}).map((_, i) => <i key={i} />)}<b /><b /><b /><b /></div></div>
        </section>

        <section className="create-section" id="create">
          <div className="section-number">01</div>
          <form onSubmit={(event) => void submit(event)}>
            <div className="section-heading"><div><span>NEW TRANSCRIPTION</span><h2>Recordings become<br/>working scores.</h2></div><p>Best with a clean solo fingerstyle performance. WAV, MP3, M4A, FLAC, and OGG are supported.</p></div>
            <button className={`dropzone ${audio ? "selected" : ""}`} type="button" onClick={() => audioInput.current?.click()}><Icon name={audio ? "check" : "upload"}/><div><strong>{audio ? audio.name : "Choose a guitar recording"}</strong><span>{audio ? `${(audio.size / 1024 / 1024).toFixed(1)} MB · ready to upload` : "Click to browse · up to 200 MB"}</span></div><b>{audio ? "CHANGE" : "BROWSE"}</b></button>
            <input ref={audioInput} className="sr-only" type="file" accept={audioAccept} onChange={(event) => { const file = event.target.files?.[0] ?? null; setAudio(file); if (file && title === "Fretwork Transcription") setTitle(file.name.replace(/\.[^.]+$/, "")); }} />
            <div className="fields">
              <label><span>SCORE TITLE</span><input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={160}/></label>
              <label><span>MODEL ROUTE</span><select value={backend} onChange={(event) => setBackend(event.target.value)}><option value="ensemble">Ensemble · CRNN + independent review</option><option value="gaps">Guitar CRNN only</option><option value="basic_pitch">Fast draft · Basic Pitch</option></select></label>
              <label><span>CAPO</span><select value={capo} onChange={(event) => setCapo(event.target.value)}><option value="auto">Auto · rank 0–7</option><option value="0">No capo</option>{Array.from({length: 12}, (_, i) => <option key={i + 1} value={i + 1}>Capo {i + 1}</option>)}</select></label>
              <label><span>MAXIMUM STRETCH · {maxSpan} FRETS</span><input type="range" min="2" max="7" value={maxSpan} onChange={(event) => setMaxSpan(Number(event.target.value))}/></label>
              <label><span>CRITIC ATTEMPTS</span><select value={maxAttempts} onChange={(event) => setMaxAttempts(Number(event.target.value))}><option value="1">1 · fastest</option><option value="2">2 · balanced</option><option value="3">3 · thorough</option></select></label>
              <label className="hint"><span>ROUGH TAB · OPTIONAL</span><button type="button" onClick={() => hintInput.current?.click()}>{tabHint?.name ?? "Add .txt or .tab hint"}</button><input ref={hintInput} className="sr-only" type="file" accept=".txt,.tab" onChange={(event) => setTabHint(event.target.files?.[0] ?? null)}/></label>
            </div>
            <label className="check"><input type="checkbox" checked={renderPdf} onChange={(event) => setRenderPdf(event.target.checked)}/><span>Render PDF when LilyPond is installed</span></label>
            {error && <p className="error" role="alert">{error}<button type="button" onClick={() => setError(null)}>×</button></p>}
            <button className="submit" disabled={submitting}><Icon name="spark"/>{submitting ? "Uploading recording…" : "Run transcription agents"}<span>↗</span></button>
          </form>
        </section>

        {job && <div id="result"><Status job={job}/>{job.status === "complete" && job.result && <Results job={job}/>}</div>}

        <section className="process" id="process"><div className="section-number">02</div><div className="section-heading"><div><span>THE PIPELINE</span><h2>Nine focused agents.<br/>One playable draft.</h2></div><p>The guitar model proposes. A separate model reviews. The critic can route one bounded corrective pass before publishing.</p></div><div className="process-grid">{[["01","Listen","Timing + evidence"],["02","Propose","Guitar CRNN"],["03","Review","Basic Pitch"],["04","Reconcile","Acoustic consensus"],["05","Arrange","Voices + fingering"],["06","Publish","Score + TAB"]].map(([number,title,copy]) => <div key={number}><span>{number}</span><Icon name="wave"/><h3>{title}</h3><p>{copy}</p></div>)}</div></section>

        {history.length > 0 && <section className="history"><div className="section-heading"><div><span>RECENT RUNS</span><h2>Your drafts</h2></div></div>{history.map((item) => <div className="history-row" key={item.id}><button className="history-open" onClick={() => setJob(item)}><Icon name={item.status === "complete" ? "check" : "clock"}/><span><strong>{item.title}</strong><small>{new Date(item.created_at).toLocaleString()} · {item.options.backend}</small></span><b>{item.status}</b></button><button className="delete-run" disabled={["queued","processing"].includes(item.status)} aria-label={`Delete ${item.title}`} onClick={() => {if (window.confirm(`Permanently delete “${item.title}” and its generated files?`)) void (async () => {try {await api.remove(item.id); if (job?.id === item.id) setJob(null); await loadHistory();} catch (caught) {setError(caught instanceof Error ? caught.message : 'Could not delete draft');}})();}}>Delete</button></div>)}</section>}
      </main>
      <footer><a className="brand" href="#top"><span>F</span>Fretwork</a><p>Local-first guitar transcription for drafts you can inspect and edit.</p><a href="/api/docs">REST API ↗</a></footer>
    </div>
  );
}

