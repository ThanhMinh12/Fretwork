import { useEffect, useRef, useState } from 'react';
import type { OpenSheetMusicDisplay } from 'opensheetmusicdisplay';

export default function ScorePreview({url}: {url: string}) {
  const target = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState('Loading sheet music…');
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    let display: OpenSheetMusicDisplay | undefined;
    void (async () => {
      try {
        const [module, response] = await Promise.all([
          import('opensheetmusicdisplay'), fetch(url, {signal:controller.signal, credentials:'same-origin'}),
        ]);
        if (!response.ok) throw new Error('Score download failed');
        const xml = await response.text();
        if (cancelled || !target.current) return;
        display = new module.OpenSheetMusicDisplay(target.current, {autoResize:true, backend:'svg', drawTitle:true, drawComposer:false});
        await display.load(xml);
        if (cancelled) return;
        display.render(); setStatus('');
      } catch {
        if (!cancelled) setStatus('Preview unavailable. You can still download the MusicXML score below.');
      }
    })();
    return () => {cancelled = true; controller.abort(); if (display) {display.AutoResizeEnabled = false; display.clear();}};
  }, [url]);
  return <article className="score-preview"><div className="card-heading"><div><span>DIGITAL SHEET MUSIC</span><h3>Notation preview</h3></div><a href={url}>Download MusicXML ↗</a></div>{status && <p role="status">{status}</p>}<div ref={target} className="score-preview-canvas" aria-label="Generated sheet music" /></article>;
}
