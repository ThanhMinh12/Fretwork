export default function WelcomeStory() {
  return <div className="welcome-story">
    <p className="welcome-eyebrow">From strings to score</p>
    <h1>Your music.<br /><em>On the page.</em></h1>
    <p className="welcome-description">Turn your guitar recordings into sheet music, chords, and TAB.</p>
    <svg className="welcome-art" viewBox="0 0 320 86" fill="none" aria-hidden="true">
      <path className="staff-lines" d="M0 20h320M0 31h320M0 42h320M0 53h320M0 64h320" />
      <g className="staff-notes"><path d="M69 50V12m65 27V8m0 0 49-7v32m62 17V12" /><ellipse cx="62" cy="51" rx="8" ry="5" transform="rotate(-20 62 51)" /><ellipse cx="127" cy="40" rx="8" ry="5" transform="rotate(-20 127 40)" /><ellipse cx="176" cy="34" rx="8" ry="5" transform="rotate(-20 176 34)" /><ellipse cx="238" cy="51" rx="8" ry="5" transform="rotate(-20 238 51)" /></g>
    </svg>
  </div>;
}
