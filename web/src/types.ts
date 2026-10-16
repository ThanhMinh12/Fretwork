export interface Artifact { filename: string; url: string }

export interface Result {
  metrics: {
    heuristic_health: number;
    selected_capo: number;
    assigned_ratio: number;
    note_count: number;
    chord_count: number;
    bpm: number;
    duration_seconds: number;
  };
  profile: string;
  backend: string;
  issues: string[];
  messages: string[];
  tab_preview: string;
  chords: Array<{ start?: number; end?: number; shape_chord?: string; symbol?: string; score?: number }>;
  artifacts: Record<string, Artifact>;
}

export interface Job {
  id: string;
  title: string;
  source_filename: string;
  status: "queued" | "processing" | "complete" | "failed";
  progress: number;
  phase: string;
  error: string | null;
  created_at: string;
  updated_at: string;
  options: {
    backend: "ensemble" | "gaps" | "basic_pitch";
    capo: "auto" | number;
    max_span: number;
    max_attempts: number;
    render_pdf: boolean;
  };
  result: Result | null;
}

