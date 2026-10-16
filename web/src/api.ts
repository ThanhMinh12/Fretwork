import type { Job } from "./types";

async function parse<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  let message = `Request failed (${response.status})`;
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) message = body.detail;
  } catch { /* Keep fallback. */ }
  throw new Error(message);
}

export interface Submission {
  audio: File;
  tabHint: File | null;
  title: string;
  backend: string;
  capo: string;
  maxSpan: number;
  maxAttempts: number;
  renderPdf: boolean;
}

export const api = {
  jobs: () => fetch("/api/jobs").then((response) => parse<Job[]>(response)),
  job: (id: string) => fetch(`/api/jobs/${id}`).then((response) => parse<Job>(response)),
  async submit(values: Submission): Promise<Job> {
    const form = new FormData();
    form.append("audio", values.audio);
    if (values.tabHint) form.append("tab_hint", values.tabHint);
    form.append("title", values.title);
    form.append("backend", values.backend);
    form.append("capo", values.capo);
    form.append("max_span", String(values.maxSpan));
    form.append("max_attempts", String(values.maxAttempts));
    form.append("render_pdf", String(values.renderPdf));
    return fetch("/api/jobs", { method: "POST", body: form }).then((response) => parse<Job>(response));
  },
  async remove(id: string): Promise<void> {
    const response = await fetch(`/api/jobs/${id}`, { method: "DELETE" });
    if (!response.ok) await parse(response);
  },
};

