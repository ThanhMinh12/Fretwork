import type { Job } from "./types";
import { runDispatch, type Dispatch } from "./dispatch";

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

export interface Runtime {
  queued_inference: boolean;
  max_upload_mb: number;
  max_audio_seconds: number;
  temporary: boolean;
}

let runtimePromise: Promise<Runtime> | undefined;
function runtime() {
  return runtimePromise ??= fetch('/api/runtime', {cache:'no-store'}).then(r => parse<Runtime>(r));
}
async function request(path: string, options?: RequestInit) {
  await runtime(); // Establish exactly one signed session before parallel calls.
  return fetch(path, options);
}

export const api = {
  runtime,
  jobs: () => request("/api/jobs").then((response) => parse<Job[]>(response)),
  job: (id: string) => request(`/api/jobs/${id}`).then((response) => parse<Job>(response)),
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
    const job = await request("/api/jobs", { method: "POST", body: form }).then((response) => parse<Job & { dispatch?: Dispatch }>(response));
    if (job.dispatch) void runDispatch(job.dispatch).catch(() => undefined);
    return job;
  },
  async remove(id: string): Promise<void> {
    const response = await request(`/api/jobs/${id}`, { method: "DELETE" });
    if (!response.ok) await parse(response);
  },
};

