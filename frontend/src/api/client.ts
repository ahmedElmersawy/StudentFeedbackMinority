/**
 * Feedback Atlas — API client (v3 dual-mode).
 * All calls go to the FastAPI backend at VITE_API_URL (default: http://localhost:8000).
 */

const BASE = import.meta.env.VITE_API_URL ?? "";

export type FeedbackMode = "student_to_student" | "student_to_professor";

export interface PredictResponse {
  text: string;
  prediction: string;
  confidence: number;
  needs_review: boolean;
  broad_sentiment: string;
  priority_score: number;
}

export interface SummaryStats {
  total: number;
  feedback_mode: FeedbackMode | string;
  minority: number;
  minority_pct: number;
  needs_review: number;
  review_pct: number;
  mismatch: number;
  mismatch_pct: number;
  avg_confidence: number;
  avg_priority_score: number;
  label_distribution: Record<string, number>;
  minority_category_breakdown: Record<string, number>;
  catme_subtype_distribution: Record<string, number>;
  success_criteria: Record<string, boolean | number | string> | null;
  split_files: string[];
  rows_truncated?: boolean;
}

export interface ResultRow {
  text?: string;
  feedback?: string;
  prediction?: string;
  confidence?: number;
  needs_review?: boolean | number;
  is_minority_pattern?: boolean | number;
  is_outlier?: boolean | number;
  is_minority_cluster?: boolean | number;
  minority_category?: string;
  keyword_minority?: boolean | number;
  keyword_categories?: string;
  cluster_id?: number;
  mismatch_flag?: boolean | number;
  mismatch_type?: string;
  catme_subtype?: "peer_feedback" | "self_assessment";
  feedback_mode?: string;
  priority_score?: number;
  [key: string]: unknown;
}

export interface AnalysisResult {
  summary: SummaryStats;
  rows: ResultRow[];
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  message: string;
  rows_processed: number;
  total_rows: number;
  feedback_mode?: FeedbackMode | string | null;
}

export interface MinorityResult {
  total: number;
  flagged: number;
  pct: number;
  feedback_mode: string;
  rows: ResultRow[];
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Backend unreachable (${res.status})`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Single-text prediction
// ---------------------------------------------------------------------------

export async function predictText(
  text: string,
  options: { modelDir?: string; feedbackMode?: FeedbackMode } = {}
): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      model_dir: options.modelDir ?? null,
      feedback_mode: options.feedbackMode ?? null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Prediction failed");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Upload CSV — async job
// ---------------------------------------------------------------------------

export async function uploadCsv(
  file: File,
  options: {
    feedbackMode?: FeedbackMode;
    modelDir?: string;
    anonymize?: boolean;
    includeMinority?: boolean;
    includeMismatch?: boolean;
    runZeroShotCategorization?: boolean;
    confidenceThreshold?: number;
    generateSplitOutputs?: boolean;
    onUploadProgress?: (pct: number) => void;   // transfer progress 0-100
  } = {}
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  if (options.feedbackMode) form.append("feedback_mode", options.feedbackMode);
  if (options.modelDir) form.append("model_dir", options.modelDir);
  form.append("anonymize", String(options.anonymize ?? true));
  form.append("include_minority", String(options.includeMinority ?? true));
  form.append("include_mismatch", String(options.includeMismatch ?? true));
  form.append("run_zero_shot_categorization", String(options.runZeroShotCategorization ?? false));
  form.append("confidence_threshold", String(options.confidenceThreshold ?? 0.65));
  form.append("generate_split_outputs", String(options.generateSplitOutputs ?? false));

  // Use XHR instead of fetch so we get upload progress events
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/upload`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && options.onUploadProgress) {
        options.onUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { reject(new Error("Invalid server response")); }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail ?? "Upload failed"));
        } catch {
          reject(new Error(`Upload failed (${xhr.status})`));
        }
      }
    };

    xhr.onerror = () => reject(new Error("Network error — check your connection and try again"));
    xhr.ontimeout = () => reject(new Error("Upload timed out — try a smaller file or use the local URL"));
    xhr.timeout = 600_000;   // 10 min max for very large files

    xhr.send(form);
  });
}

// ---------------------------------------------------------------------------
// Poll job status
// ---------------------------------------------------------------------------

export async function pollJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/jobs/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Job not found");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Fetch job results (JSON)
// ---------------------------------------------------------------------------

export async function fetchResults(jobId: string): Promise<AnalysisResult> {
  const res = await fetch(`${BASE}/jobs/${jobId}/results?format=json`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Results not available");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Download results as CSV (full combined)
// ---------------------------------------------------------------------------

export function downloadResultsCsv(jobId: string, filename = "results.csv"): void {
  const a = document.createElement("a");
  a.href = `${BASE}/jobs/${jobId}/results?format=csv`;
  a.download = filename;
  a.click();
}

export function downloadMinorityCsv(jobId: string, filename = "minority_results.csv"): void {
  const a = document.createElement("a");
  a.href = `${BASE}/jobs/${jobId}/results/minority`;
  a.download = filename;
  a.click();
}

export interface LabelStat {
  label: string;
  count: number;
  pct: number;
  avg_confidence: number;
  minority_count: number;
}

export interface LabelsResult {
  job_id: string;
  total: number;
  labels: LabelStat[];
}

export async function fetchLabels(jobId: string): Promise<LabelsResult> {
  const res = await fetch(`${BASE}/jobs/${jobId}/labels`);
  if (!res.ok) throw new Error("Labels not available");
  return res.json();
}

export function downloadLabelCsv(jobId: string, label: string): void {
  const a = document.createElement("a");
  a.href = `${BASE}/jobs/${jobId}/results/label/${encodeURIComponent(label)}`;
  a.download = `${label}.csv`;
  a.click();
}

// ---------------------------------------------------------------------------
// List split output files for a job
// ---------------------------------------------------------------------------

export async function listSplitFiles(jobId: string): Promise<{
  job_id: string;
  feedback_mode: string;
  files: string[];
  success_criteria: Record<string, boolean | number | string> | null;
}> {
  const res = await fetch(`${BASE}/jobs/${jobId}/files`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Files not available");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Download a specific split output file
// ---------------------------------------------------------------------------

export function downloadSplitFile(jobId: string, filename: string): void {
  const a = document.createElement("a");
  a.href = `${BASE}/jobs/${jobId}/download/${encodeURIComponent(filename)}`;
  a.download = filename;
  a.click();
}

// ---------------------------------------------------------------------------
// Aggregate summary — professor/dimension-level view
// ---------------------------------------------------------------------------

export interface FindingItem {
  label: string;
  count: number;
  pct: number;
}

export interface EntitySummary {
  entity: string;
  group_by: string;
  total_responses: number;
  label_counts: Record<string, number>;
  top_negative_label: string;
  flagged_findings: FindingItem[];
  needs_attention: boolean;
  minority_count: number;
  avg_priority_score: number;
}

export interface AggregateResult {
  job_id: string;
  feedback_mode: string;
  entities: EntitySummary[];
}

export async function fetchAggregate(jobId: string): Promise<AggregateResult> {
  const res = await fetch(`${BASE}/jobs/${jobId}/aggregate`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Aggregate not available");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Minority-only detection endpoint
// ---------------------------------------------------------------------------

export async function detectMinority(
  file: File,
  options: {
    anonymize?: boolean;
    runZeroShotCategorization?: boolean;
    feedbackMode?: FeedbackMode;
  } = {}
): Promise<MinorityResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("anonymize", String(options.anonymize ?? true));
  form.append(
    "run_zero_shot_categorization",
    String(options.runZeroShotCategorization ?? false)
  );
  if (options.feedbackMode) form.append("feedback_mode", options.feedbackMode);

  const res = await fetch(`${BASE}/minority`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Minority detection failed");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Helper: SSE stream (replaces polling — real-time, lower overhead)
// ---------------------------------------------------------------------------

export async function waitForJob(
  jobId: string,
  onProgress?: (status: JobStatus) => void,
  _intervalMs = 1500   // kept for backward compat, ignored (SSE used instead)
): Promise<AnalysisResult> {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`${BASE}/jobs/${jobId}/stream`);

    es.onmessage = async (e) => {
      try {
        const status = JSON.parse(e.data) as JobStatus;
        onProgress?.(status);

        if (status.status === "done") {
          es.close();
          const results = await fetchResults(jobId);
          resolve(results);
        } else if (status.status === "error") {
          es.close();
          reject(new Error(status.message));
        }
      } catch (err) {
        es.close();
        reject(err);
      }
    };

    es.onerror = () => {
      es.close();
      // SSE dropped (tunnel / proxy timeout) — silently fall back to polling loop.
      // Never surface "SSE connection lost" to the user; just keep going.
      const poll = setInterval(async () => {
        try {
          const status = await pollJob(jobId);
          onProgress?.(status);
          if (status.status === "done") {
            clearInterval(poll);
            fetchResults(jobId).then(resolve).catch(reject);
          } else if (status.status === "error") {
            clearInterval(poll);
            reject(new Error(status.message));
          }
          // still "running" → keep polling
        } catch (e) {
          clearInterval(poll);
          reject(e);
        }
      }, 2000);
    };
  });
}

// Extended status type with speed/ETA
export interface ExtendedJobStatus extends JobStatus {
  speed_rows_per_sec?: number;
  stage?: string;
}
