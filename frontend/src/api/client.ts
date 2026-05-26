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
  } = {}
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append("file", file);
  if (options.feedbackMode) form.append("feedback_mode", options.feedbackMode);
  if (options.modelDir) form.append("model_dir", options.modelDir);
  form.append("anonymize", String(options.anonymize ?? true));
  form.append("include_minority", String(options.includeMinority ?? true));
  form.append("include_mismatch", String(options.includeMismatch ?? true));
  form.append(
    "run_zero_shot_categorization",
    String(options.runZeroShotCategorization ?? false)
  );
  form.append(
    "confidence_threshold",
    String(options.confidenceThreshold ?? 0.65)
  );
  form.append(
    "generate_split_outputs",
    String(options.generateSplitOutputs ?? false)
  );

  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Upload failed");
  }
  return res.json();
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
// Helper: poll until done
// ---------------------------------------------------------------------------

export async function waitForJob(
  jobId: string,
  onProgress?: (status: JobStatus) => void,
  intervalMs = 1500
): Promise<AnalysisResult> {
  return new Promise((resolve, reject) => {
    const timer = setInterval(async () => {
      try {
        const status = await pollJob(jobId);
        onProgress?.(status);
        if (status.status === "done") {
          clearInterval(timer);
          const results = await fetchResults(jobId);
          resolve(results);
        } else if (status.status === "error") {
          clearInterval(timer);
          reject(new Error(status.message));
        }
      } catch (e) {
        clearInterval(timer);
        reject(e);
      }
    }, intervalMs);
  });
}
