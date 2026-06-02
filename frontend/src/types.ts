/** API result types — mirrors backend response shapes. */
export type { SummaryStats, ResultRow, AnalysisResult, JobStatus, MinorityResult } from "./api/client";

export type Phase = "idle" | "uploading" | "running" | "done" | "error";

// Navigation views
export type View =
  | "dashboard"
  | "datasets-upload"
  | "datasets-history"
  | "analysis-results"
  | "analysis-labels"
  | "analysis-minority"
  | "analysis-mismatch"
  | "pipeline"
  | "review"
  | "exports"
  | "settings";

// Pipeline stages
export type StageStatus = "pending" | "running" | "done" | "error";

export interface PipelineStageInfo {
  id: string;
  label: string;
  description: string;
  status: StageStatus;
  startedAt?: number;
  completedAt?: number;
}

// Analysis history (stored in localStorage)
export interface HistoryItem {
  id: string;
  filename: string;
  uploadedAt: string;       // ISO string
  mode: string;
  totalRows: number;
  minority: number;
  needsReview: number;
  mismatch: number;
  labels: string[];
  avgConfidence: number;
  jobId: string;
}
