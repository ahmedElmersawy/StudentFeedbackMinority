/** API result types — mirrors backend response shapes. */

export interface SummaryStats {
  total: number;
  minority: number;
  minority_pct: number;
  needs_review: number;
  review_pct: number;
  mismatch: number;
  mismatch_pct: number;
  avg_confidence: number;
  label_distribution: Record<string, number>;
  minority_category_breakdown: Record<string, number>;
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
  cluster_id?: number;
  mismatch_flag?: boolean | number;
  mismatch_type?: string;
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
}

export type Phase = "idle" | "uploading" | "running" | "done" | "error";

export type TabId = "overview" | "results" | "minority" | "mismatch" | "review";
