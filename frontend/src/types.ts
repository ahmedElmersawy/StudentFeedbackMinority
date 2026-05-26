/** API result types — mirrors backend response shapes. */

export type { SummaryStats, ResultRow, AnalysisResult, JobStatus, MinorityResult } from "./api/client";

export type Phase = "idle" | "uploading" | "running" | "done" | "error";

export type TabId = "overview" | "summary" | "results" | "minority" | "mismatch" | "review";
