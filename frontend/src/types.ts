export type RowLabel = "majority" | "minority" | "needs_review";

export interface DetectionRow {
  id: string;
  feedback: string;
  label: RowLabel;
  confidence: number;
  theme: string;
}

export interface ThemeCount {
  name: string;
  count: number;
}

export interface RatingByLabel {
  label: RowLabel;
  avgRating: number;
  count: number;
}

export interface TrendPoint {
  period: string;
  minority: number;
  majority: number;
}

export interface DetectionSummary {
  total: number;
  majority: number;
  minority: number;
  needsReview: number;
  minorityPct: number;
  majorityThemes: ThemeCount[];
  minorityThemes: ThemeCount[];
  ratingByLabel?: RatingByLabel[];
  trend?: TrendPoint[];
}

export interface DetectionResponse {
  rows: DetectionRow[];
  summary: DetectionSummary;
}

export interface ParsedDataset {
  headers: string[];
  rows: Record<string, string>[];
  suggestedFeedbackColumns: string[];
  suggestedRatingColumn: string | null;
  suggestedDateColumn: string | null;
}
