import { useState } from "react";
import type { AnalysisResult, HistoryItem, Phase } from "../types";
import { UploadPanel } from "../components/UploadPanel";
import type { UploadOptions } from "../components/UploadPanel";

interface Props {
  tab: "upload" | "history";
  history: HistoryItem[];
  phase: Phase;
  jobMessage: string;
  jobProgress: { done: number; total: number };
  result: AnalysisResult | null;
  onUpload: (file: File, opts: UploadOptions) => void;
  onReset: () => void;
}

function StatBadge({ label, value, color }: { label: string; value: number | string; color?: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 18, fontWeight: 800, color: color ?? "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{label}</div>
    </div>
  );
}

export function DatasetManager({ tab, history, phase, jobMessage, jobProgress, result, onUpload, onReset }: Props) {
  const [search, setSearch] = useState("");

  const filtered = history.filter(h =>
    h.filename.toLowerCase().includes(search.toLowerCase()) ||
    h.mode.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", margin: 0 }}>
          {tab === "upload" ? "Upload Dataset" : "Dataset History"}
        </h1>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
          {tab === "upload"
            ? "Upload a CSV file to run the full Feedback Atlas analysis pipeline"
            : `${history.length} previous analyses stored locally`}
        </p>
      </div>

      {tab === "upload" && (
        <>
          {/* Upload card */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Upload Dataset</div>
                <div className="card-subtitle">Supports CATME peer review and course evaluation formats · up to 1M rows</div>
              </div>
            </div>
            <div className="card-body">
              <UploadPanel
                phase={phase}
                jobMessage={jobMessage}
                jobProgress={jobProgress}
                onUpload={onUpload}
                onReset={onReset}
              />
            </div>
          </div>

          {/* Format guide */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(260px,1fr))", gap: 16 }}>
            {[
              {
                title: "CATME Peer Feedback",
                mode: "student_to_student",
                desc: "Headerless CSV with one feedback text per row. The pipeline auto-detects self-assessments vs peer reviews.",
                labels: ["Majority_Positive", "Minority_Peer_Experience", "Suggestion_To_Peer", "Negative_*"],
              },
              {
                title: "Course Evaluation",
                mode: "student_to_professor",
                desc: "Structured CSV with columns for teaching, content, exam, lab dimensions. Ratings and text detected automatically.",
                labels: ["Teaching_*", "Content_*", "Exam_*", "Lab_*", "Support_*"],
              },
            ].map(fmt => (
              <div key={fmt.title} className="card" style={{ border: "1px solid var(--border)" }}>
                <div className="card-body">
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                    <span className="badge badge-violet">{fmt.mode}</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{fmt.title}</span>
                  </div>
                  <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, margin: "0 0 12px" }}>
                    {fmt.desc}
                  </p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {fmt.labels.map(l => (
                      <span key={l} style={{ fontSize: 10, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "2px 6px", color: "var(--text-muted)", fontFamily: "monospace" }}>
                        {l}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {tab === "history" && (
        <>
          {history.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.5">
                  <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <h3 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", margin: "0 0 6px" }}>No history yet</h3>
              <p style={{ fontSize: 13, color: "var(--text-muted)" }}>Upload your first dataset to get started.</p>
            </div>
          ) : (
            <>
              {/* Search */}
              <div className="search-bar" style={{ maxWidth: 360 }}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
                  <circle cx="6" cy="6" r="4"/><path d="M10 10l4 4"/>
                </svg>
                <input
                  placeholder="Search by filename or mode…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
              </div>

              {/* History list */}
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {filtered.map(h => (
                  <div key={h.id} className="card" style={{ transition: "box-shadow 0.15s" }}
                    onMouseEnter={e => (e.currentTarget.style.boxShadow = "var(--shadow-md)")}
                    onMouseLeave={e => (e.currentTarget.style.boxShadow = "")}>
                    <div className="card-body" style={{ display: "flex", alignItems: "center", gap: 20, flexWrap: "wrap" }}>
                      {/* File icon + name */}
                      <div style={{ display: "flex", alignItems: "center", gap: 12, flex: 1, minWidth: 200 }}>
                        <div style={{ width: 40, height: 40, borderRadius: 10, background: "var(--brand-light)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--brand)" strokeWidth="1.5">
                            <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                          </svg>
                        </div>
                        <div>
                          <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{h.filename}</div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                            {new Date(h.uploadedAt).toLocaleString()} · Job {h.jobId.slice(0, 8)}…
                          </div>
                        </div>
                      </div>

                      {/* Stats */}
                      <div style={{ display: "flex", gap: 28, flexWrap: "wrap" }}>
                        <StatBadge label="Records"  value={h.totalRows}      />
                        <StatBadge label="Mode"     value={h.mode === "student_to_student" ? "CATME" : "Professor"} />
                        <StatBadge label="Minority" value={h.minority}        color={h.minority > 0 ? "var(--warning)" : undefined} />
                        <StatBadge label="Review"   value={h.needsReview}     color={h.needsReview > 0 ? "var(--danger)" : undefined} />
                        <StatBadge label="Conf"     value={`${(h.avgConfidence * 100).toFixed(1)}%`} color="var(--success)" />
                      </div>

                      {/* Labels */}
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, maxWidth: 280 }}>
                        {h.labels.slice(0, 5).map(l => (
                          <span key={l} style={{ fontSize: 10, background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 6px", color: "var(--text-muted)" }}>
                            {l.replace(/_/g, " ")}
                          </span>
                        ))}
                        {h.labels.length > 5 && (
                          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>+{h.labels.length - 5} more</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
