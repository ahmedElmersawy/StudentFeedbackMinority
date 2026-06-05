import type { AnalysisResult, HistoryItem, Phase, PipelineStageInfo, View } from "../types";
import type { AggregateResult } from "../api/client";
import type { UploadOptions } from "../components/UploadPanel";
import { UploadPanel } from "../components/UploadPanel";
import { HorizontalBarChart, DonutChart, labelColor } from "../components/Charts";

interface Props {
  result: AnalysisResult | null;
  aggregate: AggregateResult | null;
  phase: Phase;
  jobMessage: string;
  jobProgress: { done: number; total: number };
  stages: PipelineStageInfo[];
  history: HistoryItem[];
  startTime: number | null;
  filename: string;
  onNavigate: (v: View) => void;
  onUpload: (file: File, opts: UploadOptions) => void;
  onReset: () => void;
}

function KpiCard({
  label, value, sub, accent, icon,
}: {
  label: string; value: string | number; sub: string; accent: string; icon: React.ReactNode;
}) {
  return (
    <div className={`kpi-card ${accent}`}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)" }}>
          {label}
        </span>
        <span style={{ fontSize: 20, lineHeight: 1 }}>{icon}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: "var(--text-primary)", lineHeight: 1.1, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>{sub}</div>
    </div>
  );
}

function Skeleton({ w = "100%", h = 16 }: { w?: string | number; h?: number }) {
  return <div className="skeleton" style={{ width: w, height: h, borderRadius: 6 }} />;
}

function SkeletonKpi() {
  return (
    <div className="kpi-card slate">
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
        <Skeleton w={80} h={10} />
        <Skeleton w={24} h={24} />
      </div>
      <Skeleton w={100} h={32} />
      <div style={{ marginTop: 8 }}><Skeleton w={120} h={10} /></div>
    </div>
  );
}

export function DashboardView({
  result, aggregate, phase, jobMessage, jobProgress, stages,
  history, startTime, filename, onNavigate, onUpload, onReset,
}: Props) {
  const isRunning = phase === "uploading" || phase === "running";
  const elapsed   = startTime ? Math.round((Date.now() - startTime) / 1000) : 0;

  // Confidence slices for donut
  const confSlices = result ? (() => {
    const confs = result.rows.map(r => (typeof r.confidence === "number" ? r.confidence : 0));
    const hi  = confs.filter(c => c >= 0.9).length;
    const med = confs.filter(c => c >= 0.7 && c < 0.9).length;
    const lo  = confs.filter(c => c < 0.7).length;
    return [
      { label: "High ≥90%", value: hi,  color: "#059669" },
      { label: "Mid 70–90%", value: med, color: "#d97706" },
      { label: "Low <70%",  value: lo,  color: "#dc2626" },
    ].filter(s => s.value > 0);
  })() : [];

  // Label distribution
  const labelData = result
    ? Object.entries(result.summary.label_distribution)
        .sort((a, b) => b[1] - a[1])
        .map(([label, value], i) => ({ label, value, color: labelColor(label, i) }))
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Page header ────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", margin: 0 }}>
            {result ? "Analysis Dashboard" : "Welcome to Feedback Atlas"}
          </h1>
          <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
            {result
              ? `${result.summary.total.toLocaleString()} records · ${filename}`
              : "AI-powered minority signal detection for student feedback"}
          </p>
        </div>
        {result && (
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary btn-sm" onClick={() => onNavigate("exports")}>
              Export Results
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => onNavigate("datasets-upload")}>
              + New Analysis
            </button>
          </div>
        )}
      </div>

      {/* ── Upload card (always visible, compact when result exists) ─ */}
      {(!result || phase === "idle") && (
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Upload Dataset</div>
              <div className="card-subtitle">CSV · Auto-detects CATME peer review or course evaluation format</div>
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
      )}

      {/* ── Pipeline running state ──────────────────────────────────── */}
      {isRunning && (
        <div className="card animate-fade-up">
          <div className="card-header">
            <div className="card-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="animate-spin" style={{ display: "inline-block", width: 14, height: 14, border: "2px solid var(--brand)", borderTopColor: "transparent", borderRadius: "50%" }} />
              Pipeline Running
            </div>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{elapsed}s elapsed</span>
          </div>
          <div className="card-body" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {stages.map(stage => (
              <div key={stage.id} className="stage-row">
                <div className={`stage-dot ${stage.status}`} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, color: stage.status === "running" ? "var(--brand)" : stage.status === "done" ? "var(--success)" : "var(--text-secondary)" }}>
                    {stage.label}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{stage.description}</div>
                </div>
                {stage.status === "done" && (
                  <span style={{ fontSize: 11, color: "var(--success)", fontWeight: 600 }}>✓</span>
                )}
                {stage.status === "running" && (
                  <span style={{ fontSize: 11, color: "var(--brand)", fontWeight: 600 }}>{jobMessage.slice(0, 30)}</span>
                )}
              </div>
            ))}
            {jobProgress.total > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 4 }}>
                  <span>{jobProgress.done.toLocaleString()} / {jobProgress.total.toLocaleString()} rows classified</span>
                  <span>{Math.round((jobProgress.done / jobProgress.total) * 100)}%</span>
                </div>
                <div className="progress-track">
                  <div className="progress-bar" style={{ width: `${(jobProgress.done / jobProgress.total) * 100}%` }} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── KPI cards ──────────────────────────────────────────────── */}
      {(result || isRunning) && (
        <div className="kpi-grid stagger">
          {isRunning && !result ? (
            [...Array(6)].map((_, i) => <SkeletonKpi key={i} />)
          ) : result ? (
            <>
              <KpiCard label="Total Records"   value={result.summary.total.toLocaleString()}                    sub="feedback entries processed"     accent="violet" icon="📋" />
              <KpiCard label="Labels"          value={Object.keys(result.summary.label_distribution).length}    sub="distinct prediction categories"  accent="blue"   icon="🏷️" />
              <KpiCard label="Minority Found"  value={result.summary.minority.toLocaleString()}                  sub={`${result.summary.minority_pct}% flagged`}       accent="amber"  icon="⚑" />
              <KpiCard label="Needs Review"    value={result.summary.needs_review.toLocaleString()}              sub={`${result.summary.review_pct}% low-confidence`}  accent="red"    icon="⚠" />
              <KpiCard label="Mismatches"      value={result.summary.mismatch.toLocaleString()}                  sub={`${result.summary.mismatch_pct}% of dataset`}    accent="slate"  icon="↯" />
              <KpiCard label="Avg Confidence"  value={`${(result.summary.avg_confidence * 100).toFixed(1)}%`}   sub="across all predictions"          accent="emerald" icon="◎" />
            </>
          ) : null}
        </div>
      )}

      {/* ── Charts row ─────────────────────────────────────────────── */}
      {result && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) min(380px,100%)", gap: 16 }}
             className="charts-row">

          {/* Label distribution */}
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Label Distribution</div>
                <div className="card-subtitle">{labelData.length} prediction categories</div>
              </div>
              <button className="btn btn-ghost btn-sm" onClick={() => onNavigate("analysis-labels")}>
                Explore →
              </button>
            </div>
            <div className="card-body">
              {labelData.length > 0
                ? <HorizontalBarChart data={labelData} height={28} maxLabelWidth={200} />
                : <span style={{ fontSize: 13, color: "var(--text-muted)" }}>No label data.</span>
              }
            </div>
          </div>

          {/* Confidence donut */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Confidence Split</div>
            </div>
            <div className="card-body">
              <DonutChart
                slices={confSlices}
                size={130}
                thickness={20}
                centerLabel={`${(result.summary.avg_confidence * 100).toFixed(0)}%`}
                centerSub="avg conf"
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Minority category snapshot ─────────────────────────────── */}
      {result && Object.keys(result.summary.minority_category_breakdown).length > 0 && (
        <div className="card animate-fade-up">
          <div className="card-header">
            <div>
              <div className="card-title">Minority Category Breakdown</div>
              <div className="card-subtitle">{result.summary.minority.toLocaleString()} rows across {Object.keys(result.summary.minority_category_breakdown).length} categories</div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => onNavigate("analysis-minority")}>
              View all →
            </button>
          </div>
          <div style={{ padding: "16px 20px", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 12 }}>
            {Object.entries(result.summary.minority_category_breakdown)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 10)
              .map(([cat, count]) => (
                <div key={cat} style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 10, padding: "12px 14px", textAlign: "center", cursor: "default" }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = "var(--brand)")}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = "var(--border)")}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", fontVariantNumeric: "tabular-nums" }}>
                    {count.toLocaleString()}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4, lineHeight: 1.4 }}>
                    {cat.replace(/_/g, " ")}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* ── Action summary (professor mode) ────────────────────────── */}
      {aggregate && aggregate.entities.length > 0 && (
        <div className="card animate-fade-up">
          <div className="card-header">
            <div>
              <div className="card-title">Action Summary</div>
              <div className="card-subtitle">
                {aggregate.entities.filter(e => e.needs_attention).length} of {aggregate.entities.length} {aggregate.entities[0]?.group_by ?? "entities"} need attention
              </div>
            </div>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Entity</th>
                  <th>Responses</th>
                  <th>Status</th>
                  <th>Top Finding</th>
                  <th>Minority</th>
                </tr>
              </thead>
              <tbody>
                {aggregate.entities.slice(0, 8).map(e => (
                  <tr key={e.entity}>
                    <td style={{ fontWeight: 500 }}>{e.entity.replace(/_/g, " ")}</td>
                    <td style={{ fontVariantNumeric: "tabular-nums" }}>{e.total_responses.toLocaleString()}</td>
                    <td>
                      <span className={`badge ${e.needs_attention ? "badge-red" : "badge-green"}`}>
                        {e.needs_attention ? "Needs attention" : "OK"}
                      </span>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                      {e.flagged_findings[0]?.label?.replace(/_/g, " ") ?? "—"}
                    </td>
                    <td style={{ fontVariantNumeric: "tabular-nums", color: e.minority_count > 0 ? "var(--warning)" : "var(--text-muted)" }}>
                      {e.minority_count > 0 ? e.minority_count : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Recent history ─────────────────────────────────────────── */}
      {history.length > 0 && (
        <div className="card animate-fade-up">
          <div className="card-header">
            <div className="card-title">Recent Analyses</div>
            <button className="btn btn-ghost btn-sm" onClick={() => onNavigate("datasets-history")}>
              View all →
            </button>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Date</th>
                  <th>Mode</th>
                  <th>Records</th>
                  <th>Minority</th>
                  <th>Avg Conf</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 5).map(h => (
                  <tr key={h.id}>
                    <td>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 16 }}>📄</span>
                        <div>
                          <div style={{ fontWeight: 500, fontSize: 13 }}>{h.filename}</div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{h.labels.length} labels</div>
                        </div>
                      </div>
                    </td>
                    <td style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                      {new Date(h.uploadedAt).toLocaleDateString()}
                    </td>
                    <td>
                      <span className="badge badge-violet">
                        {h.mode === "student_to_student" ? "CATME" : h.mode === "student_to_professor" ? "Professor" : h.mode}
                      </span>
                    </td>
                    <td style={{ fontVariantNumeric: "tabular-nums" }}>{h.totalRows.toLocaleString()}</td>
                    <td style={{ color: h.minority > 0 ? "var(--warning)" : "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
                      {h.minority.toLocaleString()}
                    </td>
                    <td style={{ fontVariantNumeric: "tabular-nums" }}>
                      {(h.avgConfidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Empty state ─────────────────────────────────────────────── */}
      {!result && !isRunning && history.length === 0 && (
        <div className="empty-state" style={{ padding: "48px 24px" }}>
          <div className="empty-icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.5">
              <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
          </div>
          <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", margin: "0 0 8px" }}>
            Start your first analysis
          </h3>
          <p style={{ fontSize: 13, color: "var(--text-muted)", margin: "0 0 24px", maxWidth: 380, lineHeight: 1.6 }}>
            Upload a feedback CSV to detect minority patterns, mismatches, and label distributions using fine-tuned distilroberta models.
          </p>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
            {["CATME Peer Review", "Course Evaluations", "Rated Surveys", "Plain Text Feedback"].map(f => (
              <span key={f} style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 20, padding: "4px 12px", fontSize: 12, color: "var(--text-muted)" }}>
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
