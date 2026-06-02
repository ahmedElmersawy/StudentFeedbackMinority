import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisResult, HistoryItem, JobStatus, Phase, PipelineStageInfo, View } from "./types";
import {
  uploadCsv, waitForJob, fetchAggregate, downloadResultsCsv, downloadMinorityCsv,
} from "./api/client";
import type { AggregateResult } from "./api/client";
import type { UploadOptions } from "./components/UploadPanel";

// ── Views ────────────────────────────────────────────────────────────────────
import { DashboardView }    from "./views/DashboardView";
import { PipelineMonitor }  from "./views/PipelineMonitor";
import { DatasetManager }   from "./views/DatasetManager";
import { LabelExplorer }    from "./views/LabelExplorer";
import { ExportCenter }     from "./views/ExportCenter";
import { ReviewWorkspace }  from "./views/ReviewWorkspace";
import { MinorityPanel }    from "./components/MinorityPanel";
import { MismatchPanel }    from "./components/MismatchPanel";
import { ResultsTable }     from "./components/ResultsTable";

// ── Pipeline stage definitions ────────────────────────────────────────────────
const STAGES: PipelineStageInfo[] = [
  { id: "upload",     label: "Upload",              description: "Receiving file",            status: "pending" },
  { id: "preprocess", label: "Preprocessing",        description: "Cleaning & anonymizing",    status: "pending" },
  { id: "embed",      label: "Embedding Generation", description: "MiniLM-L6 encoding",        status: "pending" },
  { id: "classify",   label: "Classification",       description: "Distilroberta inference",   status: "pending" },
  { id: "minority",   label: "Minority Detection",   description: "IsolationForest + DBSCAN",  status: "pending" },
  { id: "mismatch",   label: "Mismatch Detection",   description: "Rating vs text comparison", status: "pending" },
  { id: "report",     label: "Report Generation",    description: "Building summaries",        status: "pending" },
];

// ── History helpers ───────────────────────────────────────────────────────────
const HISTORY_KEY = "fa_analysis_history";

function loadHistory(): HistoryItem[] {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]"); }
  catch { return []; }
}

function saveHistory(items: HistoryItem[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 20)));
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function useToast() {
  const [msg, setMsg] = useState<string | null>(null);
  const t = useRef<ReturnType<typeof setTimeout>>();
  const show = (m: string) => {
    clearTimeout(t.current);
    setMsg(m);
    t.current = setTimeout(() => setMsg(null), 2800);
  };
  return { msg, show };
}

// ── Nav icon helper ───────────────────────────────────────────────────────────
function Icon({ d, size = 15 }: { d: string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
interface NavGroup {
  label: string;
  items: { id: View; label: string; icon: string; badge?: number; requiresResult?: boolean }[];
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [phase,       setPhase]       = useState<Phase>("idle");
  const [jobId,       setJobId]       = useState<string | null>(null);
  const [jobMessage,  setJobMessage]  = useState("");
  const [jobProgress, setJobProgress] = useState({ done: 0, total: 0 });
  const [result,      setResult]      = useState<AnalysisResult | null>(null);
  const [aggregate,   setAggregate]   = useState<AggregateResult | null>(null);
  const [view,        setView]        = useState<View>("dashboard");
  const [stages,      setStages]      = useState<PipelineStageInfo[]>(STAGES.map(s => ({ ...s })));
  const [pipelineLog, setPipelineLog] = useState<string[]>([]);
  const [history,     setHistory]     = useState<HistoryItem[]>(loadHistory);
  const [startTime,   setStartTime]   = useState<number | null>(null);
  const [filename,    setFilename]    = useState<string>("");
  const [speed,       setSpeed]       = useState<number>(0);
  const { msg: toast, show: showToast } = useToast();

  // Fetch aggregate once job is done
  useEffect(() => {
    if (!jobId || phase !== "done") return;
    fetchAggregate(jobId).then(setAggregate).catch(() => setAggregate(null));
  }, [jobId, phase]);

  // Stage progress from message
  function advanceStage(message: string, done = 0, total = 0) {
    setStages(prev => {
      const next = [...prev];
      const msg = message.toLowerCase();
      const stageMap: Record<string, number> = {
        uploading: 0, upload: 0,
        "anonymiz": 1, "clean": 1, "preprocessing": 1,
        "embedding": 2, "encod": 2,
        "classif": 3, "inference": 3,
        "minority": 4, "isolation": 4, "dbscan": 4,
        "mismatch": 5,
        "complete": 6, "report": 6,
      };

      let active = -1;
      for (const [key, idx] of Object.entries(stageMap)) {
        if (msg.includes(key)) { active = idx; break; }
      }

      if (active >= 0) {
        for (let i = 0; i < active; i++) {
          if (next[i].status !== "done") next[i] = { ...next[i], status: "done", completedAt: Date.now() };
        }
        if (next[active].status !== "done") {
          next[active] = { ...next[active], status: "running", startedAt: next[active].startedAt ?? Date.now() };
        }
      }

      if (done >= total && total > 0) {
        next[3] = { ...next[3], status: "done" };
      }

      return next;
    });

    setPipelineLog(prev => {
      const line = `[${new Date().toLocaleTimeString()}] ${message}`;
      return [...prev.slice(-49), line];
    });
  }

  function completeStages() {
    setStages(STAGES.map(s => ({ ...s, status: "done", completedAt: Date.now() })));
  }

  const handleUpload = useCallback(async (file: File, opts: UploadOptions) => {
    setPhase("uploading");
    setJobMessage("Uploading…");
    setResult(null);
    setAggregate(null);
    setFilename(file.name);
    setStartTime(Date.now());
    setStages(STAGES.map(s => ({ ...s, status: "pending", startedAt: undefined, completedAt: undefined })));
    setPipelineLog([]);
    advanceStage("uploading");

    try {
      const { job_id } = await uploadCsv(file, {
        feedbackMode: opts.feedbackMode === "auto" ? undefined : opts.feedbackMode,
        modelDir: opts.modelDir || undefined,
        anonymize: opts.anonymize,
        includeMinority: opts.includeMinority,
        includeMismatch: opts.includeMismatch,
        confidenceThreshold: opts.confidenceThreshold,
        onUploadProgress: (pct) => {
          setJobMessage(pct < 100 ? `Uploading… ${pct}%` : "Processing upload…");
          setJobProgress({ done: pct, total: 100 });
        },
      });
      setJobId(job_id);
      setPhase("running");
      advanceStage("preprocessing");

      const analysis = await waitForJob(
        job_id,
        (status: JobStatus) => {
          setJobMessage(status.message);
          setJobProgress({ done: status.rows_processed, total: status.total_rows });
          advanceStage(status.message, status.rows_processed, status.total_rows);
          // speed_rows_per_sec is included in SSE payload (extended field)
          const ext = status as JobStatus & { speed_rows_per_sec?: number };
          if (ext.speed_rows_per_sec) setSpeed(ext.speed_rows_per_sec);
        },
        1500,
      );

      setResult(analysis);
      setPhase("done");
      completeStages();
      showToast("Analysis complete — results ready");
      setView("dashboard");

      // Save to history
      const hi: HistoryItem = {
        id: job_id,
        filename: file.name,
        uploadedAt: new Date().toISOString(),
        mode: analysis.summary.feedback_mode,
        totalRows: analysis.summary.total,
        minority: analysis.summary.minority,
        needsReview: analysis.summary.needs_review,
        mismatch: analysis.summary.mismatch,
        labels: Object.keys(analysis.summary.label_distribution),
        avgConfidence: analysis.summary.avg_confidence,
        jobId: job_id,
      };
      const newHistory = [hi, ...history.filter(h => h.jobId !== job_id)];
      setHistory(newHistory);
      saveHistory(newHistory);
    } catch (e) {
      setPhase("error");
      setJobMessage(e instanceof Error ? e.message : "An error occurred.");
      setStages(prev => prev.map(s => s.status === "running" ? { ...s, status: "error" } : s));
    }
  }, [history]);

  const handleReset = useCallback(() => {
    setPhase("idle");
    setJobId(null);
    setJobMessage("");
    setJobProgress({ done: 0, total: 0 });
    setResult(null);
    setAggregate(null);
    setFilename("");
    setStartTime(null);
    setStages(STAGES.map(s => ({ ...s, status: "pending", startedAt: undefined, completedAt: undefined })));
    setPipelineLog([]);
    setSpeed(0);
  }, []);

  const isRunning = phase === "uploading" || phase === "running";

  // ── Nav config ──────────────────────────────────────────────────────────────
  const navGroups: NavGroup[] = [
    {
      label: "Overview",
      items: [
        { id: "dashboard",  label: "Dashboard",   icon: "M1 1h6v6H1zM9 1h6v6H9zM1 9h6v6H1zM9 9h6v6H9z" },
        { id: "pipeline",   label: "Pipeline",    icon: "M8 2v12M2 6l6-4 6 4M2 10l6 4 6-4" },
      ],
    },
    {
      label: "Datasets",
      items: [
        { id: "datasets-upload",  label: "Upload Dataset",  icon: "M8 12V4M4 8l4-4 4 4M2 14h12" },
        { id: "datasets-history", label: "Dataset History", icon: "M12 8a4 4 0 11-8 0 4 4 0 018 0zM2 14s0-4 6-4 6 4 6 4" },
      ],
    },
    {
      label: "Analysis",
      items: [
        { id: "analysis-results",  label: "All Results",       icon: "M1 3h14M1 7h14M1 11h8",        requiresResult: true },
        { id: "analysis-labels",   label: "By Label",          icon: "M6 2l2-1 2 1v3l-2 1-2-1zM1 9l2-1 2 1v3l-2 1-2-1zM11 9l2-1 2 1v3l-2 1-2-1z", requiresResult: true, badge: result ? Object.keys(result.summary.label_distribution).length : 0 },
        { id: "analysis-minority", label: "Minority Patterns", icon: "M8 2L2 13h12L8 2zM8 7v3M8 11.5v.5", requiresResult: true, badge: result?.summary.minority },
        { id: "analysis-mismatch", label: "Mismatch",          icon: "M2 8h5l2-4 2 8 2-4",           requiresResult: true, badge: result?.summary.mismatch },
      ],
    },
    {
      label: "Workspace",
      items: [
        { id: "review",  label: "Review Queue", icon: "M1 3h14M1 7h10M1 11h6", requiresResult: true, badge: result?.summary.needs_review },
        { id: "exports", label: "Export Center", icon: "M8 12V4M4 8l4 4 4-4M2 14h12",               requiresResult: true },
      ],
    },
  ];

  // ── Render helpers ──────────────────────────────────────────────────────────
  function renderView() {
    switch (view) {
      case "dashboard":
        return (
          <DashboardView
            result={result}
            aggregate={aggregate}
            phase={phase}
            jobMessage={jobMessage}
            jobProgress={jobProgress}
            stages={stages}
            history={history}
            startTime={startTime}
            filename={filename}
            onNavigate={setView}
            onUpload={handleUpload}
            onReset={handleReset}
          />
        );
      case "pipeline":
        return (
          <PipelineMonitor
            stages={stages}
            log={pipelineLog}
            phase={phase}
            jobMessage={jobMessage}
            jobProgress={jobProgress}
            startTime={startTime}
            filename={filename}
            speed={speed}
          />
        );
      case "datasets-upload":
        return (
          <DatasetManager
            tab="upload"
            history={history}
            phase={phase}
            jobMessage={jobMessage}
            jobProgress={jobProgress}
            onUpload={handleUpload}
            onReset={handleReset}
            result={result}
          />
        );
      case "datasets-history":
        return (
          <DatasetManager
            tab="history"
            history={history}
            phase={phase}
            jobMessage={jobMessage}
            jobProgress={jobProgress}
            onUpload={handleUpload}
            onReset={handleReset}
            result={result}
          />
        );
      case "analysis-results":
        return result ? (
          <div>
            <SectionHeader title="All Results" sub={`${result.summary.total.toLocaleString()} records`} />
            <ResultsTable rows={result.rows} onDownloadCsv={jobId ? () => { downloadResultsCsv(jobId, "results.csv"); showToast("Downloading…"); } : undefined} />
          </div>
        ) : <NoDataState onNavigate={setView} />;
      case "analysis-labels":
        return result && jobId ? (
          <LabelExplorer jobId={jobId} result={result} onToast={showToast} />
        ) : <NoDataState onNavigate={setView} />;
      case "analysis-minority":
        return result ? (
          <div>
            <SectionHeader title="Minority Patterns" sub={`${result.summary.minority.toLocaleString()} flagged rows`} />
            <MinorityPanel result={result} onDownloadCsv={jobId ? () => { downloadMinorityCsv(jobId); showToast("Downloading minority CSV…"); } : undefined} />
          </div>
        ) : <NoDataState onNavigate={setView} />;
      case "analysis-mismatch":
        return result ? (
          <div>
            <SectionHeader title="Mismatch Detection" sub={`${result.summary.mismatch.toLocaleString()} mismatches`} />
            <MismatchPanel result={result} />
          </div>
        ) : <NoDataState onNavigate={setView} />;
      case "review":
        return result ? (
          <ReviewWorkspace result={result} />
        ) : <NoDataState onNavigate={setView} />;
      case "exports":
        return result && jobId ? (
          <ExportCenter result={result} jobId={jobId} onToast={showToast} />
        ) : <NoDataState onNavigate={setView} />;
      default:
        return <NoDataState onNavigate={setView} />;
    }
  }

  return (
    <div className="app-layout">
      {/* ── Sidebar ────────────────────────────────────────────────── */}
      <aside className="sidebar">
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 13L8 3l5 10H3z"/>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", lineHeight: 1.2 }}>
              Feedback Atlas
            </div>
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>
              AI Analytics Platform
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, overflowY: "auto", padding: "4px 0" }}>
          {navGroups.map(group => (
            <div key={group.label} className="nav-section">
              <div className="nav-section-label">{group.label}</div>
              {group.items.map(item => {
                const disabled = !!(item.requiresResult && !result);
                const active   = view === item.id;
                const badge    = (item.badge ?? 0) > 0 ? item.badge : 0;
                return (
                  <button
                    key={item.id}
                    className={`nav-item${active ? " active" : ""}`}
                    disabled={disabled}
                    onClick={() => setView(item.id)}
                  >
                    <Icon d={item.icon} size={14} />
                    <span style={{ flex: 1 }}>{item.label}</span>
                    {isRunning && item.id === "pipeline" && (
                      <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--brand)", display: "inline-block", animation: "pulse-dot 1.2s infinite" }} />
                    )}
                    {badge! > 0 && !active && (
                      <span className="nav-badge" style={{
                        background: item.id === "analysis-minority" ? "var(--warning-bg)" :
                                    item.id === "analysis-mismatch" ? "#fff7ed" :
                                    item.id === "review"            ? "var(--danger-bg)" :
                                    "var(--brand-light)",
                        color: item.id === "analysis-minority" ? "var(--warning)" :
                               item.id === "analysis-mismatch" ? "#c2410c" :
                               item.id === "review"            ? "var(--danger)" :
                               "var(--brand)",
                      }}>
                        {badge! > 999 ? `${(badge! / 1000).toFixed(1)}k` : badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        {/* Status footer */}
        <div style={{ borderTop: "1px solid var(--border)", padding: "12px 16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%",
              background: isRunning ? "var(--brand)" : phase === "done" ? "var(--success)" : "#d1d5db",
              display: "inline-block",
              animation: isRunning ? "pulse-dot 1.2s infinite" : "none",
            }} />
            <span style={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 500 }}>
              {isRunning ? "Processing…" : phase === "done" ? "Ready" : "Idle"}
            </span>
          </div>
          {result && (
            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
              {result.summary.total.toLocaleString()} rows · {Object.keys(result.summary.label_distribution).length} labels
            </div>
          )}
        </div>
      </aside>

      {/* ── Main area ──────────────────────────────────────────────── */}
      <div className="main-area">
        {/* Top bar */}
        <div className="top-bar">
          <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1 }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Feedback Atlas</span>
            <span style={{ color: "var(--text-muted)", fontSize: 12 }}>/</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>
              {navGroups.flatMap(g => g.items).find(i => i.id === view)?.label ?? "Dashboard"}
            </span>
          </div>

          {result && jobId && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <button className="btn btn-secondary btn-sm" onClick={() => { setView("exports"); }}>
                <Icon d="M8 12V4M4 8l4 4 4-4M2 14h12" size={13} />
                Export
              </button>
              {phase !== "idle" && (
                <button className="btn btn-ghost btn-sm" onClick={handleReset} style={{ color: "var(--danger)" }}>
                  Reset
                </button>
              )}
            </div>
          )}
        </div>

        {/* Stats strip when result is ready */}
        {result && (
          <div className="stat-strip">
            {[
              { label: "Records",    value: result.summary.total.toLocaleString(),          color: "var(--text-primary)" },
              { label: "Mode",       value: result.summary.feedback_mode === "student_to_student" ? "CATME" : "Professor", color: "var(--brand)" },
              { label: "Minority",   value: `${result.summary.minority.toLocaleString()} (${result.summary.minority_pct}%)`, color: "var(--warning)" },
              { label: "Review",     value: result.summary.needs_review.toLocaleString(),   color: "var(--danger)" },
              { label: "Mismatch",   value: result.summary.mismatch.toLocaleString(),        color: "#c2410c" },
              { label: "Avg Conf",   value: `${(result.summary.avg_confidence * 100).toFixed(1)}%`, color: "var(--success)" },
              { label: "Labels",     value: Object.keys(result.summary.label_distribution).length.toString(), color: "var(--info)" },
            ].map(s => (
              <div key={s.label} className="stat-pill">
                <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.06em" }}>{s.label}</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: s.color, fontVariantNumeric: "tabular-nums" }}>{s.value}</span>
              </div>
            ))}
          </div>
        )}

        {/* Page content */}
        <div className="page-content">
          <div className="animate-fade-up">
            {renderView()}
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

// ── Helper components ─────────────────────────────────────────────────────────

function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <h1 style={{ fontSize: 20, fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>{title}</h1>
      {sub && <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>{sub}</p>}
    </div>
  );
}

function NoDataState({ onNavigate }: { onNavigate: (v: View) => void }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="1.5">
          <path d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
        </svg>
      </div>
      <h3 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", margin: "0 0 6px" }}>
        No data available
      </h3>
      <p style={{ fontSize: 13, color: "var(--text-muted)", margin: "0 0 20px", maxWidth: 340, lineHeight: 1.6 }}>
        Upload a feedback CSV to begin analysis. The pipeline will automatically detect the mode and run all detections.
      </p>
      <button className="btn btn-primary" onClick={() => onNavigate("datasets-upload")}>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M8 12V4M4 8l4-4 4 4M2 14h12"/>
        </svg>
        Upload Dataset
      </button>
    </div>
  );
}
