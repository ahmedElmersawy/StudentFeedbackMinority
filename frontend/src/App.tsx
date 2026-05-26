import { useCallback, useState } from "react";
import type { AnalysisResult, JobStatus, Phase, TabId } from "./types";
import { uploadCsv, waitForJob, downloadResultsCsv } from "./api/client";
import { UploadPanel } from "./components/UploadPanel";
import type { UploadOptions } from "./components/UploadPanel";
import { ResultsTable } from "./components/ResultsTable";
import { MinorityPanel } from "./components/MinorityPanel";
import { MismatchPanel } from "./components/MismatchPanel";
import { ReviewQueue } from "./components/ReviewQueue";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cn(...xs: (string | false | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

const TABS: { id: TabId; label: string; icon: string }[] = [
  { id: "overview", label: "Overview", icon: "⬡" },
  { id: "results", label: "All Results", icon: "≡" },
  { id: "minority", label: "Minority Patterns", icon: "◈" },
  { id: "mismatch", label: "Mismatch", icon: "↯" },
  { id: "review", label: "Review Queue", icon: "⊛" },
];

// ---------------------------------------------------------------------------
// KPI card
// ---------------------------------------------------------------------------

function KpiCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string | number;
  sub: string;
  accent: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3.5">
      <div className={cn("absolute inset-x-0 top-0 h-0.5", accent)} />
      <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-bold tabular-nums tracking-tight text-slate-100">{value}</p>
      <p className="mt-0.5 text-[11px] text-slate-600">{sub}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Label distribution bars
// ---------------------------------------------------------------------------

function LabelBars({ dist }: { dist: Record<string, number> }) {
  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  const maxCount = Math.max(...entries.map(([, v]) => v), 1);

  const colorFor = (label: string) => {
    const l = label.toLowerCase();
    if (
      l.startsWith("positive_") || l.startsWith("majority_positive") ||
      l.startsWith("self_positive") || l.includes("_positive_")
    ) return "#34d399";
    if (
      l.startsWith("negative_") || l.startsWith("self_struggle") ||
      l.startsWith("self_minority") || l.includes("_negative_") ||
      l.startsWith("minority_")
    ) return "#fb7185";
    return "#fbbf24";
  };

  return (
    <div className="space-y-2.5">
      {entries.map(([label, count]) => {
        const color = colorFor(label);
        const barW = Math.round((count / maxCount) * 100);
        const pct = ((count / total) * 100).toFixed(1);
        return (
          <div key={label} className="flex items-center gap-3">
            <div className="w-32 shrink-0 truncate text-right text-xs text-slate-500">{label}</div>
            <div className="h-5 flex-1 overflow-hidden rounded-md bg-white/[0.06]">
              <div
                className="flex h-full items-center rounded-md pl-2 text-[11px] font-semibold text-white/90 transition-[width] duration-700"
                style={{ width: `${barW}%`, background: color }}
              >
                {count}
              </div>
            </div>
            <div className="w-10 shrink-0 text-right text-xs text-slate-600">{pct}%</div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Confidence donut
// ---------------------------------------------------------------------------

function ConfidenceDonut({ avg, high, med, low }: { avg: number; high: number; med: number; low: number }) {
  const r = 32, c = 2 * Math.PI * r;
  const segs = [
    { pct: high, color: "#818cf8" },
    { pct: med, color: "#2dd4bf" },
    { pct: low, color: "#fb7185" },
  ];
  let offset = 0;
  const circles = segs.map((s, i) => {
    const len = (s.pct / 100) * c;
    const el = (
      <circle key={i} cx="45" cy="45" r={r} fill="none" stroke={s.color} strokeWidth="12"
        strokeDasharray={`${len} ${c}`} strokeDashoffset={offset} strokeLinecap="round"
        transform="rotate(-90 45 45)" />
    );
    offset -= len;
    return el;
  });
  return (
    <svg width="90" height="90" viewBox="0 0 90 90" className="shrink-0">
      <circle cx="45" cy="45" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12" />
      {circles}
      <text x="45" y="42" textAnchor="middle" fill="#f1f5f9" fontSize="13" fontWeight="700">
        {(avg * 100).toFixed(0)}%
      </text>
      <text x="45" y="56" textAnchor="middle" fill="#64748b" fontSize="9">avg conf</text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Overview tab
// ---------------------------------------------------------------------------

function OverviewTab({ result }: { result: AnalysisResult }) {
  const { summary } = result;
  const confs = result.rows.map((r) => typeof r.confidence === "number" ? r.confidence : 0);
  const highConf = confs.filter((c) => c >= 0.9).length;
  const medConf = confs.filter((c) => c >= 0.7 && c < 0.9).length;
  const lowConf = confs.filter((c) => c < 0.7).length;
  const t = confs.length || 1;

  const modeLabel =
    summary.feedback_mode === "student_to_student" ? "Student → Student (CATME)"
    : summary.feedback_mode === "student_to_professor" ? "Student → Professor"
    : summary.feedback_mode;

  return (
    <div className="space-y-5">
      {/* Mode badge */}
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-slate-500">Mode</span>
        <span className="inline-flex items-center rounded-full bg-indigo-500/15 px-2.5 py-0.5 text-[11px] font-semibold text-indigo-300">
          {modeLabel}
        </span>
        {summary.catme_subtype_distribution && Object.keys(summary.catme_subtype_distribution).length > 0 && (
          <span className="text-[11px] text-slate-600">
            ({Object.entries(summary.catme_subtype_distribution).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(" · ")})
          </span>
        )}
      </div>
      {/* KPIs */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <KpiCard label="Total rows" value={summary.total.toLocaleString()} sub="feedback entries" accent="bg-indigo-500" />
        <KpiCard label="Minority patterns" value={summary.minority} sub={`${summary.minority_pct}% flagged`} accent="bg-rose-400" />
        <KpiCard label="Needs review" value={summary.needs_review} sub={`${summary.review_pct}% low confidence`} accent="bg-amber-400" />
        <KpiCard label="Avg confidence" value={`${(summary.avg_confidence * 100).toFixed(1)}%`} sub="across all predictions" accent="bg-teal-400" />
      </div>

      {/* Charts row */}
      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
          <h3 className="mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            Label Distribution
          </h3>
          {Object.keys(summary.label_distribution).length > 0 ? (
            <LabelBars dist={summary.label_distribution} />
          ) : (
            <p className="text-sm text-slate-600">No label data.</p>
          )}
        </div>
        <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
          <h3 className="mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            Confidence Split
          </h3>
          <div className="flex items-center gap-4">
            <ConfidenceDonut
              avg={summary.avg_confidence}
              high={Math.round((highConf / t) * 100)}
              med={Math.round((medConf / t) * 100)}
              low={Math.round((lowConf / t) * 100)}
            />
            <div className="space-y-2">
              {[
                { label: "High ≥ 0.9", count: highConf, color: "#818cf8" },
                { label: "Med 0.7–0.9", count: medConf, color: "#2dd4bf" },
                { label: "Low < 0.7", count: lowConf, color: "#fb7185" },
              ].map((l) => (
                <div key={l.label} className="flex items-center gap-2 text-[11px]">
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: l.color }} />
                  <span className="text-slate-400">{l.label}</span>
                  <span className="ml-2 text-slate-600">{l.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Minority category preview */}
      {Object.keys(summary.minority_category_breakdown).length > 0 && (
        <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
          <h3 className="mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            Minority Category Snapshot
          </h3>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
            {Object.entries(summary.minority_category_breakdown)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 10)
              .map(([cat, count]) => (
                <div key={cat} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2 text-center">
                  <p className="text-lg font-bold text-slate-100">{count}</p>
                  <p className="mt-0.5 text-[10px] text-slate-500 leading-snug">{cat.replace(/_/g, " ")}</p>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Minority spotlight */}
      <div className="rounded-2xl border border-white/[0.08] bg-[#141928] p-5">
        <h3 className="mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
          Minority Spotlight
        </h3>
        <ul className="divide-y divide-white/[0.05]">
          {result.rows
            .filter((r) => r.is_minority_pattern)
            .slice(0, 6)
            .map((r, i) => (
              <li key={i} className="flex items-start gap-2.5 py-2.5 first:pt-0">
                <span className="mt-0.5 shrink-0 rounded-full bg-rose-500/15 px-2 py-0.5 text-[10px] font-semibold text-rose-300">
                  minority
                </span>
                <span className="min-w-0 flex-1 truncate text-xs text-slate-400">
                  {((r.text ?? r.feedback ?? "") as string)}
                </span>
                {typeof r.confidence === "number" && (
                  <span className="shrink-0 text-[10px] text-slate-600">{(r.confidence * 100).toFixed(0)}%</span>
                )}
              </li>
            ))}
          {result.summary.minority === 0 && (
            <li className="py-4 text-sm text-slate-600">No minority rows detected.</li>
          )}
        </ul>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobMessage, setJobMessage] = useState("");
  const [jobProgress, setJobProgress] = useState({ done: 0, total: 0 });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const handleUpload = useCallback(async (file: File, opts: UploadOptions) => {
    setPhase("uploading");
    setJobMessage("Uploading…");
    setResult(null);
    try {
      const { job_id } = await uploadCsv(file, {
        feedbackMode: opts.feedbackMode === "auto" ? undefined : opts.feedbackMode,
        modelDir: opts.modelDir || undefined,
        anonymize: opts.anonymize,
        includeMinority: opts.includeMinority,
        includeMismatch: opts.includeMismatch,
        confidenceThreshold: opts.confidenceThreshold,
      });
      setJobId(job_id);
      setPhase("running");
      setJobMessage("Pipeline running…");

      const analysis = await waitForJob(
        job_id,
        (status: JobStatus) => {
          setJobMessage(status.message);
          setJobProgress({ done: status.rows_processed, total: status.total_rows });
        },
        1500,
      );
      setResult(analysis);
      setPhase("done");
      setActiveTab("overview");
    } catch (e) {
      setPhase("error");
      setJobMessage(e instanceof Error ? e.message : "An error occurred.");
    }
  }, []);

  const handleReset = useCallback(() => {
    setPhase("idle");
    setJobId(null);
    setJobMessage("");
    setJobProgress({ done: 0, total: 0 });
    setResult(null);
    setActiveTab("overview");
  }, []);

  const minorityCount = result?.summary.minority ?? 0;
  const reviewCount = result?.summary.needs_review ?? 0;
  const mismatchCount = result?.summary.mismatch ?? 0;

  return (
    <div className="flex min-h-screen bg-[#0f1117] font-sans text-slate-200">
      {/* Sidebar */}
      <aside className="hidden w-[220px] min-w-[220px] flex-col border-r border-white/[0.07] bg-[#0a0d14] lg:flex">
        {/* Brand */}
        <div className="border-b border-white/[0.07] px-4 pb-4 pt-5">
          <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-[10px] bg-gradient-to-br from-indigo-500 to-cyan-500 shadow-lg shadow-indigo-900/40">
            <span className="text-lg font-bold text-white">F</span>
          </div>
          <p className="text-[13px] font-semibold text-slate-100">Feedback Atlas</p>
          <p className="mt-0.5 text-[11px] text-slate-600">Minority signal detection</p>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2.5 py-3">
          {TABS.map((tab) => {
            const active = activeTab === tab.id;
            const badge =
              tab.id === "minority" ? minorityCount
              : tab.id === "review" ? reviewCount
              : tab.id === "mismatch" ? mismatchCount
              : 0;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                disabled={!result && tab.id !== "overview"}
                className={cn(
                  "mb-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors disabled:cursor-not-allowed disabled:opacity-30",
                  active
                    ? "bg-indigo-500/15 text-indigo-200"
                    : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-400",
                )}
              >
                <span className="shrink-0 text-base leading-none">{tab.icon}</span>
                <span className="truncate">{tab.label}</span>
                {badge > 0 && (
                  <span className="ml-auto rounded-full bg-indigo-500 px-1.5 py-px text-[10px] font-semibold text-white">
                    {badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Status */}
        <div className="border-t border-white/[0.07] px-4 py-4 space-y-2">
          {[
            { label: "Model ready", done: true },
            { label: result ? `${result.summary.total.toLocaleString()} rows processed` : "Upload CSV", done: !!result },
            { label: phase === "running" ? "Running pipeline…" : result ? "Analysis complete" : "Run pending", done: !!result, active: phase === "running" },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-2">
              <span className={cn(
                "h-2 w-2 shrink-0 rounded-full",
                s.active ? "bg-indigo-400" : s.done ? "bg-emerald-500" : "bg-slate-700",
              )} />
              <span className={cn(
                "text-[11px]",
                s.active ? "text-indigo-200" : s.done ? "text-emerald-400/90" : "text-slate-600",
              )}>
                {s.label}
              </span>
            </div>
          ))}
        </div>
      </aside>

      {/* Main */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Header */}
        <header className="flex h-[52px] shrink-0 items-center justify-between border-b border-white/[0.07] bg-[#0a0d14] px-5">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-slate-100">
              {TABS.find((t) => t.id === activeTab)?.label ?? "Overview"}
            </span>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
              <span className="block h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Live
            </span>
          </div>
          {result && jobId && (
            <button
              onClick={() => downloadResultsCsv(jobId, "feedback_atlas_results.csv")}
              className="rounded-lg border border-white/[0.1] px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-slate-200"
            >
              Download full CSV
            </button>
          )}
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {/* Upload always shown when no result or on overview+no-result */}
          {(!result || activeTab === "overview") && (
            <div className={cn(
              "mb-6 rounded-2xl border border-white/[0.08] bg-[#141928] p-5",
              result ? "hidden lg:block" : undefined,
            )}>
              <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">
                {result ? "Replace dataset" : "Upload CSV"}
              </h2>
              <UploadPanel
                phase={phase}
                jobMessage={jobMessage}
                jobProgress={jobProgress}
                onUpload={handleUpload}
                onReset={handleReset}
              />
            </div>
          )}

          {!result && phase === "idle" && (
            <div className="rounded-xl border border-indigo-500/25 bg-indigo-500/[0.07] px-4 py-3 text-sm text-slate-400">
              Upload a CSV to begin. Supports any feedback format — CATME peer review, course
              evaluations, rated surveys, or plain text.
            </div>
          )}

          {result && activeTab === "overview" && <OverviewTab result={result} />}
          {result && activeTab === "results" && (
            <ResultsTable
              rows={result.rows}
              onDownloadCsv={jobId ? () => downloadResultsCsv(jobId, "results.csv") : undefined}
            />
          )}
          {result && activeTab === "minority" && (
            <MinorityPanel
              result={result}
              onDownloadCsv={jobId ? () => downloadResultsCsv(jobId, "minority_results.csv") : undefined}
            />
          )}
          {result && activeTab === "mismatch" && <MismatchPanel result={result} />}
          {result && activeTab === "review" && <ReviewQueue result={result} />}
        </div>
      </div>
    </div>
  );
}
