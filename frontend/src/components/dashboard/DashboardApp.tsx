import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { runMockMinorityDetection } from "../../api/mockDetect";
import type { DetectionResponse, DetectionRow, ParsedDataset } from "../../types";
import { parseCsvFile, validateFeedbackColumns } from "../../lib/csv";
import { guessSegmentColumn, segmentBreakdown } from "../../lib/segments";
import { keywordFrequency } from "../../lib/themes";
import { BrandMark } from "./BrandMark";
import { Icon, type IconName } from "./Icons";
import { ColumnSelector } from "../ColumnSelector";
import { ExportButtons } from "../ExportButtons";
import { ResultsTable } from "../ResultsTable";

type Phase = "empty" | "loaded" | "running" | "done" | "error";

type NavId =
  | "overview"
  | "segment"
  | "compare"
  | "minority"
  | "anomalies"
  | "table"
  | "config";

const NAV: {
  id: NavId;
  icon: IconName;
  label: string;
  section: "main" | "detection" | "data";
  badgeFrom?: "minority";
}[] = [
  { id: "overview", icon: "grid", label: "Overview", section: "main" },
  { id: "segment", icon: "bar", label: "By segment", section: "main" },
  { id: "compare", icon: "circles", label: "Compare groups", section: "main" },
  { id: "minority", icon: "star", label: "Minority patterns", section: "detection", badgeFrom: "minority" },
  { id: "anomalies", icon: "shield", label: "Needs review", section: "detection" },
  { id: "table", icon: "table", label: "All rows", section: "data" },
  { id: "config", icon: "settings", label: "Data mapping", section: "data" },
];

const LABEL_COLORS: Record<string, string> = {
  majority: "#2dd4bf",
  minority: "#fb7185",
  needs_review: "#fbbf24",
};

function cn(...xs: (string | false | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

function Card({
  title,
  titleRight,
  children,
  className,
}: {
  title: string;
  titleRight?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-white/[0.08] bg-[#141928] p-4 sm:p-5 shadow-[0_1px_0_rgba(255,255,255,0.04)_inset]",
        className,
      )}
    >
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-400">{title}</h3>
        {titleRight}
      </div>
      {children}
    </div>
  );
}

function KpiCard({
  label,
  value,
  delta,
  deltaClass,
  accent,
}: {
  label: string;
  value: string;
  delta: string;
  deltaClass?: string;
  accent: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-white/[0.08] bg-[#141928] px-4 py-3.5">
      <div className={cn("absolute inset-x-0 top-0 h-0.5", accent)} />
      <p className="text-[10px] font-medium uppercase tracking-[0.1em] text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-bold tabular-nums tracking-tight text-slate-100">{value}</p>
      <p className={cn("mt-1 text-[11px] text-slate-500", deltaClass)}>{delta}</p>
    </div>
  );
}

function confidenceBuckets(rows: DetectionRow[]) {
  let high = 0;
  let med = 0;
  let low = 0;
  for (const r of rows) {
    if (r.confidence >= 0.9) high++;
    else if (r.confidence >= 0.7) med++;
    else low++;
  }
  const t = rows.length || 1;
  return {
    highPct: Math.round((high / t) * 100),
    medPct: Math.round((med / t) * 100),
    lowPct: Math.round((low / t) * 100),
    avg: rows.length ? rows.reduce((s, r) => s + r.confidence, 0) / rows.length : 0,
  };
}

function sparkFromResult(result: DetectionResponse | null): number[] {
  if (!result) return [];
  const { trend } = result.summary;
  if (trend && trend.length > 1) {
    return trend.map((p) => {
      const s = p.minority + p.majority + 1e-6;
      return p.minority / s;
    });
  }
  const rows = result.rows;
  const bins = 16;
  const out: number[] = [];
  const chunk = Math.max(1, Math.ceil(rows.length / bins));
  for (let b = 0; b < bins; b++) {
    const slice = rows.slice(b * chunk, (b + 1) * chunk);
    if (!slice.length) break;
    out.push(slice.reduce((s, r) => s + r.confidence, 0) / slice.length);
  }
  return out.length ? out : [0.82];
}

function Sparkline({ values }: { values: number[] }) {
  if (!values.length) return <div className="h-[52px] text-xs text-slate-600">Run analysis to see activity.</div>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return (
    <div className="flex h-[52px] items-end gap-0.5">
      {values.map((v, i) => {
        const pct = ((v - min) / span) * 100;
        const bg = v >= max - span * 0.15 ? "#818cf8" : v >= min + span * 0.35 ? "#2dd4bf" : "#fbbf24";
        return (
          <div
            key={i}
            className="min-h-[4px] flex-1 rounded-t-[3px] transition-[height] duration-500"
            style={{ height: `${Math.max(pct, 10)}%`, background: bg }}
          />
        );
      })}
    </div>
  );
}

function MinorityTag({ kind }: { kind: "focus" | "review" }) {
  const styles =
    kind === "review"
      ? "bg-rose-500/15 text-rose-300"
      : "bg-amber-500/15 text-amber-300";
  return <span className={cn("shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold", styles)}>{kind === "review" ? "review" : "minority"}</span>;
}

function PredictionBars({
  mode,
  items,
}: {
  mode: "Count" | "Pct";
  items: { label: string; count: number; color: string }[];
}) {
  const total = items.reduce((s, p) => s + p.count, 0) || 1;
  const maxCount = Math.max(...items.map((p) => p.count), 1);
  return (
    <div className="flex flex-col gap-2">
      {items.map((p) => {
        const display = mode === "Count" ? p.count : `${Math.round((p.count / total) * 100)}%`;
        const barWidth = mode === "Count" ? (p.count / maxCount) * 100 : (p.count / total) * 100;
        return (
          <div key={p.label} className="flex items-center gap-2.5">
            <div className="w-[5.5rem] shrink-0 text-right text-[11px] text-slate-500">{p.label}</div>
            <div className="h-5 flex-1 overflow-hidden rounded-md bg-white/[0.06]">
              <div
                className="flex h-full items-center rounded-md pl-2 text-[11px] font-semibold text-white/90 transition-[width] duration-500"
                style={{ width: `${barWidth}%`, background: p.color }}
              >
                {display}
              </div>
            </div>
            <div className="w-9 shrink-0 text-right text-[11px] text-slate-600">{p.count}</div>
          </div>
        );
      })}
    </div>
  );
}

function Donut({ avg, highPct, medPct, lowPct }: { avg: number; highPct: number; medPct: number; lowPct: number }) {
  const r = 32;
  const c = 2 * Math.PI * r;
  const segs = [
    { pct: highPct, color: "#818cf8" },
    { pct: medPct, color: "#2dd4bf" },
    { pct: lowPct, color: "#fb7185" },
  ];
  let dashOffset = 0;
  const circles = segs.map((s, i) => {
    const len = (s.pct / 100) * c;
    const el = (
      <circle
        key={i}
        cx="45"
        cy="45"
        r={r}
        fill="none"
        stroke={s.color}
        strokeWidth="12"
        strokeDasharray={`${len} ${c}`}
        strokeDashoffset={dashOffset}
        strokeLinecap="round"
        transform="rotate(-90 45 45)"
      />
    );
    dashOffset -= len;
    return el;
  });
  return (
    <svg width="90" height="90" viewBox="0 0 90 90" className="shrink-0">
      <circle cx="45" cy="45" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12" />
      {circles}
      <text x="45" y="42" textAnchor="middle" fill="#f1f5f9" fontSize="13" fontWeight="700">
        {(avg * 100).toFixed(1)}%
      </text>
      <text x="45" y="56" textAnchor="middle" fill="#64748b" fontSize="9">
        avg conf
      </text>
    </svg>
  );
}

const chartTooltipStyle = {
  backgroundColor: "#1e293b",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 8,
  fontSize: 12,
};

export default function DashboardApp() {
  const [phase, setPhase] = useState<Phase>("empty");
  const [fileName, setFileName] = useState("");
  const [dataset, setDataset] = useState<ParsedDataset | null>(null);
  const [feedbackCols, setFeedbackCols] = useState<string[]>([]);
  const [ratingCol, setRatingCol] = useState<string | null>(null);
  const [dateCol, setDateCol] = useState<string | null>(null);
  const [segmentCol, setSegmentCol] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState<NavId>("overview");
  const [barMode, setBarMode] = useState<"Count" | "Pct">("Count");
  const [dragOver, setDragOver] = useState(false);

  const onFile = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setPhase("empty");
    try {
      const parsed = await parseCsvFile(file);
      setFileName(file.name);
      setDataset(parsed);
      const fb = parsed.suggestedFeedbackColumns.length ? parsed.suggestedFeedbackColumns : parsed.headers.slice(0, 3);
      setFeedbackCols(fb);
      setRatingCol(parsed.suggestedRatingColumn);
      setDateCol(parsed.suggestedDateColumn);
      setSegmentCol(guessSegmentColumn(parsed, fb));
      setPhase("loaded");
      setActiveNav("overview");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to parse CSV");
      setPhase("error");
    }
  }, []);

  useEffect(() => {
    if (!dataset?.headers.length) return;
    if (segmentCol && !dataset.headers.includes(segmentCol)) {
      setSegmentCol(guessSegmentColumn(dataset, feedbackCols));
    }
  }, [dataset, feedbackCols, segmentCol]);

  const runDetection = async () => {
    if (!dataset) return;
    const v = validateFeedbackColumns(dataset.headers, feedbackCols);
    if (v) {
      setError(v);
      return;
    }
    setError(null);
    setPhase("running");
    try {
      const res = await runMockMinorityDetection(dataset, feedbackCols, ratingCol, dateCol);
      setResult(res);
      setPhase("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed");
      setPhase("loaded");
    }
  };

  const minorityBadge = result?.summary.minority ?? 0;
  const segments = useMemo(
    () => (dataset && segmentCol ? segmentBreakdown(dataset, segmentCol) : []),
    [dataset, segmentCol],
  );

  const distributionItems = useMemo(() => {
    if (!result) return [];
    const s = result.summary;
    return [
      { label: "Majority", count: s.majority, color: LABEL_COLORS.majority },
      { label: "Minority", count: s.minority, color: LABEL_COLORS.minority },
      { label: "Needs review", count: s.needsReview, color: LABEL_COLORS.needs_review },
    ];
  }, [result]);

  const bucket = result ? confidenceBuckets(result.rows) : null;
  const sparks = useMemo(() => sparkFromResult(result), [result]);

  const pieData = result
    ? [
        { name: "Majority", value: result.summary.majority },
        { name: "Minority", value: result.summary.minority },
        { name: "Needs review", value: result.summary.needsReview },
      ]
    : [];

  const themeBars = useMemo(() => {
    if (!result) return [];
    const maj = result.summary.majorityThemes.map((t) => ({
      label: `M · ${t.name}`,
      count: t.count,
      fill: "#2dd4bf",
    }));
    const min = result.summary.minorityThemes.map((t) => ({
      label: `m · ${t.name}`,
      count: t.count,
      fill: "#fb7185",
    }));
    return [...maj, ...min].slice(0, 12);
  }, [result]);

  const kwAnomaly = useMemo(() => {
    if (!result) return [];
    const texts = result.rows.filter((r) => r.label === "needs_review").map((r) => r.feedback);
    return keywordFrequency(texts, 14);
  }, [result]);

  const steps = useMemo(() => {
    const rows = dataset?.rows.length ?? 0;
    return [
      { label: "Model ready (browser demo)", state: "done" as const },
      {
        label: dataset ? `CSV ingested (${rows.toLocaleString()} rows)` : "CSV ingested",
        state: dataset ? ("done" as const) : ("idle" as const),
      },
      {
        label: phase === "running" ? "Scoring rows…" : result ? "Scoring complete" : "Run analysis",
        state: phase === "running" ? ("active" as const) : result ? ("done" as const) : ("idle" as const),
      },
      {
        label: result ? "Patterns & export ready" : "Patterns pending",
        state: result ? ("done" as const) : ("idle" as const),
      },
    ];
  }, [dataset, phase, result]);

  const navTitle = NAV.find((n) => n.id === activeNav)?.label ?? "Overview";

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.toLowerCase().endsWith(".csv")) void onFile(f);
  };

  return (
    <div className="flex min-h-screen bg-[#0f1117] font-sans text-slate-200">
      {/* Sidebar */}
      <aside className="flex w-[220px] min-w-[220px] flex-col border-r border-white/[0.07] bg-[#0a0d14]">
        <div className="border-b border-white/[0.07] px-4 pb-4 pt-5">
          <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-[10px] bg-gradient-to-br from-indigo-500 to-cyan-500 shadow-lg shadow-indigo-900/40">
            <BrandMark className="h-7 w-7" />
          </div>
          <p className="text-[13px] font-semibold text-slate-100">Feedback Atlas</p>
          <p className="mt-0.5 text-[11px] text-slate-600">Minority signals · local mock</p>
        </div>

        <nav className="flex-1 space-y-4 overflow-y-auto px-2.5 py-3">
          {(["main", "detection", "data"] as const).map((section) => (
            <div key={section}>
              <p className="mb-1.5 px-2 text-[10px] font-medium uppercase tracking-[0.12em] text-slate-600">
                {section === "main" ? "Main" : section === "detection" ? "Detection" : "Data"}
              </p>
              {NAV.filter((n) => n.section === section).map((item) => {
                const active = activeNav === item.id;
                const badge = item.badgeFrom === "minority" && minorityBadge > 0 ? minorityBadge : undefined;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActiveNav(item.id)}
                    className={cn(
                      "mb-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors",
                      active ? "bg-indigo-500/15 text-indigo-200" : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-400",
                    )}
                  >
                    <Icon name={item.icon} />
                    <span className="truncate">{item.label}</span>
                    {badge !== undefined && (
                      <span className="ml-auto rounded-full bg-indigo-500 px-1.5 py-px text-[10px] font-semibold text-white">
                        {badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="border-t border-white/[0.07] px-4 py-4">
          {steps.map((s) => (
            <div key={s.label} className="mb-2.5 flex items-center gap-2 last:mb-0">
              <span
                className={cn(
                  "h-2 w-2 shrink-0 rounded-full",
                  s.state === "done" && "bg-emerald-500",
                  s.state === "active" && "bg-indigo-400",
                  s.state === "idle" && "bg-slate-700",
                )}
              />
              <span
                className={cn(
                  "text-[11px] leading-snug",
                  s.state === "done" && "text-emerald-400/90",
                  s.state === "active" && "text-indigo-200",
                  s.state === "idle" && "text-slate-600",
                )}
              >
                {s.label}
              </span>
            </div>
          ))}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[52px] shrink-0 items-center justify-between border-b border-white/[0.07] bg-[#0a0d14] px-5">
          <div className="flex min-w-0 items-center gap-3">
            <h1 className="shrink-0 text-sm font-semibold text-slate-100">{navTitle}</h1>
            {dataset && fileName && (
              <span className="hidden max-w-[220px] truncate text-xs text-slate-600 md:inline" title={fileName}>
                · {fileName}
              </span>
            )}
            <span className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
              <span className="block h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Live
            </span>
          </div>
          <div className="flex items-center gap-2">
            {result && <ExportButtons result={result} variant="dark" />}
            <button
              type="button"
              onClick={() => void runDetection()}
              disabled={phase === "running" || !dataset || !feedbackCols.length}
              className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3.5 py-1.5 text-xs font-semibold text-white shadow-md shadow-indigo-900/30 hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Icon name="play" />
              {phase === "running" ? "Running…" : "Run analysis"}
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-5">
          {error && (
            <div className="mb-4 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          {activeNav === "config" && dataset && (
            <div className="mb-6 space-y-4">
              <ColumnSelector
                variant="dark"
                dataset={dataset}
                feedbackColumns={feedbackCols}
                onChangeFeedback={setFeedbackCols}
                ratingColumn={ratingCol}
                onChangeRating={setRatingCol}
                dateColumn={dateCol}
                onChangeDate={setDateCol}
              />
              <Card title="Segment column (optional)">
                <p className="mb-3 text-xs text-slate-500">
                  Used for &quot;By segment&quot; charts. Auto-picked when possible; override if your cohort column has
                  another name.
                </p>
                <select
                  className="w-full max-w-md rounded-lg border border-white/[0.1] bg-[#0f1117] px-3 py-2 text-sm text-slate-200"
                  value={segmentCol ?? ""}
                  onChange={(e) => setSegmentCol(e.target.value || null)}
                >
                  <option value="">— None —</option>
                  {dataset.headers.map((h) => (
                    <option key={h} value={h}>
                      {h}
                    </option>
                  ))}
                </select>
              </Card>
            </div>
          )}

          {activeNav === "overview" && (
            <>
              {!dataset && (
                <div className="mb-5 rounded-xl border border-indigo-500/25 bg-indigo-500/[0.07] px-4 py-3 text-sm text-slate-400">
                  Upload a CSV to unlock KPIs, segments, and exports. Detection stays in the browser (mock) until you wire{" "}
                  <code className="rounded bg-white/5 px-1 font-mono text-[11px] text-indigo-200">/api/detect-minority</code>.
                </div>
              )}
              <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
                <KpiCard
                  label="Total rows"
                  value={dataset ? dataset.rows.length.toLocaleString() : "—"}
                  delta={dataset ? "feedback entries" : "Upload a CSV"}
                  accent="bg-indigo-500"
                />
                <KpiCard
                  label="Mean confidence"
                  value={bucket ? bucket.avg.toFixed(3) : "—"}
                  delta={bucket ? "from last run" : "Run analysis"}
                  deltaClass={bucket && bucket.avg >= 0.8 ? "text-emerald-400/90" : undefined}
                  accent="bg-teal-400"
                />
                <KpiCard
                  label="Label classes"
                  value={result ? "3" : "—"}
                  delta="majority · minority · review"
                  accent="bg-amber-400"
                />
                <KpiCard
                  label="Minority flags"
                  value={result ? String(result.summary.minority) : "—"}
                  delta={result ? `${result.summary.minorityPct}% of dataset` : "pending"}
                  deltaClass="text-rose-300/90"
                  accent="bg-rose-500"
                />
              </div>

              <div className="mb-4 grid gap-3 lg:grid-cols-[2fr_1fr]">
                <Card
                  title="Label distribution"
                  titleRight={
                    <div className="flex gap-1">
                      {(["Count", "Pct"] as const).map((m) => (
                        <button
                          key={m}
                          type="button"
                          onClick={() => setBarMode(m)}
                          className={cn(
                            "rounded-md px-2.5 py-1 text-[11px] font-semibold transition-colors",
                            barMode === m ? "bg-indigo-500/20 text-indigo-200" : "text-slate-600 hover:text-slate-400",
                          )}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                  }
                >
                  {result ? (
                    <PredictionBars mode={barMode} items={distributionItems} />
                  ) : (
                    <p className="text-sm text-slate-500">Run analysis to populate distribution bars.</p>
                  )}
                </Card>

                <Card title="Confidence split">
                  {result && bucket ? (
                    <>
                      <div className="mb-3 flex justify-center">
                        <Donut
                          avg={bucket.avg}
                          highPct={bucket.highPct}
                          medPct={bucket.medPct}
                          lowPct={bucket.lowPct}
                        />
                      </div>
                      <div className="space-y-2">
                        {[
                          { label: "High ≥ 0.9", pct: `${bucket.highPct}%`, color: "#818cf8" },
                          { label: "Med 0.7–0.9", pct: `${bucket.medPct}%`, color: "#2dd4bf" },
                          { label: "Low < 0.7", pct: `${bucket.lowPct}%`, color: "#fb7185" },
                        ].map((l) => (
                          <div key={l.label} className="flex items-center gap-2 text-[11px]">
                            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: l.color }} />
                            <span className="text-slate-400">{l.label}</span>
                            <span className="ml-auto text-slate-600">{l.pct}</span>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p className="text-sm text-slate-500">Run analysis to see confidence buckets.</p>
                  )}
                </Card>
              </div>

              <div className="mb-4 grid gap-3 lg:grid-cols-2">
                <Card title="Minority spotlight" titleRight="highest-signal rows">
                  {result ? (
                    <ul className="divide-y divide-white/[0.05]">
                      {result.rows
                        .filter((r) => r.label === "minority")
                        .slice(0, 6)
                        .map((r, i) => (
                          <li key={`${r.id}-${i}`} className="flex items-center gap-2.5 py-2.5 first:pt-0">
                            <MinorityTag kind="focus" />
                            <span className="min-w-0 flex-1 truncate text-[11px] text-slate-400">{r.feedback}</span>
                            <span className="shrink-0 text-[10px] text-slate-600">{r.confidence.toFixed(2)}</span>
                          </li>
                        ))}
                      {result.summary.minority === 0 && (
                        <li className="py-4 text-sm text-slate-500">No minority rows in this mock run.</li>
                      )}
                    </ul>
                  ) : (
                    <p className="text-sm text-slate-500">Run analysis to list minority-tagged feedback.</p>
                  )}
                </Card>

                <Card title={segmentCol ? `Segments · ${segmentCol}` : "Segments"}>
                  {segments.length > 0 ? (
                    <>
                      <div className="mb-4 space-y-2.5">
                        {segments.map((s) => (
                          <div key={s.label}>
                            <div className="mb-1 flex justify-between text-[11px]">
                              <span className="text-slate-400">{s.label}</span>
                              <span className="font-semibold" style={{ color: s.color }}>
                                {s.pct}%
                              </span>
                            </div>
                            <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                              <div
                                className="h-full rounded-full transition-all duration-500"
                                style={{ width: `${s.pct}%`, background: s.color }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-slate-500">
                        Activity spark
                      </p>
                      <Sparkline values={sparks} />
                    </>
                  ) : (
                    <p className="text-sm text-slate-500">
                      Upload CSV and set a segment column under Data mapping, or rely on auto-detect.
                    </p>
                  )}
                </Card>
              </div>

              <Card title="Upload or replace dataset">
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  className={cn(
                    "rounded-xl border border-dashed px-5 py-6 text-center transition-colors",
                    dragOver ? "border-indigo-400/90 bg-indigo-500/10" : "border-indigo-500/35 bg-transparent",
                  )}
                >
                  <label htmlFor="dash-csv-upload" className="block cursor-pointer">
                    <div className="mx-auto mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500/15 text-indigo-300">
                      <Icon name="upload" />
                    </div>
                    <p className="text-xs text-slate-400">Drop a CSV here or click to browse</p>
                    <p className="mt-1 text-[10px] text-slate-600">Text columns suggested automatically · optional rating & date</p>
                    <input
                      id="dash-csv-upload"
                      type="file"
                      accept=".csv"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) void onFile(f);
                        e.target.value = "";
                      }}
                    />
                  </label>
                </div>
              </Card>
            </>
          )}

          {activeNav === "segment" && (
            <div className="space-y-4">
              {!dataset && <p className="text-sm text-slate-500">Upload a CSV first.</p>}
              {dataset && segments.length === 0 && (
                <p className="text-sm text-slate-500">Choose a segment column under Data mapping.</p>
              )}
              {segments.length > 0 && (
                <Card title="Rows per segment">
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={segments} layout="vertical" margin={{ left: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis type="number" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <YAxis
                          dataKey="label"
                          type="category"
                          width={120}
                          stroke="#64748b"
                          tick={{ fill: "#94a3b8", fontSize: 11 }}
                        />
                        <Tooltip contentStyle={chartTooltipStyle} />
                        <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                          {segments.map((s, i) => (
                            <Cell key={i} fill={s.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              )}
            </div>
          )}

          {activeNav === "compare" && (
            <div className="grid gap-4 lg:grid-cols-2">
              {!result && <p className="text-sm text-slate-500 lg:col-span-2">Run analysis to unlock comparisons.</p>}
              {result && (
                <>
                  <Card title="Majority vs minority">
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={88} label>
                            {pieData.map((_, i) => (
                              <Cell key={i} fill={["#2dd4bf", "#fb7185", "#fbbf24"][i % 3]} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={chartTooltipStyle} />
                          <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                  <Card title="Themes (majority + minority)">
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={themeBars} layout="vertical" margin={{ left: 4 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                          <XAxis type="number" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                          <YAxis dataKey="label" type="category" width={132} tick={{ fontSize: 10, fill: "#94a3b8" }} />
                          <Tooltip contentStyle={chartTooltipStyle} />
                          <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                            {themeBars.map((e, i) => (
                              <Cell key={i} fill={e.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </Card>
                  {result.summary.ratingByLabel && result.summary.ratingByLabel.some((x) => x.count > 0) && (
                    <Card title="Average rating by label" className="lg:col-span-2">
                      <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={result.summary.ratingByLabel}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="label" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                            <YAxis stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                            <Tooltip contentStyle={chartTooltipStyle} />
                            <Bar dataKey="avgRating" fill="#818cf8" radius={[8, 8, 0, 0]} name="Avg rating" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </Card>
                  )}
                  {result.summary.trend && result.summary.trend.length > 0 && (
                    <Card title="Minority vs majority over time" className="lg:col-span-2">
                      <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={result.summary.trend}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="period" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                            <YAxis stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                            <Tooltip contentStyle={chartTooltipStyle} />
                            <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
                            <Line type="monotone" dataKey="minority" stroke="#fb7185" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="majority" stroke="#2dd4bf" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </Card>
                  )}
                </>
              )}
            </div>
          )}

          {activeNav === "minority" && result && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Sample minority comments">
                <ul className="space-y-2">
                  {result.rows
                    .filter((r) => r.label === "minority")
                    .slice(0, 12)
                    .map((r, i) => (
                      <li
                        key={`${r.id}-${i}`}
                        className="rounded-xl border border-amber-500/20 bg-amber-500/[0.06] p-3 text-sm text-slate-300"
                      >
                        <span className="text-[10px] font-semibold uppercase tracking-wide text-amber-400/90">
                          {r.theme}
                        </span>
                        <p className="mt-1 line-clamp-5 text-slate-400">{r.feedback}</p>
                      </li>
                    ))}
                  {result.summary.minority === 0 && <li className="text-slate-500">No minority rows.</li>}
                </ul>
              </Card>
              <Card title="Keywords (minority corpus)">
                <div className="h-[420px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={keywordFrequency(
                        result.rows.filter((r) => r.label === "minority").map((r) => r.feedback),
                        20,
                      )}
                      layout="vertical"
                      margin={{ left: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <YAxis dataKey="word" type="category" width={72} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                      <Tooltip contentStyle={chartTooltipStyle} />
                      <Bar dataKey="count" fill="#fb7185" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
          )}
          {activeNav === "minority" && !result && (
            <p className="text-sm text-slate-500">Run analysis to explore minority patterns.</p>
          )}

          {activeNav === "anomalies" && result && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Needs review — sample rows">
                <ul className="divide-y divide-white/[0.06]">
                  {result.rows
                    .filter((r) => r.label === "needs_review")
                    .slice(0, 10)
                    .map((r, i) => (
                      <li key={`${r.id}-${i}`} className="flex items-start gap-2 py-3 first:pt-0">
                        <MinorityTag kind="review" />
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] text-slate-400">{r.feedback}</p>
                          <p className="mt-1 text-[10px] text-slate-600">conf {r.confidence.toFixed(2)}</p>
                        </div>
                      </li>
                    ))}
                  {result.summary.needsReview === 0 && <li className="py-4 text-slate-500">No rows flagged for review.</li>}
                </ul>
              </Card>
              <Card title="Keywords (review queue)">
                <div className="h-[360px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={kwAnomaly} layout="vertical" margin={{ left: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <YAxis dataKey="word" type="category" width={72} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                      <Tooltip contentStyle={chartTooltipStyle} />
                      <Bar dataKey="count" fill="#fbbf24" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
          )}
          {activeNav === "anomalies" && !result && (
            <p className="text-sm text-slate-500">Run analysis to inspect the review queue.</p>
          )}

          {activeNav === "table" && result && (
            <div>
              <ResultsTable rows={result.rows} variant="dark" />
            </div>
          )}
          {activeNav === "table" && !result && <p className="text-sm text-slate-500">Run analysis to open the table.</p>}
        </div>
      </div>
    </div>
  );
}
