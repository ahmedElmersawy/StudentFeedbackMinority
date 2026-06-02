/**
 * Lightweight SVG chart components — no external dependencies.
 * All charts are self-contained, animated, and responsive.
 */

import { useMemo } from "react";

// ─── Color palette ───────────────────────────────────────────────────────────

export const CHART_COLORS = [
  "#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626",
  "#0891b2", "#9333ea", "#16a34a", "#ca8a04", "#b91c1c",
  "#0284c7", "#7c3aed", "#15803d", "#b45309", "#991b1b",
];

const LABEL_PALETTE: Record<string, string> = {
  Majority_Positive:           "#059669",
  Self_Positive:               "#059669",
  Suggestion_To_Peer:          "#2563eb",
  Self_Suggestion:             "#2563eb",
  Negative_Communication:      "#dc2626",
  Negative_Contribution:       "#b91c1c",
  Negative_Reliability:        "#c2410c",
  Negative_Attitude:           "#9a3412",
  Minority_Peer_Experience:    "#d97706",
  Minority_Student_Experience: "#b45309",
  Self_Minority_Reveal:        "#ca8a04",
  Self_Struggle:               "#9333ea",
};

export function labelColor(label: string, fallbackIdx = 0): string {
  return LABEL_PALETTE[label] ?? CHART_COLORS[fallbackIdx % CHART_COLORS.length];
}

// ─── HorizontalBarChart ───────────────────────────────────────────────────────

interface BarItem { label: string; value: number; color?: string }

export function HorizontalBarChart({
  data,
  height = 32,
  showPct = true,
  maxLabelWidth = 180,
}: {
  data: BarItem[];
  height?: number;
  showPct?: boolean;
  maxLabelWidth?: number;
}) {
  const total = data.reduce((s, d) => s + d.value, 0) || 1;
  const max   = Math.max(...data.map(d => d.value), 1);

  return (
    <div className="space-y-2">
      {data.map((item, i) => {
        const pct = ((item.value / total) * 100).toFixed(1);
        const w   = Math.max((item.value / max) * 100, 2);
        const color = item.color ?? labelColor(item.label, i);
        return (
          <div key={item.label} className="flex items-center gap-3 group">
            <div
              className="shrink-0 truncate text-right text-xs text-gray-500 group-hover:text-gray-700 transition-colors"
              style={{ width: maxLabelWidth, fontSize: 12 }}
              title={item.label}
            >
              {item.label.replace(/_/g, " ")}
            </div>
            <div className="relative flex-1 overflow-hidden rounded-md bg-gray-100" style={{ height }}>
              <div
                className="flex h-full items-center rounded-md px-2.5 transition-all duration-700"
                style={{ width: `${w}%`, background: `${color}22`, borderRight: `3px solid ${color}` }}
              >
                <span className="text-xs font-semibold tabular-nums" style={{ color, fontSize: 11 }}>
                  {item.value.toLocaleString()}
                </span>
              </div>
            </div>
            {showPct && (
              <div className="w-10 shrink-0 text-right text-xs text-gray-400 tabular-nums">{pct}%</div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── VerticalBarChart ─────────────────────────────────────────────────────────

export function VerticalBarChart({
  data,
  svgHeight = 160,
}: {
  data: BarItem[];
  svgHeight?: number;
}) {
  const max = Math.max(...data.map(d => d.value), 1);
  const barW = 100 / Math.max(data.length, 1);

  return (
    <svg width="100%" height={svgHeight} viewBox={`0 0 100 ${svgHeight}`} preserveAspectRatio="none">
      {data.map((item, i) => {
        const h = (item.value / max) * (svgHeight - 24);
        const x = i * barW + barW * 0.15;
        const w = barW * 0.7;
        const y = svgHeight - 24 - h;
        const color = item.color ?? labelColor(item.label, i);
        return (
          <g key={item.label}>
            <rect x={x} y={y} width={w} height={h} fill={color} rx="2" opacity="0.85">
              <animate attributeName="height" from="0" to={h} dur="0.5s" fill="freeze" />
              <animate attributeName="y" from={svgHeight - 24} to={y} dur="0.5s" fill="freeze" />
            </rect>
          </g>
        );
      })}
      <line x1="0" y1={svgHeight - 24} x2="100" y2={svgHeight - 24} stroke="#e5e7eb" strokeWidth="0.5" />
    </svg>
  );
}

// ─── DonutChart ───────────────────────────────────────────────────────────────

interface DonutSlice { label: string; value: number; color: string }

export function DonutChart({
  slices,
  size = 120,
  thickness = 18,
  centerLabel,
  centerSub,
}: {
  slices: DonutSlice[];
  size?: number;
  thickness?: number;
  centerLabel?: string;
  centerSub?: string;
}) {
  const total = slices.reduce((s, d) => s + d.value, 0) || 1;
  const r     = (size - thickness) / 2;
  const c     = size / 2;
  const circ  = 2 * Math.PI * r;

  const arcs = useMemo(() => {
    let offset = 0;
    return slices.map(s => {
      const len = (s.value / total) * circ;
      const arc = { offset: -offset, len, color: s.color };
      offset += len;
      return arc;
    });
  }, [slices, circ, total]);

  return (
    <div className="flex items-center gap-5 flex-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={c} cy={c} r={r} fill="none" stroke="#f3f4f6" strokeWidth={thickness} />
        {arcs.map((arc, i) => (
          <circle
            key={i}
            cx={c} cy={c} r={r}
            fill="none"
            stroke={arc.color}
            strokeWidth={thickness}
            strokeDasharray={`${arc.len} ${circ}`}
            strokeDashoffset={arc.offset}
            transform={`rotate(-90 ${c} ${c})`}
            strokeLinecap="butt"
            style={{ transition: "stroke-dasharray 0.6s ease" }}
          />
        ))}
        {centerLabel && (
          <>
            <text x={c} y={c - 4} textAnchor="middle" fill="#111827" fontSize="14" fontWeight="700">
              {centerLabel}
            </text>
            {centerSub && (
              <text x={c} y={c + 12} textAnchor="middle" fill="#9ca3af" fontSize="9">
                {centerSub}
              </text>
            )}
          </>
        )}
      </svg>
      <div className="space-y-2">
        {slices.map(s => (
          <div key={s.label} className="flex items-center gap-2 text-xs">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: s.color }} />
            <span className="text-gray-600">{s.label}</span>
            <span className="ml-auto font-semibold tabular-nums text-gray-900">
              {s.value.toLocaleString()} ({((s.value / total) * 100).toFixed(0)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── MiniSparkBar (for table cells) ──────────────────────────────────────────

export function SparkBar({ value, max, color = "#7c3aed", height = 20 }: {
  value: number; max: number; color?: string; height?: number;
}) {
  const w = max > 0 ? Math.max((value / max) * 100, 2) : 0;
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1 rounded-sm overflow-hidden bg-gray-100" style={{ height }}>
        <div className="absolute inset-y-0 left-0 transition-all duration-700" style={{ width: `${w}%`, background: `${color}40` }} />
      </div>
      <span className="w-10 shrink-0 text-right text-xs tabular-nums text-gray-600">
        {value.toLocaleString()}
      </span>
    </div>
  );
}

// ─── ConfidenceBar (inline) ───────────────────────────────────────────────────

export function ConfidenceBar({ value }: { value: number }) {
  const pct   = Math.round(value * 100);
  const color = pct >= 90 ? "#059669" : pct >= 70 ? "#d97706" : "#dc2626";
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-gray-200">
        <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="w-7 shrink-0 text-right text-xs tabular-nums" style={{ color }}>{pct}%</span>
    </div>
  );
}

// ─── RadialGauge ─────────────────────────────────────────────────────────────

export function RadialGauge({ value, label, color = "#7c3aed", size = 80 }: {
  value: number; label: string; color?: string; size?: number;
}) {
  const r    = (size - 8) / 2;
  const c    = size / 2;
  const circ = 2 * Math.PI * r;
  const len  = (value / 100) * circ;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={c} cy={c} r={r} fill="none" stroke="#f3f4f6" strokeWidth="8" />
        <circle
          cx={c} cy={c} r={r} fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={`${len} ${circ}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${c} ${c})`}
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
        <text x={c} y={c + 4} textAnchor="middle" fill="#111827" fontSize="12" fontWeight="700">
          {value}%
        </text>
      </svg>
      <span className="mt-1 text-xs text-gray-500">{label}</span>
    </div>
  );
}
