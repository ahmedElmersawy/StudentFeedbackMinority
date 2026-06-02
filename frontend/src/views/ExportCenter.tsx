import type { AnalysisResult } from "../types";
import { downloadResultsCsv, downloadMinorityCsv, downloadLabelCsv } from "../api/client";

interface Props {
  result: AnalysisResult;
  jobId: string;
  onToast: (msg: string) => void;
}

function ExportCard({
  icon, title, desc, count, actions,
}: {
  icon: string;
  title: string;
  desc: string;
  count?: string;
  actions: { label: string; onClick: () => void; primary?: boolean }[];
}) {
  return (
    <div className="card card-hover" style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
        <div style={{ width: 44, height: 44, borderRadius: 12, background: "var(--brand-light)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, flexShrink: 0 }}>
          {icon}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>{title}</span>
            {count && <span className="badge badge-slate">{count}</span>}
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>{desc}</p>
          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            {actions.map(a => (
              <button key={a.label} className={`btn btn-sm ${a.primary ? "btn-primary" : "btn-secondary"}`} onClick={a.onClick}>
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 12V4M4 8l4 4 4-4M2 14h12"/></svg>
                {a.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function downloadJson(data: object, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
}

export function ExportCenter({ result, jobId, onToast }: Props) {
  const labels = Object.keys(result.summary.label_distribution);

  function downloadSummaryJson() {
    downloadJson({
      generated_at: new Date().toISOString(),
      summary: result.summary,
      label_count: labels.length,
    }, "feedback_atlas_summary.json");
    onToast("Downloading summary JSON…");
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", margin: 0 }}>Export Center</h1>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
          Download results in CSV or JSON format · {result.summary.total.toLocaleString()} total records
        </p>
      </div>

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {[
          { label: "Total Records",    value: result.summary.total.toLocaleString(),     color: "var(--brand)" },
          { label: "Minority Records", value: result.summary.minority.toLocaleString(),  color: "var(--warning)" },
          { label: "Labels Available", value: labels.length.toString(),                  color: "var(--info)" },
          { label: "Mismatch Records", value: result.summary.mismatch.toLocaleString(),  color: "#c2410c" },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: 16, textAlign: "center" }}>
            <div style={{ fontSize: 24, fontWeight: 800, color: s.color, fontVariantNumeric: "tabular-nums" }}>{s.value}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Export cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <h2 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>Full Dataset Exports</h2>

        <ExportCard
          icon="📊"
          title="Full Analysis Results"
          desc="All records with prediction labels, confidence scores, minority flags, and mismatch indicators."
          count={`${result.summary.total.toLocaleString()} rows`}
          actions={[
            { label: "Download CSV", primary: true, onClick: () => { downloadResultsCsv(jobId, "feedback_atlas_results.csv"); onToast("Downloading full CSV…"); } },
            { label: "Download JSON", onClick: () => { downloadSummaryJson(); } },
          ]}
        />

        {result.summary.minority > 0 && (
          <ExportCard
            icon="⚑"
            title="Minority Patterns Only"
            desc="Rows flagged as minority patterns by IsolationForest, DBSCAN, or keyword detection."
            count={`${result.summary.minority.toLocaleString()} rows`}
            actions={[
              { label: "Download CSV", primary: true, onClick: () => { downloadMinorityCsv(jobId, "minority_results.csv"); onToast("Downloading minority CSV…"); } },
            ]}
          />
        )}

        {result.summary.mismatch > 0 && (
          <ExportCard
            icon="↯"
            title="Mismatch Records"
            desc="Records where the numeric rating contradicts the text sentiment — the most actionable feedback."
            count={`${result.summary.mismatch.toLocaleString()} rows`}
            actions={[
              {
                label: "Download CSV", primary: true, onClick: () => {
                  const mismatchRows = result.rows.filter(r => Boolean(r.mismatch_flag));
                  const cols = ["text", "prediction", "confidence", "mismatch_type"];
                  const csv = [cols.join(","), ...mismatchRows.map(r => cols.map(c => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(","))].join("\n");
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
                  a.download = "mismatch_records.csv";
                  a.click();
                  onToast("Downloading mismatch CSV…");
                }
              },
            ]}
          />
        )}

        {result.summary.needs_review > 0 && (
          <ExportCard
            icon="⚠"
            title="Review Queue"
            desc="Low-confidence predictions that need human verification before use in retraining."
            count={`${result.summary.needs_review.toLocaleString()} rows`}
            actions={[
              {
                label: "Download CSV", primary: true, onClick: () => {
                  const reviewRows = result.rows.filter(r => Boolean(r.needs_review));
                  const cols = ["text", "prediction", "confidence"];
                  const csv = [cols.join(","), ...reviewRows.map(r => cols.map(c => `"${String(r[c] ?? "").replace(/"/g, '""')}"`).join(","))].join("\n");
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
                  a.download = "review_queue.csv";
                  a.click();
                  onToast("Downloading review queue CSV…");
                }
              },
            ]}
          />
        )}
      </div>

      {/* Per-label exports */}
      <div>
        <h2 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", margin: "0 0 12px" }}>
          Per-Label Exports
        </h2>
        <div className="card" style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Label</th>
                <th>Records</th>
                <th>Share</th>
                <th>Avg Confidence</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.summary.label_distribution)
                .sort((a, b) => b[1] - a[1])
                .map(([label, count]) => (
                  <tr key={label}>
                    <td>
                      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: `hsl(${label.charCodeAt(0) * 11 % 360},65%,52%)`, flexShrink: 0 }} />
                        <span style={{ fontWeight: 500 }}>{label.replace(/_/g, " ")}</span>
                      </span>
                    </td>
                    <td style={{ fontVariantNumeric: "tabular-nums" }}>{count.toLocaleString()}</td>
                    <td>
                      <span className="badge badge-slate">
                        {((count / result.summary.total) * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>
                      {(() => {
                        const rows = result.rows.filter(r => r.prediction === label);
                        const avg  = rows.reduce((s, r) => s + (typeof r.confidence === "number" ? r.confidence : 0), 0) / Math.max(1, rows.length);
                        return <span style={{ fontVariantNumeric: "tabular-nums", color: avg >= 0.8 ? "var(--success)" : "var(--warning)" }}>{(avg * 100).toFixed(1)}%</span>;
                      })()}
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => { downloadLabelCsv(jobId, label); onToast(`Downloading ${label}…`); }}
                      >
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 12V4M4 8l4 4 4-4M2 14h12"/></svg>
                        CSV
                      </button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Report */}
      <ExportCard
        icon="📋"
        title="Analysis Report (JSON)"
        desc="Complete metadata including label distribution, minority breakdown, confidence stats, and mode detection."
        actions={[
          { label: "Download JSON Report", primary: true, onClick: () => {
            downloadJson({
              generated_at: new Date().toISOString(),
              summary: result.summary,
              label_distribution: result.summary.label_distribution,
              minority_categories: result.summary.minority_category_breakdown,
              catme_subtypes: result.summary.catme_subtype_distribution,
            }, "feedback_atlas_report.json");
            onToast("Downloading report JSON…");
          }},
        ]}
      />
    </div>
  );
}
