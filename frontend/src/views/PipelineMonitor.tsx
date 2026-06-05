import type { Phase, PipelineStageInfo } from "../types";

interface Props {
  stages: PipelineStageInfo[];
  log: string[];
  phase: Phase;
  jobMessage: string;
  jobProgress: { done: number; total: number };
  startTime: number | null;
  filename: string;
  speed?: number;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

const STATUS_COLOR: Record<string, string> = {
  done:    "var(--success)",
  running: "var(--brand)",
  error:   "var(--danger)",
  pending: "var(--text-muted)",
};

const STATUS_BG: Record<string, string> = {
  done:    "var(--success-bg)",
  running: "var(--brand-light)",
  error:   "var(--danger-bg)",
  pending: "var(--surface-3)",
};

export function PipelineMonitor({ stages, log, phase, jobMessage, jobProgress, startTime, filename, speed = 0 }: Props) {
  const isRunning = phase === "uploading" || phase === "running";
  const elapsed   = startTime ? Date.now() - startTime : 0;
  const doneCount = stages.filter(s => s.status === "done").length;
  const pct       = Math.round((doneCount / stages.length) * 100);
  const pct2      = jobProgress.total > 0 ? Math.round((jobProgress.done / jobProgress.total) * 100) : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: "var(--text-primary)", margin: 0 }}>Pipeline Monitor</h1>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
          {filename ? `Processing: ${filename}` : "No pipeline running"}
        </p>
      </div>

      {/* Summary bar */}
      {(() => {
        const eta = speed > 0 && jobProgress.total > jobProgress.done
          ? Math.round((jobProgress.total - jobProgress.done) / speed)
          : null;
        const etaStr = eta == null ? "—" : eta < 60 ? `${eta}s` : `${Math.floor(eta/60)}m ${eta%60}s`;
        return (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(120px,1fr))", gap: 12 }}>
            {[
              { label: "Status",    value: phase === "done" ? "Complete" : isRunning ? "Running" : "Idle", color: phase === "done" ? "var(--success)" : isRunning ? "var(--brand)" : "var(--text-muted)" },
              { label: "Elapsed",   value: startTime ? formatDuration(elapsed) : "—",                       color: "var(--text-primary)" },
              { label: "Stages",    value: `${doneCount}/${stages.length}`,                                 color: "var(--text-primary)" },
              { label: "Speed",     value: speed > 0 ? `${speed.toLocaleString(undefined,{maximumFractionDigits:0})} r/s` : "—", color: speed > 0 ? "var(--success)" : "var(--text-muted)" },
              { label: "ETA",       value: etaStr,                                                          color: eta != null ? "var(--brand)" : "var(--text-muted)" },
            ].map(s => (
              <div key={s.label} className="card" style={{ padding: 16 }}>
                <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)", marginBottom: 8 }}>
                  {s.label}
                </div>
                <div style={{ fontSize: 20, fontWeight: 800, color: s.color, fontVariantNumeric: "tabular-nums" }}>{s.value}</div>
              </div>
            ))}
          </div>
        );
      })()}

      {/* Overall progress */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">Overall Progress</div>
          <span style={{ fontSize: 13, fontWeight: 700, color: "var(--brand)", fontVariantNumeric: "tabular-nums" }}>{pct}%</span>
        </div>
        <div className="card-body">
          <div className="progress-track" style={{ height: 8, marginBottom: 16 }}>
            <div className="progress-bar" style={{ width: `${pct}%`, transition: "width 0.6s ease" }} />
          </div>

          {pct2 != null && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--text-muted)", marginBottom: 6 }}>
                <span>
                  Classification: {jobProgress.done.toLocaleString()} / {jobProgress.total.toLocaleString()} rows
                  {speed > 0 && (
                    <span style={{ color: "var(--success)", marginLeft: 8, fontWeight: 600 }}>
                      · {speed.toLocaleString(undefined, { maximumFractionDigits: 0 })} rows/sec
                    </span>
                  )}
                </span>
                <span style={{ fontWeight: 600, color: "var(--brand)" }}>{pct2}%</span>
              </div>
              <div className="progress-track" style={{ height: 6 }}>
                <div className="progress-bar" style={{ width: `${pct2}%` }} />
              </div>
            </div>
          )}

          {/* Stage timeline */}
          <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
            {stages.map((stage, i) => {
              const dur = (stage.completedAt && stage.startedAt)
                ? formatDuration(stage.completedAt - stage.startedAt)
                : stage.status === "running" && stage.startedAt
                ? formatDuration(Date.now() - stage.startedAt)
                : null;

              return (
                <div key={stage.id} className="stage-row">
                  {/* Connector line */}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 20, flexShrink: 0 }}>
                    <div className={`stage-dot ${stage.status}`} />
                    {i < stages.length - 1 && (
                      <div style={{ width: 2, flex: 1, minHeight: 16, background: stage.status === "done" ? "var(--success)" : "var(--border-2)", opacity: 0.5, marginTop: 4 }} />
                    )}
                  </div>

                  <div style={{ flex: 1, paddingBottom: i < stages.length - 1 ? 8 : 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: STATUS_COLOR[stage.status] }}>
                        {stage.label}
                      </span>
                      <span className="badge" style={{ background: STATUS_BG[stage.status], color: STATUS_COLOR[stage.status], fontSize: 10 }}>
                        {stage.status}
                      </span>
                      {dur && (
                        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>
                          {dur}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{stage.description}</div>
                    {stage.status === "running" && jobMessage && (
                      <div style={{ fontSize: 11, color: "var(--brand)", marginTop: 4, fontStyle: "italic" }}>
                        {jobMessage}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Pipeline logs */}
      {log.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Pipeline Logs</div>
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{log.length} entries</span>
          </div>
          <div className="card-body" style={{ padding: "12px 16px" }}>
            <div className="log-panel">
              {log.map((line, i) => {
                const isOk  = line.includes("complete") || line.includes("done");
                const isWarn = line.includes("warning") || line.includes("warn");
                const isErr  = line.includes("error") || line.includes("fail");
                const isInfo = line.includes("INFO") || line.includes("[pipeline]");
                return (
                  <div key={i} className={`log-line${isOk ? " ok" : isWarn ? " warn" : isErr ? " error" : isInfo ? " info" : ""}`}>
                    {line}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Tips */}
      {!isRunning && phase === "idle" && (
        <div className="card" style={{ border: "1px solid var(--brand-mid)", background: "var(--brand-light)" }}>
          <div className="card-body">
            <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
              <span style={{ fontSize: 22 }}>💡</span>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "var(--brand)", marginBottom: 4 }}>Pipeline Tips</div>
                <ul style={{ margin: 0, padding: "0 0 0 16px", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8 }}>
                  <li>Models are cached after first run — subsequent jobs are much faster</li>
                  <li>DBSCAN runs on an 8,000-row sample for datasets larger than 8k rows</li>
                  <li>Sentence transformer embeddings use GPU when available (cuda:0)</li>
                  <li>All 128 CPUs are used for IsolationForest and NearestNeighbors</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
