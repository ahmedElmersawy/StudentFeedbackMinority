import { useCallback, useState } from "react";
import type { FeedbackMode } from "../api/client";
import type { Phase } from "../types";

interface Props {
  phase: Phase;
  jobMessage: string;
  jobProgress: { done: number; total: number };
  onUpload: (file: File, opts: UploadOptions) => void;
  onReset: () => void;
}

export interface UploadOptions {
  feedbackMode: FeedbackMode | "auto";
  modelDir: string;
  anonymize: boolean;
  includeMinority: boolean;
  includeMismatch: boolean;
  confidenceThreshold: number;
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", userSelect: "none" }}>
      <div className="toggle" onClick={() => onChange(!checked)}>
        <div className="toggle-track" style={{ background: checked ? "var(--brand)" : undefined }} />
        <div className="toggle-thumb" style={{ transform: checked ? "translateX(16px)" : undefined }} />
      </div>
      <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>{label}</span>
    </label>
  );
}

export function UploadPanel({ phase, jobMessage, jobProgress, onUpload, onReset }: Props) {
  const [drag, setDrag]   = useState(false);
  const [file, setFile]   = useState<File | null>(null);
  const [opts, setOpts]   = useState<UploadOptions>({
    feedbackMode: "auto", modelDir: "", anonymize: true,
    includeMinority: true, includeMismatch: true, confidenceThreshold: 0.65,
  });

  const handleFile = useCallback((f: File) => {
    if (!f.name.toLowerCase().endsWith(".csv")) { alert("Please upload a .csv file."); return; }
    setFile(f);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const running = phase === "uploading" || phase === "running";
  const pct     = jobProgress.total > 0 ? Math.round((jobProgress.done / jobProgress.total) * 100) : null;
  const sizeLabel = file
    ? file.size > 1024 * 1024
      ? `${(file.size / 1024 / 1024).toFixed(1)} MB`
      : `${(file.size / 1024).toFixed(1)} KB`
    : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Drop zone */}
      <div
        className={`dropzone${drag ? " drag-over" : file ? " has-file" : ""}`}
        onDragOver={e => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        <label htmlFor="csv-upload" style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "32px 24px", cursor: "pointer" }}>
          {file ? (
            <>
              <div style={{ width: 48, height: 48, borderRadius: 14, background: "var(--success-bg)", border: "1px solid #bbf7d0", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 12 }}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="1.5">
                  <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--success)" }}>{file.name}</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>{sizeLabel} · Click to replace</div>
            </>
          ) : (
            <>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: "var(--brand-light)", border: "1px solid var(--brand-mid)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 14, transition: "transform 0.15s", transform: drag ? "scale(1.12)" : "scale(1)" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--brand)" strokeWidth="1.5">
                  <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
                </svg>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)" }}>
                {drag ? "Drop it!" : "Drop your CSV here or click to browse"}
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 6 }}>
                Supports CATME peer feedback and course evaluation CSVs
              </div>
              <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
                {["CSV", "UTF-8", "Up to 1M rows"].map(t => (
                  <span key={t} style={{ fontSize: 10, background: "var(--surface-3)", border: "1px solid var(--border)", borderRadius: 20, padding: "2px 8px", color: "var(--text-muted)" }}>{t}</span>
                ))}
              </div>
            </>
          )}
          <input id="csv-upload" type="file" accept=".csv" aria-label="Upload CSV file" style={{ display: "none" }}
            onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }} />
        </label>
      </div>

      {/* Options */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))", gap: 12 }}>
        <div>
          <label className="field-label">Feedback Mode</label>
          <select
            className="select-field"
            value={opts.feedbackMode}
            onChange={e => setOpts(o => ({ ...o, feedbackMode: e.target.value as UploadOptions["feedbackMode"] }))}
          >
            <option value="auto">Auto-detect</option>
            <option value="student_to_student">Student → Student (CATME)</option>
            <option value="student_to_professor">Student → Professor</option>
          </select>
        </div>
        <div>
          <label className="field-label">Review Threshold: {opts.confidenceThreshold.toFixed(2)}</label>
          <div style={{ display: "flex", alignItems: "center", gap: 10, height: 38 }}>
            <input
              type="range" min={0.1} max={1.0} step={0.05}
              value={opts.confidenceThreshold}
              onChange={e => setOpts(o => ({ ...o, confidenceThreshold: parseFloat(e.target.value) }))}
              style={{ flex: 1, accentColor: "var(--brand)" }}
            />
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <Toggle checked={opts.anonymize}      onChange={v => setOpts(o => ({ ...o, anonymize: v }))}      label="Anonymize names" />
        <Toggle checked={opts.includeMinority} onChange={v => setOpts(o => ({ ...o, includeMinority: v }))} label="Minority detection" />
        <Toggle checked={opts.includeMismatch} onChange={v => setOpts(o => ({ ...o, includeMismatch: v }))} label="Mismatch detection" />
      </div>

      {/* Progress */}
      {running && (
        <div style={{ background: "var(--brand-light)", border: "1px solid var(--brand-mid)", borderRadius: 10, padding: "12px 16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--brand)", fontWeight: 500, marginBottom: 8 }}>
            <span>{jobMessage}</span>
            {pct != null && <span style={{ fontVariantNumeric: "tabular-nums" }}>{pct}%</span>}
          </div>
          <div className="progress-track" style={{ height: 6 }}>
            {pct != null
              ? <div className="progress-bar" style={{ width: `${pct}%` }} />
              : <div className="progress-bar" style={{ width: "30%", animation: "none", background: "linear-gradient(90deg, var(--brand), #a855f7)" }} />
            }
          </div>
          {jobProgress.total > 0 && (
            <div style={{ fontSize: 11, color: "var(--brand)", marginTop: 6, fontVariantNumeric: "tabular-nums" }}>
              {jobProgress.done.toLocaleString()} / {jobProgress.total.toLocaleString()} rows
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 10 }}>
        <button
          className="btn btn-primary btn-lg"
          style={{ flex: 1 }}
          disabled={!file || running}
          onClick={() => file && onUpload(file, opts)}
        >
          {running ? (
            <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 14, height: 14, border: "2px solid rgba(255,255,255,0.4)", borderTopColor: "white", borderRadius: "50%", display: "inline-block", animation: "spin 0.8s linear infinite" }} />
              Processing…
            </span>
          ) : "Run Analysis"}
        </button>
        {(phase === "done" || phase === "error") && (
          <button className="btn btn-secondary btn-lg" onClick={onReset}>Reset</button>
        )}
      </div>

      {phase === "error" && (
        <div style={{ background: "var(--danger-bg)", border: "1px solid #fca5a5", borderRadius: 10, padding: "12px 16px", fontSize: 13, color: "var(--danger)" }}>
          {jobMessage}
        </div>
      )}
    </div>
  );
}
