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

function cn(...xs: (string | false | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

export function UploadPanel({ phase, jobMessage, jobProgress, onUpload, onReset }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [opts, setOpts] = useState<UploadOptions>({
    feedbackMode: "auto",
    modelDir: "",
    anonymize: true,
    includeMinority: true,
    includeMismatch: true,
    confidenceThreshold: 0.65,
  });

  const handleFile = useCallback((f: File) => {
    if (!f.name.toLowerCase().endsWith(".csv")) {
      alert("Please upload a .csv file.");
      return;
    }
    setFile(f);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const running = phase === "uploading" || phase === "running";
  const pct =
    jobProgress.total > 0
      ? Math.round((jobProgress.done / jobProgress.total) * 100)
      : null;

  return (
    <div className="space-y-5">
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={cn(
          "rounded-2xl border-2 border-dashed px-6 py-10 text-center transition-colors",
          dragOver
            ? "border-indigo-400 bg-indigo-500/10"
            : file
            ? "border-emerald-500/50 bg-emerald-500/5"
            : "border-white/15 bg-white/[0.02] hover:border-white/25",
        )}
      >
        <label htmlFor="csv-upload" className="block cursor-pointer">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-500/15">
            <svg className="h-6 w-6 text-indigo-300" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10 2a1 1 0 011 1v7.586l2.293-2.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L9 10.586V3a1 1 0 011-1z" />
              <path d="M3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" />
            </svg>
          </div>
          {file ? (
            <>
              <p className="text-sm font-semibold text-emerald-300">{file.name}</p>
              <p className="mt-0.5 text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB · Click to replace</p>
            </>
          ) : (
            <>
              <p className="text-sm font-medium text-slate-300">Drop a CSV here or click to browse</p>
              <p className="mt-1 text-xs text-slate-600">
                Supports any feedback CSV · headerless CATME · rated datasets
              </p>
            </>
          )}
          <input
            id="csv-upload"
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }}
          />
        </label>
      </div>

      {/* Options */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <label className="space-y-1">
          <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
            Feedback mode
          </span>
          <select
            value={opts.feedbackMode}
            onChange={(e) => setOpts((o) => ({ ...o, feedbackMode: e.target.value as UploadOptions["feedbackMode"] }))}
            className="w-full rounded-lg border border-white/[0.1] bg-[#0f1117] px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          >
            <option value="auto">Auto-detect</option>
            <option value="student_to_student">Student → Student (CATME)</option>
            <option value="student_to_professor">Student → Professor</option>
          </select>
        </label>
        <label className="space-y-1">
          <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
            Review threshold
          </span>
          <input
            type="number"
            min={0.1}
            max={1.0}
            step={0.05}
            value={opts.confidenceThreshold}
            onChange={(e) => setOpts((o) => ({ ...o, confidenceThreshold: parseFloat(e.target.value) }))}
            className="w-full rounded-lg border border-white/[0.1] bg-white/[0.04] px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          />
        </label>
        <label className="space-y-1 sm:col-span-2">
          <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500">
            Model directory (optional)
          </span>
          <input
            type="text"
            placeholder="auto-selected based on mode"
            value={opts.modelDir}
            onChange={(e) => setOpts((o) => ({ ...o, modelDir: e.target.value }))}
            className="w-full rounded-lg border border-white/[0.1] bg-white/[0.04] px-3 py-1.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-400"
          />
        </label>
      </div>

      <div className="flex flex-wrap gap-4 text-sm">
        {([
          ["anonymize", "Anonymize names"],
          ["includeMinority", "Minority detection"],
          ["includeMismatch", "Mismatch detection"],
        ] as const).map(([key, label]) => (
          <label key={key} className="flex cursor-pointer items-center gap-2 text-slate-400">
            <input
              type="checkbox"
              checked={opts[key] as boolean}
              onChange={(e) => setOpts((o) => ({ ...o, [key]: e.target.checked }))}
              className="h-4 w-4 rounded border-white/20 bg-white/5 accent-indigo-500"
            />
            {label}
          </label>
        ))}
      </div>

      {/* Progress bar */}
      {running && (
        <div className="space-y-1.5">
          <p className="text-xs text-slate-500">{jobMessage}</p>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
            <div
              className="h-full rounded-full bg-indigo-500 transition-all duration-300"
              style={{ width: pct != null ? `${pct}%` : "100%", animation: pct == null ? "pulse 1.5s ease-in-out infinite" : undefined }}
            />
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={() => file && onUpload(file, opts)}
          disabled={!file || running}
          className="flex-1 rounded-xl bg-indigo-500 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-40 transition-colors"
        >
          {running ? "Processing…" : "Run Analysis"}
        </button>
        {phase === "done" || phase === "error" ? (
          <button
            onClick={onReset}
            className="rounded-xl border border-white/[0.1] px-4 py-2.5 text-sm font-medium text-slate-400 hover:text-slate-200 transition-colors"
          >
            Reset
          </button>
        ) : null}
      </div>

      {phase === "error" && (
        <p className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          {jobMessage}
        </p>
      )}
    </div>
  );
}
