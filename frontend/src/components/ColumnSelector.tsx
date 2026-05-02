import type { ParsedDataset } from "../types";

type Props = {
  dataset: ParsedDataset;
  feedbackColumns: string[];
  onChangeFeedback: (cols: string[]) => void;
  ratingColumn: string | null;
  onChangeRating: (col: string | null) => void;
  dateColumn: string | null;
  onChangeDate: (col: string | null) => void;
  variant?: "light" | "dark";
};

export function ColumnSelector({
  dataset,
  feedbackColumns,
  onChangeFeedback,
  ratingColumn,
  onChangeRating,
  dateColumn,
  onChangeDate,
  variant = "light",
}: Props) {
  const toggle = (h: string) => {
    if (feedbackColumns.includes(h)) onChangeFeedback(feedbackColumns.filter((x) => x !== h));
    else onChangeFeedback([...feedbackColumns, h]);
  };

  const dark = variant === "dark";

  return (
    <div
      className={
        dark
          ? "rounded-2xl border border-white/[0.08] bg-[#141928] p-6 shadow-[0_1px_0_rgba(255,255,255,0.04)_inset]"
          : "rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      }
    >
      <h3 className={dark ? "text-sm font-semibold uppercase tracking-wide text-indigo-300" : "text-sm font-semibold uppercase tracking-wide text-indigo-600"}>
        Column mapping
      </h3>
      <p className={dark ? "mt-1 text-sm text-slate-500" : "mt-1 text-sm text-slate-500"}>
        Auto-suggestions are pre-filled. Adjust if your file uses different headers — the model API will use the same
        selection later.
      </p>

      <div className="mt-4">
        <p className={dark ? "text-sm font-medium text-slate-200" : "text-sm font-medium text-slate-800"}>Feedback text columns</p>
        <div className="mt-2 flex max-h-40 flex-wrap gap-2 overflow-y-auto">
          {dataset.headers.map((h) => (
            <label
              key={h}
              className={
                dark
                  ? `inline-flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${
                      feedbackColumns.includes(h)
                        ? "border-indigo-400/60 bg-indigo-500/15 text-indigo-100"
                        : "border-white/10 bg-[#0f1117] text-slate-400 hover:border-white/15"
                    }`
                  : `inline-flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-1.5 text-sm ${
                      feedbackColumns.includes(h)
                        ? "border-indigo-500 bg-indigo-50 text-indigo-900"
                        : "border-slate-200 bg-slate-50 text-slate-600 hover:border-slate-300"
                    }`
              }
            >
              <input
                type="checkbox"
                checked={feedbackColumns.includes(h)}
                onChange={() => toggle(h)}
                className={dark ? "rounded border-white/20 bg-[#0f1117] text-indigo-400" : "rounded border-slate-300 text-indigo-600"}
              />
              {h}
            </label>
          ))}
        </div>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <div>
          <label className={dark ? "text-sm font-medium text-slate-200" : "text-sm font-medium text-slate-800"}>Rating column (optional)</label>
          <select
            className={
              dark
                ? "mt-1 w-full rounded-lg border border-white/10 bg-[#0f1117] px-3 py-2 text-sm text-slate-200"
                : "mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
            }
            value={ratingColumn ?? ""}
            onChange={(e) => onChangeRating(e.target.value || null)}
          >
            <option value="">— None —</option>
            {dataset.headers.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={dark ? "text-sm font-medium text-slate-200" : "text-sm font-medium text-slate-800"}>Date column (optional, for trends)</label>
          <select
            className={
              dark
                ? "mt-1 w-full rounded-lg border border-white/10 bg-[#0f1117] px-3 py-2 text-sm text-slate-200"
                : "mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
            }
            value={dateColumn ?? ""}
            onChange={(e) => onChangeDate(e.target.value || null)}
          >
            <option value="">— None —</option>
            {dataset.headers.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
