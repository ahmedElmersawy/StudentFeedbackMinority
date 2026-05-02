/** Simple keyword buckets for demo “themes” — replace with model output when API exists. */

const BUCKETS: { theme: string; words: string[] }[] = [
  { theme: "Teaching quality", words: ["teacher", "lecture", "professor", "staff", "class"] },
  { theme: "Course content", words: ["course", "material", "syllabus", "slides", "content"] },
  { theme: "Exams & grading", words: ["exam", "grade", "paper", "test", "mark"] },
  { theme: "Labs & practicals", words: ["lab", "practical", "equipment", "experiment"] },
  { theme: "Facilities", words: ["library", "campus", "wifi", "building", "facility"] },
  { theme: "Extracurricular", words: ["club", "sport", "event", "fest", "activity"] },
  { theme: "Workload & stress", words: ["stress", "deadline", "heavy", "overload", "burden"] },
];

export function detectTheme(text: string): string {
  const low = text.toLowerCase();
  for (const b of BUCKETS) {
    if (b.words.some((w) => low.includes(w))) return b.theme;
  }
  return "General";
}

export function topThemes(texts: string[], topN = 5): { name: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const t of texts) {
    const th = detectTheme(t);
    counts.set(th, (counts.get(th) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, topN);
}

export function keywordFrequency(texts: string[], topN = 25): { word: string; count: number }[] {
  const stop = new Set([
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "she", "use", "her", "many", "than", "them", "these", "this", "that", "with", "from", "have", "been", "good", "very", "also", "some", "more", "when", "what", "will", "your", "into", "just", "like", "over", "such", "only", "other", "about", "their", "there", "would", "could",
  ]);
  const freq = new Map<string, number>();
  for (const t of texts) {
    const words = t.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/);
    for (const w of words) {
      if (w.length < 4 || stop.has(w)) continue;
      freq.set(w, (freq.get(w) ?? 0) + 1);
    }
  }
  return [...freq.entries()]
    .map(([word, count]) => ({ word, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, topN);
}
