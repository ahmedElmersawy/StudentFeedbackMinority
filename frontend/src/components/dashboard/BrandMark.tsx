/** Distinct mark: prism + arc (not a generic square grid). */

export function BrandMark({ className = "" }: { className?: string }) {
  return (
    <svg
      className={className}
      width="32"
      height="32"
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden
    >
      <defs>
        <linearGradient id="bm-a" x1="4" y1="28" x2="28" y2="4" gradientUnits="userSpaceOnUse">
          <stop stopColor="#a5b4fc" />
          <stop offset="1" stopColor="#22d3ee" />
        </linearGradient>
        <linearGradient id="bm-b" x1="16" y1="2" x2="16" y2="30" gradientUnits="userSpaceOnUse">
          <stop stopColor="#818cf8" stopOpacity="0.35" />
          <stop offset="1" stopColor="#2dd4bf" stopOpacity="0.9" />
        </linearGradient>
      </defs>
      <path
        d="M16 3 L28 10 L28 22 L16 29 L4 22 L4 10 Z"
        stroke="url(#bm-a)"
        strokeWidth="1.6"
        fill="url(#bm-b)"
        strokeLinejoin="round"
      />
      <path
        d="M9 17 Q16 11 23 17 Q16 23 9 17"
        stroke="white"
        strokeOpacity="0.85"
        strokeWidth="1.4"
        fill="none"
        strokeLinecap="round"
      />
      <circle cx="16" cy="17" r="2.2" fill="white" fillOpacity="0.95" />
    </svg>
  );
}
