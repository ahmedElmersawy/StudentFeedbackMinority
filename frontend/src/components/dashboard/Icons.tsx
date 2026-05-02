import type { ReactNode } from "react";

export type IconName =
  | "grid"
  | "bar"
  | "circles"
  | "star"
  | "shield"
  | "settings"
  | "download"
  | "play"
  | "upload"
  | "table";

const paths: Record<IconName, ReactNode> = {
  grid: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" fill="none" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" fill="none" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" fill="none" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" fill="none" />
    </>
  ),
  bar: <path d="M12 20V10M18 20V4M6 20v-4" />,
  circles: (
    <>
      <circle cx="9" cy="9" r="5.5" />
      <circle cx="16" cy="16" r="5.5" />
    </>
  ),
  star: <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />,
  shield: <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2" />
    </>
  ),
  download: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />
    </>
  ),
  play: <polygon points="5 3 19 12 5 21 5 3" />,
  upload: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </>
  ),
  table: (
    <>
      <path d="M3 6h18v14H3zM3 12h18M9 6v14" fill="none" />
    </>
  ),
};

export function Icon({ name, className = "shrink-0" }: { name: IconName; className?: string }) {
  return (
    <svg
      className={className}
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}
