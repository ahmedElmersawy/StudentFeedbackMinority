/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        // Brand — Hex-inspired violet
        brand: {
          50:  "#f5f3ff",
          100: "#ede9fe",
          200: "#ddd6fe",
          300: "#c4b5fd",
          400: "#a78bfa",
          500: "#7c3aed",
          600: "#6d28d9",
          700: "#5b21b6",
          800: "#4c1d95",
          900: "#3b0764",
        },
        // Surface / neutral scale
        surface: {
          50:  "#fafafa",
          100: "#f4f5f7",
          200: "#e8eaed",
          300: "#d1d5db",
        },
        // Sidebar
        sidebar: "#ffffff",
        // Content background
        canvas: "#f4f5f7",
      },
      boxShadow: {
        card:   "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        "card-md": "0 4px 12px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.04)",
        "card-lg": "0 8px 24px rgba(0,0,0,0.10), 0 2px 6px rgba(0,0,0,0.04)",
        focus:  "0 0 0 3px rgba(124,58,237,0.18)",
      },
      borderRadius: {
        xl:  "12px",
        "2xl": "16px",
        "3xl": "20px",
      },
      fontSize: {
        "2xs": ["10px", "14px"],
        xs:   ["11px", "16px"],
        sm:   ["12px", "18px"],
        base: ["13px", "20px"],
        md:   ["14px", "20px"],
        lg:   ["16px", "24px"],
        xl:   ["18px", "28px"],
        "2xl":["22px", "30px"],
        "3xl":["28px", "36px"],
      },
    },
  },
  plugins: [],
};
