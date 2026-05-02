/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "sans-serif"],
      },
      colors: {
        ink: { 950: "#0b1020", 900: "#121a2e", 800: "#1c2742" },
      },
    },
  },
  plugins: [],
};
