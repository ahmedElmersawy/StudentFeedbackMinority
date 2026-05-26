import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/predict": "http://localhost:8000",
      "/upload": "http://localhost:8000",
      "/jobs": "http://localhost:8000",
      "/minority": "http://localhost:8000",
    },
  },
});
