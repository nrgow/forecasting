import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/event_stats_table": "http://localhost:8000",
      "/event_groups": "http://localhost:8000"
    }
  }
});
