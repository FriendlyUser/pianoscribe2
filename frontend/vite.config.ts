/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // PS_API_URL points the dev server's proxy at a locally running API.
  const env = loadEnv(mode, ".", "PS_");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: { "/api": { target: env.PS_API_URL ?? "http://localhost:8000", changeOrigin: true } },
    },
    build: { target: "es2022", sourcemap: false },
    test: { environment: "node", include: ["src/**/*.test.ts"] },
  };
});
