import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// The UI ships as ONE self-contained index.html so it can be embedded into
// ChessEngine-viz with the same .incbin / RCDATA mechanism used for the net.
// One binary, no asset directory to lose.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Inline everything; no code splitting, no separate chunks.
    assetsInlineLimit: 100 * 1024 * 1024,
    chunkSizeWarningLimit: 100 * 1024,
    cssCodeSplit: false,
    reportCompressedSize: false,
  },
  server: {
    port: 5173,
    // During `npm run dev` the UI talks to a running ChessEngine-viz.
    proxy: {
      "/api": "http://127.0.0.1:7777",
    },
  },
});
