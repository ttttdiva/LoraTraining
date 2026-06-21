import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  build: {
    emptyOutDir: false,
  },
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
});
