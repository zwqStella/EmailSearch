import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev the React app runs on :5173 and proxies /api → the FastAPI server on :8765.
// In prod we run `npm run build`; the FastAPI server serves the static dist/ folder.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
