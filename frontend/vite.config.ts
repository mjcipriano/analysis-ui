import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The production build is written into the Python package so that
// `analysis-ui` serves the app directly.
export default defineConfig({
  plugins: [react()],
  build: { outDir: '../backend/analysis_ui/static', emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8765' } },
})
