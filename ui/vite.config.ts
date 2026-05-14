import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/serve_engine/ui',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  server: {
    proxy: {
      '/admin': 'http://127.0.0.1:11500',
      '/v1':    'http://127.0.0.1:11500',
    },
  },
})
