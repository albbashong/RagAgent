import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 9012,
    host: true,
    proxy: {
      '/ws': {
        target: 'ws://localhost:9013', // 실제 bridge_server.py
        ws: true,
        changeOrigin: true
      }
    }
  }
})