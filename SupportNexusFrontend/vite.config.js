import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api/python': {
        // 此地址由 Vite 所在进程访问，可安全使用 Docker 内网服务名。
        // 保留旧变量兼容已有本地启动命令。
        target: process.env.VITE_PYTHON_API_PROXY_TARGET || process.env.VITE_PYTHON_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/python/, '')
      }
    }
  }
})
