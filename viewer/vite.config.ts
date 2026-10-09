import path from 'node:path'
import { createRequire } from 'node:module'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from 'vite-plugin-static-copy'

// CesiumJS ships its Workers/Assets/Widgets/ThirdParty as static files that must be served under a
// known base path (window.CESIUM_BASE_URL, set in index.html). Copy them into dist/cesium so the
// build is fully self-contained and OFFLINE-CAPABLE — no CDN, no Cesium Ion, at build or run time.
const require = createRequire(import.meta.url)
const cesiumBuild = path.join(path.dirname(require.resolve('cesium/package.json')), 'Build', 'Cesium')

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: ['Workers', 'Assets', 'Widgets', 'ThirdParty'].map((dir) => ({
        src: path.join(cesiumBuild, dir).replace(/\\/g, '/'),
        dest: 'cesium',
      })),
    }),
  ],
  server: {
    port: 5173,
    // In dev the API runs separately (uvicorn on :8000). Proxy /api there so the SPA uses same-origin
    // relative URLs in every environment (in production FastAPI serves this build at / itself).
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
