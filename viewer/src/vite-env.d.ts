/// <reference types="vite/client" />

interface Window {
  // Set in index.html so CesiumJS can locate its static assets offline.
  CESIUM_BASE_URL?: string
}
