import { useEffect, useRef } from 'react'
import * as Cesium from 'cesium'
import 'cesium/Build/Cesium/Widgets/widgets.css'
import { fileUrl, type ExportsDoc } from '../api'

// Renders the run's georeferenced 3D Tiles tileset (the mesh deliverable from s10_export). Fully
// offline: no Cesium Ion token, no external imagery/terrain — only the tileset served by our own API.
// Point-cloud (Potree) output is a different format Cesium can't load directly; it's offered as a
// download in the Deliverables card instead (honest: we don't fake an in-browser point-cloud render).
export function CesiumView({ runId, exports }: { runId: string; exports: ExportsDoc | null }) {
  const containerRef = useRef<HTMLDivElement | null>(null)

  const tiles = exports?.results['3dtiles']
  const tilesetUrl = tiles?.ok && tiles.path ? fileUrl(runId, `${tiles.path}/tileset.json`) : null

  useEffect(() => {
    if (!tilesetUrl || !containerRef.current) return
    let viewer: Cesium.Viewer | undefined
    let cancelled = false

    // baseLayer:false + no geocoder/terrain => zero network calls to Ion/Bing. Widgets that reach
    // out to Ion are disabled so the viewer works with no internet and no access token.
    viewer = new Cesium.Viewer(containerRef.current, {
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
    })
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#12181f')

    Cesium.Cesium3DTileset.fromUrl(tilesetUrl)
      .then((tileset) => {
        if (cancelled || !viewer || viewer.isDestroyed()) return
        viewer.scene.primitives.add(tileset)
        return viewer.zoomTo(tileset)
      })
      .catch((err) => {
        if (!cancelled) console.error('Failed to load tileset', err)
      })

    return () => {
      cancelled = true
      if (viewer && !viewer.isDestroyed()) viewer.destroy()
    }
  }, [tilesetUrl])

  if (!tilesetUrl) {
    return (
      <div className="cesium-wrap">
        <div className="overlay">{overlayMessage(exports)}</div>
      </div>
    )
  }
  return <div className="cesium-wrap" ref={containerRef} />
}

function overlayMessage(exports: ExportsDoc | null): string {
  if (!exports) return '3D output has not been exported yet — run the pipeline through s10_export.'
  const reason = exports.results['3dtiles']?.reason
  const base = 'No 3D Tiles tileset for this run, so there is nothing to render geospatially in-browser.'
  const tail = ' Mesh formats (GLB / OBJ / PLY) are downloadable below.'
  return reason ? `${base} (${reason})${tail}` : base + tail
}
