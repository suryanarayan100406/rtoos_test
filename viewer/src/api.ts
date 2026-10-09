// Typed client for the DRISHTI API (server/app.py). Same-origin relative URLs: Vite proxies /api to
// the uvicorn server in dev; FastAPI serves this SPA itself in production. Nothing here invents data —
// every field mirrors what the server derives from the on-disk manifest.

export interface StageSummary {
  name: string
  status: string // pending | running | done | degraded | failed
  environment: string | null // "local" | "cloud-t4"
  wall_seconds: number | null
  confidence: number | null
  degraded_reason: string | null
  error: string | null
}

export interface RunSummary {
  run_id: string
  dataset_name: string | null
  drishti_version: string | null
  state: string // empty | partial | running | failed | done
  crs: { epsg?: number | null } | null
  n_stages_done: number
  n_stages_failed: number
  n_stages_total: number
  stages: StageSummary[]
  launch_error?: string
}

export interface Doctor {
  compute: {
    cpu_count: number
    ram_gb: number
    cuda: boolean
    cuda_name: string | null
    cuda_vram_gb: number
    openvino: boolean
    openvino_devices: string[]
    is_cloud_gpu: boolean
  }
  modules: Record<string, boolean>
  stages: Record<string, { ready: boolean; detail: string }>
  pipeline_order: string[]
}

export interface ExportResult {
  ok: boolean
  path: string | null // bundle-relative path when ok
  reason: string | null // why it was skipped when not ok
}

export interface ExportsDoc {
  epsg: number
  results: Record<string, ExportResult>
  must_formats: string[]
  must_ok: string[]
  skipped_must: string[]
}

export interface TileState {
  tile_id: string
  ix?: number
  iy?: number
  status: string
  geometry_class: string
  confidence: number | null
  candidate_frames?: number
  frames_integrated?: number
  points_emitted?: number
  z_source?: string
  reason: string | null
}

export interface TileStatesDoc {
  tiles: TileState[]
  geometry_classes: string[]
  authoritative_classes: string[]
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`)
  return (await r.json()) as T
}

export const api = {
  health: () => getJSON<{ status: string; version: string; runs_dir: string }>('/api/health'),
  doctor: () => getJSON<Doctor>('/api/doctor'),
  listRuns: () => getJSON<{ runs: RunSummary[] }>('/api/runs').then((d) => d.runs),
  getRun: (id: string) => getJSON<RunSummary>(`/api/runs/${encodeURIComponent(id)}`),

  // exports.json is written by the s10_export stage; absent until that stage runs (treated as null).
  async getExports(id: string): Promise<ExportsDoc | null> {
    const r = await fetch(fileUrl(id, 's10_export/exports.json'))
    if (r.status === 404) return null
    if (!r.ok) throw new Error(`${r.status} loading exports.json`)
    return (await r.json()) as ExportsDoc
  },
  async getTileStates(id: string): Promise<TileStatesDoc | null> {
    const r = await fetch(fileUrl(id, 's7_dense/tile_states.json'))
    if (r.status === 404) return null
    if (!r.ok) throw new Error(`${r.status} loading tile evidence`)
    return (await r.json()) as TileStatesDoc
  },
}

// URL for a bundle artifact. Normalises Windows backslashes that may appear in manifest-recorded paths.
export function fileUrl(runId: string, relPath: string): string {
  const clean = relPath.replace(/\\/g, '/').replace(/^\/+/, '')
  return `/api/runs/${encodeURIComponent(runId)}/files/${clean}`
}

export function reportUrl(runId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/report`
}
