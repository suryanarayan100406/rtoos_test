import { useCallback, useEffect, useState } from 'react'
import { api, type Doctor, type RunSummary } from './api'
import { RunList } from './components/RunList'
import { RunDetail } from './components/RunDetail'
import { DoctorPanel } from './components/DoctorPanel'

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [doctor, setDoctor] = useState<Doctor | null>(null)
  const [version, setVersion] = useState<string>('')
  const [showDoctor, setShowDoctor] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshRuns = useCallback(async () => {
    try {
      const list = await api.listRuns()
      setRuns(list)
      setError(null)
      // auto-select the newest run on first load
      setSelectedId((cur) => cur ?? (list.length ? list[list.length - 1].run_id : null))
    } catch (e) {
      setError(`Cannot reach the DRISHTI API (${(e as Error).message}). Is the server running on :8000?`)
    }
  }, [])

  useEffect(() => {
    api.health().then((h) => setVersion(h.version)).catch(() => {})
    api.doctor().then(setDoctor).catch(() => {})
    refreshRuns()
    // A run advances stage-by-stage on the server; poll so the stage table stays live without WS.
    const t = setInterval(refreshRuns, 5000)
    return () => clearInterval(t)
  }, [refreshRuns])

  const selected = runs.find((r) => r.run_id === selectedId) ?? null

  return (
    <div className="app">
      <header className="topbar">
        <h1>DRISHTI</h1>
        <span className="tag">drone video → georeferenced 3D · offline</span>
        <span className="spacer" />
        {version && <span className="tag">v{version}</span>}
        <button onClick={() => setShowDoctor((s) => !s)}>{showDoctor ? 'Hide' : 'Show'} doctor</button>
      </header>

      <div className="layout">
        <aside className="runs">
          <h2>Runs</h2>
          {error && <div className="error-banner">{error}</div>}
          <RunList runs={runs} selectedId={selectedId} onSelect={setSelectedId} />
        </aside>

        <main className="detail">
          {showDoctor && <DoctorPanel doctor={doctor} />}
          {selected ? (
            <RunDetail run={selected} />
          ) : (
            !error && <div className="empty-state">Select a run to inspect its stages and 3D output.</div>
          )}
        </main>
      </div>
    </div>
  )
}
