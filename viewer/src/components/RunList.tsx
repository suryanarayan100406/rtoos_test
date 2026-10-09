import type { RunSummary } from '../api'

export function RunList({
  runs,
  selectedId,
  onSelect,
}: {
  runs: RunSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  if (!runs.length) {
    return <div className="muted" style={{ padding: '8px 6px' }}>No runs yet.</div>
  }
  // newest last from the API; show newest first here
  const ordered = [...runs].reverse()
  return (
    <div>
      {ordered.map((r) => (
        <div
          key={r.run_id}
          className={`run-item${r.run_id === selectedId ? ' selected' : ''}`}
          onClick={() => onSelect(r.run_id)}
        >
          <div className="rid">{r.run_id}</div>
          <div className="meta">
            <span>{r.dataset_name ?? 'unknown dataset'}</span>
            <span className={`pill ${r.state}`}>{r.state}</span>
          </div>
          <div className="meta">
            <span>
              {r.n_stages_done}/{r.n_stages_total} stages
              {r.n_stages_failed > 0 && ` · ${r.n_stages_failed} failed`}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}
