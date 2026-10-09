import type { Doctor } from '../api'

// Renders the exact same hardware/dependency truth the CLI `doctor` prints — measured, never assumed.
export function DoctorPanel({ doctor }: { doctor: Doctor | null }) {
  if (!doctor) return <div className="card muted">Doctor unavailable (API not reachable).</div>
  const c = doctor.compute
  return (
    <div className="card">
      <h3>Doctor — detected compute</h3>
      <div className="kv">
        <span className="k">CPU cores</span>
        <span>{c.cpu_count}</span>
        <span className="k">RAM</span>
        <span>{c.ram_gb} GB</span>
        <span className="k">CUDA</span>
        <span>{c.cuda ? `yes — ${c.cuda_name} (${c.cuda_vram_gb} GB)` : 'no discrete GPU'}</span>
        <span className="k">OpenVINO</span>
        <span>{c.openvino ? c.openvino_devices.join(', ') || 'yes' : 'not available'}</span>
        <span className="k">Tier</span>
        <span>{c.is_cloud_gpu ? 'cloud-t4 (GPU present)' : 'local ground station (CPU)'}</span>
      </div>

      <h3 style={{ marginTop: 14 }}>Stage readiness</h3>
      <table className="stages">
        <tbody>
          {doctor.pipeline_order.map((name) => {
            const s = doctor.stages[name]
            if (!s) return null
            return (
              <tr key={name}>
                <td style={{ fontFamily: 'ui-monospace, monospace' }}>{name}</td>
                <td>
                  <span className={`pill ${s.ready ? 'done' : 'pending'}`}>{s.ready ? 'ready' : 'blocked'}</span>
                </td>
                <td className="muted">{s.detail}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
