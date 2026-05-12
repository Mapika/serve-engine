import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

const STATUS_COLORS: Record<string, string> = {
  ready: 'bg-green-100 text-green-800',
  loading: 'bg-yellow-100 text-yellow-800',
  stopped: 'bg-gray-100 text-gray-600',
  failed: 'bg-red-100 text-red-800',
}

function fmtVram(mb: number, status: string): string {
  // KV/weight reservation is only meaningful while the engine is up. After
  // stop/failure the row's reservation is stale — show a dash rather than 0.
  if (!mb || status === 'stopped' || status === 'failed') return '—'
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

function fmtGpus(ids: number[] | undefined): string {
  if (!ids || ids.length === 0) return '—'
  // Prefix with "GPU" so a single index doesn't read like a count.
  return ids.length === 1 ? `GPU ${ids[0]}` : `GPUs ${ids.join(',')}`
}

export default function Dashboard() {
  const deps = useQuery({ queryKey: ['deps'], queryFn: api.listDeployments, refetchInterval: 2000 })
  const gpus = useQuery({ queryKey: ['gpus'], queryFn: api.listGpus, refetchInterval: 2000 })
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels, refetchInterval: 5000 })
  const [showAll, setShowAll] = useState(false)

  const all = deps.data ?? []
  const active = all.filter((d: any) => d.status === 'ready' || d.status === 'loading')
  const visible = showAll ? all : active
  const hiddenCount = all.length - visible.length

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold">Dashboard</h2>

      <section>
        <h3 className="text-lg font-semibold mb-2">GPUs</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {(gpus.data ?? []).map((g: any) => (
            <div key={g.index} className="bg-white rounded shadow p-4">
              <div className="text-sm text-gray-500">GPU {g.index}</div>
              <div className="text-xl font-mono">
                {(g.memory_used_mb / 1024).toFixed(1)} / {(g.memory_total_mb / 1024).toFixed(0)} GB
              </div>
              <div className="text-sm text-gray-500 mt-2">util {g.gpu_util_pct}% • {g.power_w} W</div>
              <div className="mt-2 h-2 bg-gray-200 rounded overflow-hidden">
                <div
                  className="h-full bg-blue-500"
                  style={{ width: `${(g.memory_used_mb / g.memory_total_mb) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold">Deployments</h3>
          <label className="text-sm text-gray-600 select-none">
            <input
              type="checkbox"
              className="mr-1"
              checked={showAll}
              onChange={e => setShowAll(e.target.checked)}
            />
            show stopped / failed{hiddenCount > 0 && !showAll ? ` (${hiddenCount} hidden)` : ''}
          </label>
        </div>
        <table className="min-w-full bg-white shadow rounded text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="text-left p-2">ID</th><th className="text-left p-2">Model</th>
              <th className="text-left p-2">Backend</th><th className="text-left p-2">Status</th>
              <th className="text-left p-2">Pin</th><th className="text-left p-2">VRAM</th>
              <th className="text-left p-2">GPU</th>
            </tr>
          </thead>
          <tbody>
            {visible.length === 0 && (
              <tr>
                <td className="p-4 text-center text-gray-500" colSpan={7}>
                  no active deployments — load one from Models
                </td>
              </tr>
            )}
            {visible.map((d: any) => {
              const m = (models.data ?? []).find((m: any) => m.id === d.model_id)
              const cls = STATUS_COLORS[d.status] ?? STATUS_COLORS.stopped
              return (
                <tr
                  key={d.id}
                  className="border-t"
                  title={d.last_error || undefined}
                >
                  <td className="p-2">{d.id}</td>
                  <td className="p-2 font-mono">{m?.name ?? '-'}</td>
                  <td className="p-2">{d.backend}</td>
                  <td className="p-2">
                    <span className={`inline-block px-2 py-0.5 rounded text-xs ${cls}`}>
                      {d.status}
                    </span>
                  </td>
                  <td className="p-2">{d.pinned ? '★' : '-'}</td>
                  <td className="p-2 font-mono">{fmtVram(d.vram_reserved_mb, d.status)}</td>
                  <td className="p-2 font-mono">{fmtGpus(d.gpu_ids)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>
    </div>
  )
}
