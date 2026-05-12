import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

function fmtVram(mb: number, status: string): string {
  if (!mb || status === 'stopped' || status === 'failed') return '—'
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

function fmtGpus(ids: number[] | undefined): string {
  if (!ids || ids.length === 0) return '—'
  return ids.length === 1 ? `gpu ${ids[0]}` : `gpu ${ids.join(',')}`
}

function GpuCard({ g }: { g: any }) {
  const pct = (g.memory_used_mb / g.memory_total_mb) * 100
  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div className="label">gpu {g.index}</div>
        <div className="text-mute text-[11px] tnum">{pct.toFixed(0)}%</div>
      </div>
      <div className="flex items-baseline gap-2 tnum">
        <div className="text-3xl font-light tracking-tightish">
          {(g.memory_used_mb / 1024).toFixed(1)}
        </div>
        <div className="text-mute text-[12px]">/ {(g.memory_total_mb / 1024).toFixed(0)} GB</div>
      </div>
      <div className="h-px bg-rule relative overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-accent transition-[width] duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex items-center gap-6 text-mute text-[11px] tnum">
        <span>util {g.gpu_util_pct}%</span>
        <span>{g.power_w} w</span>
      </div>
    </div>
  )
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
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">dashboard</h2>
        <div className="label">{(gpus.data ?? []).length} gpu · {active.length} active</div>
      </header>

      <section className="space-y-6">
        <div className="label">gpus</div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
          {(gpus.data ?? []).map((g: any) => <GpuCard key={g.index} g={g} />)}
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="label">deployments</div>
          <label className="text-mute text-[11px] tracking-wider select-none cursor-pointer hover:text-dim transition-colors">
            <input
              type="checkbox"
              className="mr-2 accent-accent align-middle"
              checked={showAll}
              onChange={e => setShowAll(e.target.checked)}
            />
            show stopped {hiddenCount > 0 && !showAll && (
              <span className="text-accent">({hiddenCount})</span>
            )}
          </label>
        </div>
        <table className="ditable">
          <thead>
            <tr>
              <th>#</th>
              <th>model</th>
              <th>backend</th>
              <th>status</th>
              <th>pin</th>
              <th className="text-right">vram</th>
              <th className="text-right">gpu</th>
            </tr>
          </thead>
          <tbody>
            {visible.length === 0 && (
              <tr>
                <td colSpan={7} className="!py-12 text-center text-mute">
                  no active deployments — load one from <span className="text-dim">models</span>
                </td>
              </tr>
            )}
            {visible.map((d: any) => {
              const m = (models.data ?? []).find((m: any) => m.id === d.model_id)
              return (
                <tr key={d.id} title={d.last_error || undefined}>
                  <td className="text-mute tnum">{d.id}</td>
                  <td>{m?.name ?? '-'}</td>
                  <td className="text-dim">{d.backend}</td>
                  <td>
                    <span className={`dot dot-${d.status}`} />
                    <span className="text-dim">{d.status}</span>
                  </td>
                  <td className={d.pinned ? 'text-accent' : 'text-mute'}>
                    {d.pinned ? '★' : '·'}
                  </td>
                  <td className="text-right tnum">{fmtVram(d.vram_reserved_mb, d.status)}</td>
                  <td className="text-right text-dim tnum">{fmtGpus(d.gpu_ids)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>
    </div>
  )
}
