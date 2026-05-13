import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

function fmtMb(mb: number | null | undefined): string {
  if (!mb) return '—'
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${mb} MB`
}

export default function Snapshots() {
  const qc = useQueryClient()
  const snaps = useQuery({ queryKey: ['snapshots'], queryFn: api.listSnapshots, refetchInterval: 10000 })
  const [keepN, setKeepN] = useState(2)
  const [maxGb, setMaxGb] = useState<string>('')

  const gc = useMutation({
    mutationFn: () => api.gcSnapshots({
      keep_last_per_model: keepN,
      max_disk_gb: maxGb.trim() ? Number(maxGb) : null,
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['snapshots'] }),
  })
  const del = useMutation({
    mutationFn: (key: string) => api.deleteSnapshot(key),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['snapshots'] }),
  })

  const rows = snaps.data ?? []
  const totalMb = rows.reduce((acc: number, s: any) => acc + (s.size_mb ?? 0), 0)

  return (
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">snapshots</h2>
        <div className="label">
          {rows.length} cached · {fmtMb(totalMb)}
        </div>
      </header>

      <section className="space-y-5">
        <div className="label">garbage collect</div>
        <div className="grid grid-cols-[120px_160px_auto] gap-3 max-w-2xl items-end">
          <div className="space-y-1">
            <div className="text-mute text-[11px]">keep last N</div>
            <input
              className="field font-mono"
              type="number" min={1}
              value={keepN}
              onChange={e => setKeepN(Number(e.target.value) || 1)}
            />
          </div>
          <div className="space-y-1">
            <div className="text-mute text-[11px]">max disk (GB, opt)</div>
            <input
              className="field font-mono"
              placeholder="unlimited"
              value={maxGb}
              onChange={e => setMaxGb(e.target.value)}
            />
          </div>
          <button
            className="btn-primary"
            disabled={gc.isPending}
            onClick={() => gc.mutate()}
          >
            {gc.isPending ? 'collecting…' : 'gc →'}
          </button>
        </div>
        {gc.data && (
          <div className="text-mute text-[11px]">
            evicted {gc.data.evicted ?? 0} snapshot(s)
          </div>
        )}
        {gc.error && (
          <div className="text-err text-[12px]">{(gc.error as Error).message}</div>
        )}
      </section>

      <section className="space-y-4">
        <div className="label">cache</div>
        <table className="ditable">
          <thead>
            <tr>
              <th>key</th>
              <th>engine</th>
              <th>repo</th>
              <th className="text-right">size</th>
              <th className="text-right">hits</th>
              <th>last used</th>
              <th className="text-right"></th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={7} className="!py-12 text-center text-mute">
                  no snapshots yet — first warm load will create one
                </td>
              </tr>
            )}
            {rows.map((s: any) => (
              <tr key={s.id}>
                <td className="font-mono text-[11px] text-dim">{s.key_prefix}…</td>
                <td>{s.engine}</td>
                <td className="text-dim font-mono text-[11px]">{s.hf_repo}</td>
                <td className="text-right tnum">{fmtMb(s.size_mb)}</td>
                <td className="text-right tnum">{s.hit_count ?? 0}</td>
                <td className="text-mute text-[11px]">{s.last_used_at ?? '—'}</td>
                <td className="text-right">
                  <button
                    className="btn-link-danger"
                    onClick={() => del.mutate(s.key)}
                  >
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
