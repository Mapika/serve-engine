import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Dashboard() {
  const deps = useQuery({ queryKey: ['deps'], queryFn: api.listDeployments, refetchInterval: 2000 })
  const gpus = useQuery({ queryKey: ['gpus'], queryFn: api.listGpus, refetchInterval: 2000 })
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels, refetchInterval: 5000 })

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold">Dashboard</h2>

      <section>
        <h3 className="text-lg font-semibold mb-2">GPUs</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {(gpus.data ?? []).map((g: any) => (
            <div key={g.index} className="bg-white rounded shadow p-4">
              <div className="text-sm text-gray-500">GPU {g.index}</div>
              <div className="text-xl font-mono">{g.memory_used_mb}/{g.memory_total_mb} MB</div>
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
        <h3 className="text-lg font-semibold mb-2">Deployments</h3>
        <table className="min-w-full bg-white shadow rounded text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="text-left p-2">ID</th><th className="text-left p-2">Model</th>
              <th className="text-left p-2">Backend</th><th className="text-left p-2">Status</th>
              <th className="text-left p-2">Pin</th><th className="text-left p-2">VRAM(MB)</th>
              <th className="text-left p-2">GPUs</th>
            </tr>
          </thead>
          <tbody>
            {(deps.data ?? []).map((d: any) => {
              const m = (models.data ?? []).find((m: any) => m.id === d.model_id)
              return (
                <tr key={d.id} className="border-t">
                  <td className="p-2">{d.id}</td>
                  <td className="p-2 font-mono">{m?.name ?? '-'}</td>
                  <td className="p-2">{d.backend}</td>
                  <td className="p-2">{d.status}</td>
                  <td className="p-2">{d.pinned ? '★' : '-'}</td>
                  <td className="p-2 font-mono">{d.vram_reserved_mb}</td>
                  <td className="p-2 font-mono">{d.gpu_ids.join(',')}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>
    </div>
  )
}
