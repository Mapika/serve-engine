import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

export default function Models() {
  const qc = useQueryClient()
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const [repo, setRepo] = useState('')
  const [name, setName] = useState('')

  const addModel = useMutation({
    mutationFn: () => api.createModel({
      name: name || repo.split('/').pop()!.toLowerCase(),
      hf_repo: repo,
    }),
    onSuccess: () => { setRepo(''); setName(''); qc.invalidateQueries({ queryKey: ['models'] }) },
  })
  const delModel = useMutation({
    mutationFn: (modelName: string) => api.deleteModel(modelName),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['models'] }),
  })
  const loadDefault = useMutation({
    mutationFn: (m: any) => api.loadModel({
      model_name: m.name, hf_repo: m.hf_repo, gpu_ids: [0], max_model_len: 4096,
    }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['deps'] }),
  })

  return (
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">models</h2>
        <div className="label">{(models.data ?? []).length} registered</div>
      </header>

      <section className="space-y-5">
        <div className="label">register</div>
        <div className="grid grid-cols-[1fr_220px_auto] gap-3 max-w-3xl">
          <input
            className="field font-mono"
            placeholder="huggingface repo (e.g. Qwen/Qwen3.6-35B-A3B-FP8)"
            value={repo}
            onChange={e => setRepo(e.target.value)}
          />
          <input
            className="field font-mono"
            placeholder="local alias (optional)"
            value={name}
            onChange={e => setName(e.target.value)}
          />
          <button
            className="btn-primary"
            disabled={!repo.trim() || addModel.isPending}
            onClick={() => addModel.mutate()}
          >
            {addModel.isPending ? 'registering...' : 'register'}
          </button>
        </div>
        {addModel.error && (
          <div className="text-err text-[12px]">{(addModel.error as Error).message}</div>
        )}
      </section>

      <section className="space-y-4">
        <div className="label">registry</div>
        <table className="ditable">
          <thead>
            <tr>
              <th>name</th>
              <th>huggingface</th>
              <th>revision</th>
              <th className="text-right"></th>
            </tr>
          </thead>
          <tbody>
            {(models.data ?? []).length === 0 && (
              <tr>
                <td colSpan={4} className="!py-12 text-center text-mute">
                  no models registered yet
                </td>
              </tr>
            )}
            {(models.data ?? []).map((m: any) => (
              <tr key={m.id}>
                <td>{m.name}</td>
                <td className="text-dim">{m.hf_repo}</td>
                <td className="text-mute">{m.revision}</td>
                <td className="text-right space-x-6">
                  <button
                    className="text-accent hover:opacity-70 transition-opacity"
                    onClick={() => loadDefault.mutate(m)}
                  >
                    load
                  </button>
                  <button
                    className="btn-link-danger"
                    onClick={() => delModel.mutate(m.name)}
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
