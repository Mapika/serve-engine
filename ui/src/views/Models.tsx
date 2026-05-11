import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

export default function Models() {
  const qc = useQueryClient()
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const [repo, setRepo] = useState('')
  const [name, setName] = useState('')

  const addModel = useMutation({
    mutationFn: () => api.createModel({ name: name || repo.split('/').pop()!.toLowerCase(), hf_repo: repo }),
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
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Models</h2>
      <div className="bg-white rounded shadow p-4 space-y-2">
        <h3 className="font-semibold">Register a model</h3>
        <input
          className="w-full border rounded px-3 py-2 font-mono"
          placeholder="HuggingFace repo (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
          value={repo}
          onChange={e => setRepo(e.target.value)}
        />
        <input
          className="w-full border rounded px-3 py-2"
          placeholder="Local alias (optional)"
          value={name}
          onChange={e => setName(e.target.value)}
        />
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          disabled={!repo.trim() || addModel.isPending}
          onClick={() => addModel.mutate()}
        >
          {addModel.isPending ? 'Registering…' : 'Register'}
        </button>
      </div>

      <table className="min-w-full bg-white shadow rounded text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="text-left p-2">Name</th><th className="text-left p-2">HF Repo</th>
            <th className="text-left p-2">Revision</th><th className="p-2"></th>
          </tr>
        </thead>
        <tbody>
          {(models.data ?? []).map((m: any) => (
            <tr key={m.id} className="border-t">
              <td className="p-2 font-mono">{m.name}</td>
              <td className="p-2 font-mono text-gray-600">{m.hf_repo}</td>
              <td className="p-2 font-mono text-gray-600">{m.revision}</td>
              <td className="p-2 space-x-2 text-right">
                <button
                  className="text-blue-600 hover:text-blue-800"
                  onClick={() => loadDefault.mutate(m)}
                >Load on GPU 0</button>
                <button
                  className="text-red-600 hover:text-red-800"
                  onClick={() => delModel.mutate(m.name)}
                >Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
