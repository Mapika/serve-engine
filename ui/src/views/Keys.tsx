import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

export default function Keys() {
  const qc = useQueryClient()
  const keys = useQuery({ queryKey: ['keys'], queryFn: api.listKeys })
  const [name, setName] = useState('')
  const [tier, setTier] = useState('standard')
  const [lastSecret, setLastSecret] = useState<string | null>(null)

  const create = useMutation({
    mutationFn: () => api.createKey({ name, tier }),
    onSuccess: (resp: any) => {
      setLastSecret(resp.secret)
      setName('')
      qc.invalidateQueries({ queryKey: ['keys'] })
    },
  })
  const revoke = useMutation({
    mutationFn: (id: number) => api.revokeKey(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['keys'] }),
  })

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">API Keys</h2>

      <div className="bg-white rounded shadow p-4 space-y-2">
        <h3 className="font-semibold">Create a key</h3>
        <div className="flex gap-2">
          <input
            className="flex-1 border rounded px-3 py-2"
            placeholder="Label (e.g. alice)"
            value={name}
            onChange={e => setName(e.target.value)}
          />
          <select className="border rounded px-3 py-2" value={tier} onChange={e => setTier(e.target.value)}>
            <option value="admin">admin</option>
            <option value="standard">standard</option>
            <option value="trial">trial</option>
          </select>
          <button
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
            disabled={!name.trim() || create.isPending}
            onClick={() => create.mutate()}
          >Create</button>
        </div>
        {lastSecret && (
          <div className="bg-yellow-50 border border-yellow-200 rounded p-3 text-sm">
            <div className="font-semibold">Save this — it won't be shown again:</div>
            <code className="font-mono break-all">{lastSecret}</code>
          </div>
        )}
      </div>

      <table className="min-w-full bg-white shadow rounded text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="text-left p-2">ID</th><th className="text-left p-2">Name</th>
            <th className="text-left p-2">Tier</th><th className="text-left p-2">Prefix</th>
            <th className="text-left p-2">Status</th><th className="p-2"></th>
          </tr>
        </thead>
        <tbody>
          {(keys.data ?? []).map((k: any) => (
            <tr key={k.id} className="border-t">
              <td className="p-2">{k.id}</td>
              <td className="p-2">{k.name}</td>
              <td className="p-2">{k.tier}</td>
              <td className="p-2 font-mono text-gray-600">{k.prefix}</td>
              <td className="p-2">{k.revoked ? <span className="text-red-600">revoked</span> : 'active'}</td>
              <td className="p-2 text-right">
                {!k.revoked && (
                  <button
                    className="text-red-600 hover:text-red-800"
                    onClick={() => revoke.mutate(k.id)}
                  >Revoke</button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
