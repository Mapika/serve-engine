import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

export default function Keys() {
  const qc = useQueryClient()
  const keys = useQuery({ queryKey: ['keys'], queryFn: api.listKeys })
  const [name, setName] = useState('')
  const [tier, setTier] = useState('standard')
  const [lastSecret, setLastSecret] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [showRevoked, setShowRevoked] = useState(false)

  const create = useMutation({
    mutationFn: () => api.createKey({ name, tier }),
    onSuccess: (resp: any) => {
      setLastSecret(resp.secret)
      setCopied(false)
      setName('')
      qc.invalidateQueries({ queryKey: ['keys'] })
    },
  })
  const revoke = useMutation({
    mutationFn: (id: number) => api.revokeKey(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['keys'] }),
  })

  async function copySecret() {
    if (!lastSecret) return
    try {
      await navigator.clipboard.writeText(lastSecret)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard blocked — user can still select manually */
    }
  }

  const all = keys.data ?? []
  const active = all.filter((k: any) => !k.revoked).length
  const revoked = all.length - active
  const visible = showRevoked
    ? all
    : all.filter((k: any) => !k.revoked)

  return (
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">api keys</h2>
        <div className="label">{active} active</div>
      </header>

      <section className="space-y-5">
        <div className="label">issue a key</div>
        <div className="grid grid-cols-[1fr_180px_auto] gap-3 max-w-3xl">
          <input
            className="field font-mono"
            placeholder="label (e.g. alice / web / cron)"
            value={name}
            onChange={e => setName(e.target.value)}
          />
          <select
            className="field font-mono"
            value={tier}
            onChange={e => setTier(e.target.value)}
          >
            <option value="admin">admin</option>
            <option value="standard">standard</option>
            <option value="trial">trial</option>
          </select>
          <button
            className="btn-primary"
            disabled={!name.trim() || create.isPending}
            onClick={() => create.mutate()}
          >
            {create.isPending ? 'issuing...' : 'issue'}
          </button>
        </div>
        {lastSecret && (
          <div className="border border-accent/40 bg-[var(--accent-soft)] px-4 py-3 max-w-3xl">
            <div className="flex items-center justify-between mb-2">
              <div className="label text-accent">save this; it won't be shown again</div>
              <div className="flex items-center gap-4">
                <button
                  onClick={copySecret}
                  className="text-accent text-[11px] tracking-wider hover:opacity-70 transition-opacity"
                >
                  {copied ? 'copied ✓' : 'copy'}
                </button>
                <button
                  onClick={() => setLastSecret(null)}
                  className="text-mute text-[11px] tracking-wider hover:text-dim transition-colors"
                  aria-label="dismiss"
                >
                  dismiss
                </button>
              </div>
            </div>
            <code className="font-mono text-[13px] break-all select-all">{lastSecret}</code>
          </div>
        )}
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="label">issued keys</div>
          {revoked > 0 && (
            <label className="text-mute text-[11px] tracking-wider select-none cursor-pointer hover:text-dim transition-colors">
              <input
                type="checkbox"
                className="mr-2 accent-accent align-middle"
                checked={showRevoked}
                onChange={e => setShowRevoked(e.target.checked)}
              />
              show revoked {!showRevoked && <span className="text-accent">({revoked})</span>}
            </label>
          )}
        </div>
        <table className="ditable">
          <thead>
            <tr>
              <th>#</th>
              <th>label</th>
              <th>tier</th>
              <th>prefix</th>
              <th>status</th>
              <th className="text-right"></th>
            </tr>
          </thead>
          <tbody>
            {visible.length === 0 && (
              <tr>
                <td colSpan={6} className="!py-12 text-center text-mute">
                  {all.length === 0 ? 'no keys yet' : 'no active keys'}
                </td>
              </tr>
            )}
            {visible.map((k: any) => (
              <tr key={k.id}>
                <td className="text-mute tnum">{k.id}</td>
                <td>{k.name}</td>
                <td className="text-dim">{k.tier}</td>
                <td className="text-mute">{k.prefix}</td>
                <td>
                  <span className={`dot ${k.revoked ? 'dot-failed' : 'dot-ready'}`} />
                  <span className={k.revoked ? 'text-err' : 'text-dim'}>
                    {k.revoked ? 'revoked' : 'active'}
                  </span>
                </td>
                <td className="text-right">
                  {!k.revoked && (
                    <button
                      className="btn-link-danger disabled:opacity-40"
                      disabled={revoke.isPending && revoke.variables === k.id}
                      onClick={() => {
                        if (confirm(`revoke key "${k.name}" (#${k.id})? this cannot be undone.`)) {
                          revoke.mutate(k.id)
                        }
                      }}
                    >
                      {revoke.isPending && revoke.variables === k.id ? 'revoking…' : 'revoke'}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
