import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'

export default function Services() {
  const qc = useQueryClient()
  const profiles = useQuery({ queryKey: ['profiles'], queryFn: api.listProfiles })
  const routes = useQuery({ queryKey: ['routes'], queryFn: api.listRoutes })
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const backends = useQuery({ queryKey: ['backends'], queryFn: api.listBackends })

  const hasProfiles = (profiles.data ?? []).length > 0

  const [profileForm, setProfileForm] = useState({
    name: '',
    model_name: '',
    hf_repo: '',
    backend: '',
    gpu_ids: '0',
    max_model_len: '8192',
    pinned: false,
  })
  const [profileFormError, setProfileFormError] = useState('')
  const [profileActionError, setProfileActionError] = useState('')

  const createProfile = useMutation({
    mutationFn: () => {
      const gpu_ids = profileForm.gpu_ids
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)
        .map(Number)
        .filter(n => Number.isInteger(n) && n >= 0)
      if (gpu_ids.length === 0) throw new Error('gpu_ids: at least one integer required')
      const max_model_len = Number(profileForm.max_model_len)
      if (!Number.isInteger(max_model_len) || max_model_len < 128) {
        throw new Error('max_model_len: integer >= 128')
      }
      return api.createProfile({
        name: profileForm.name.trim(),
        model_name: profileForm.model_name.trim(),
        hf_repo: profileForm.hf_repo.trim(),
        backend: profileForm.backend || undefined,
        gpu_ids,
        max_model_len,
        pinned: profileForm.pinned,
      })
    },
    onMutate: () => setProfileFormError(''),
    onError: (e: Error) => setProfileFormError(e.message),
    onSuccess: () => {
      setProfileForm({
        name: '', model_name: '', hf_repo: '', backend: '',
        gpu_ids: '0', max_model_len: '8192', pinned: false,
      })
      qc.invalidateQueries({ queryKey: ['profiles'] })
    },
  })

  const deployProfile = useMutation({
    mutationFn: (name: string) => api.deployProfile(name),
    onMutate: () => setProfileActionError(''),
    onError: (e: Error) => setProfileActionError(e.message),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['deps'] }),
  })

  const deleteProfile = useMutation({
    mutationFn: (name: string) => api.deleteProfile(name),
    onMutate: () => setProfileActionError(''),
    onError: (e: Error) => setProfileActionError(e.message),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['profiles'] })
      qc.invalidateQueries({ queryKey: ['routes'] })
    },
  })

  const [routeForm, setRouteForm] = useState({
    name: '',
    match_model: '',
    profile_name: '',
    fallback_profile_name: '',
    priority: '100',
  })
  const [routeError, setRouteError] = useState('')

  const createRoute = useMutation({
    mutationFn: () => {
      const priority = Number(routeForm.priority)
      if (!Number.isInteger(priority)) throw new Error('priority must be an integer')
      return api.createRoute({
        name: routeForm.name.trim(),
        match_model: routeForm.match_model.trim(),
        profile_name: routeForm.profile_name,
        fallback_profile_name: routeForm.fallback_profile_name || null,
        priority,
      })
    },
    onMutate: () => setRouteError(''),
    onError: (e: Error) => setRouteError(e.message),
    onSuccess: () => {
      setRouteForm({
        name: '', match_model: '', profile_name: '',
        fallback_profile_name: '', priority: '100',
      })
      qc.invalidateQueries({ queryKey: ['routes'] })
    },
  })

  const deleteRoute = useMutation({
    mutationFn: (name: string) => api.deleteRoute(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['routes'] }),
  })

  return (
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">services</h2>
        <div className="label">
          {(profiles.data ?? []).length} profiles / {(routes.data ?? []).length} routes
        </div>
      </header>

      <section className="space-y-4">
        <div className="flex items-baseline justify-between">
          <div className="label">profiles</div>
          <div className="text-mute text-[11px] tracking-wider">
            reusable launch definition
          </div>
        </div>
        {profileActionError && (
          <div className="text-err text-[11px] tracking-wider">{profileActionError}</div>
        )}

        <div className="bg-elev/40 border border-rule p-5 space-y-4">
          <div className="grid grid-cols-12 gap-3">
            <div className="space-y-1 col-span-12 md:col-span-3">
              <div className="label">profile name</div>
              <input
                className="field font-mono w-full text-[12px]"
                placeholder="qwen-vllm"
                value={profileForm.name}
                onChange={e => setProfileForm(f => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div className="space-y-1 col-span-12 md:col-span-3">
              <div className="label">model name</div>
              <input
                className="field font-mono w-full text-[12px]"
                list="profile-model-list"
                placeholder="qwen"
                value={profileForm.model_name}
                onChange={e => {
                  const next = e.target.value
                  const m = (models.data ?? []).find((m: any) => m.name === next)
                  setProfileForm(f => ({
                    ...f,
                    model_name: next,
                    hf_repo: m ? m.hf_repo : f.hf_repo,
                  }))
                }}
              />
              <datalist id="profile-model-list">
                {(models.data ?? []).map((m: any) => (
                  <option key={m.id} value={m.name} />
                ))}
              </datalist>
            </div>
            <div className="space-y-1 col-span-12 md:col-span-6">
              <div className="label">hf repo</div>
              <input
                className="field font-mono w-full text-[12px]"
                placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                value={profileForm.hf_repo}
                onChange={e => setProfileForm(f => ({ ...f, hf_repo: e.target.value }))}
              />
            </div>
            <div className="space-y-1 col-span-6 md:col-span-3">
              <div className="label">backend</div>
              <select
                className="field font-mono w-full text-[12px]"
                value={profileForm.backend}
                onChange={e => setProfileForm(f => ({ ...f, backend: e.target.value }))}
              >
                <option value="">auto</option>
                {(backends.data ?? []).map(b => (
                  <option key={b.name} value={b.name}>{b.name}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1 col-span-6 md:col-span-3">
              <div className="label">gpu ids</div>
              <input
                className="field font-mono w-full text-[12px] tnum"
                placeholder="0 or 0,1"
                value={profileForm.gpu_ids}
                onChange={e => setProfileForm(f => ({ ...f, gpu_ids: e.target.value }))}
              />
            </div>
            <div className="space-y-1 col-span-6 md:col-span-3">
              <div className="label">max model len</div>
              <input
                className="field font-mono w-full text-[12px] tnum"
                value={profileForm.max_model_len}
                onChange={e => setProfileForm(f => ({ ...f, max_model_len: e.target.value }))}
              />
            </div>
            <div className="space-y-1 col-span-6 md:col-span-3 flex flex-col">
              <div className="label">options</div>
              <label className="text-[12px] text-dim flex items-center gap-2 select-none cursor-pointer pt-2">
                <input
                  type="checkbox"
                  className="accent-accent"
                  checked={profileForm.pinned}
                  onChange={e => setProfileForm(f => ({ ...f, pinned: e.target.checked }))}
                />
                pinned (skip idle reaper)
              </label>
            </div>
          </div>
          {profileFormError && (
            <div className="text-err text-[11px] tracking-wider">{profileFormError}</div>
          )}
          <div className="flex items-center gap-3">
            <button
              className="btn-primary"
              disabled={
                !profileForm.name.trim() ||
                !profileForm.model_name.trim() ||
                !profileForm.hf_repo.trim() ||
                createProfile.isPending
              }
              onClick={() => createProfile.mutate()}
            >
              {createProfile.isPending ? 'creating…' : 'create profile'}
            </button>
          </div>
        </div>

        <table className="ditable">
          <thead>
            <tr>
              <th>name</th>
              <th>model</th>
              <th>backend</th>
              <th className="text-right">gpus</th>
              <th className="text-right">ctx</th>
              <th>pinned</th>
              <th className="text-right">actions</th>
            </tr>
          </thead>
          <tbody>
            {(profiles.data ?? []).length === 0 && (
              <tr>
                <td colSpan={7} className="!py-12 text-center text-mute">
                  no profiles yet. create one above to define how a model is launched.
                </td>
              </tr>
            )}
            {(profiles.data ?? []).map(p => {
              const isDeploying = deployProfile.isPending && deployProfile.variables === p.name
              const isDeleting = deleteProfile.isPending && deleteProfile.variables === p.name
              return (
                <tr key={p.id}>
                  <td>{p.name}</td>
                  <td className="text-dim">{p.model_name}</td>
                  <td className="text-dim">{p.backend}</td>
                  <td className="text-right text-dim tnum">{p.gpu_ids.join(',') || '—'}</td>
                  <td className="text-right tnum">{p.max_model_len}</td>
                  <td>
                    {p.pinned
                      ? <span className="text-accent">yes</span>
                      : <span className="text-mute">no</span>}
                  </td>
                  <td className="text-right space-x-5 whitespace-nowrap">
                    <button
                      className="text-accent hover:opacity-70 transition-opacity disabled:opacity-40"
                      disabled={isDeploying}
                      onClick={() => deployProfile.mutate(p.name)}
                    >
                      {isDeploying ? 'deploying…' : 'deploy'}
                    </button>
                    <button
                      className="btn-link-danger disabled:opacity-40"
                      disabled={isDeleting}
                      onClick={() => {
                        if (confirm(`delete profile ${p.name}?`)) deleteProfile.mutate(p.name)
                      }}
                    >
                      {isDeleting ? 'deleting…' : 'delete'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </section>

      <section className="space-y-4">
        <div className="flex items-baseline justify-between">
          <div className="label">routes</div>
          <div className="text-mute text-[11px] tracking-wider">
            public model name → profile · lower priority wins
          </div>
        </div>

        {!hasProfiles ? (
          <div className="border border-rule bg-elev/40 px-5 py-12 text-center text-mute text-[12px]">
            create a profile above first — routes point at profiles.
          </div>
        ) : (
          <div className="bg-elev/40 border border-rule p-5 space-y-4">
            <div className="grid grid-cols-12 gap-3">
              <div className="space-y-1 col-span-6 md:col-span-2">
                <div className="label">name</div>
                <input
                  className="field font-mono w-full text-[12px]"
                  placeholder="chat-default"
                  value={routeForm.name}
                  onChange={e => setRouteForm(f => ({ ...f, name: e.target.value }))}
                />
              </div>
              <div className="space-y-1 col-span-6 md:col-span-3">
                <div className="label">profile</div>
                <select
                  className="field font-mono w-full text-[12px]"
                  value={routeForm.profile_name}
                  onChange={e => setRouteForm(f => ({ ...f, profile_name: e.target.value }))}
                >
                  <option value="">choose…</option>
                  {(profiles.data ?? []).map(p => (
                    <option key={p.id} value={p.name}>{p.name}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1 col-span-6 md:col-span-3">
                <div className="label">fallback (optional)</div>
                <select
                  className="field font-mono w-full text-[12px]"
                  value={routeForm.fallback_profile_name}
                  onChange={e => setRouteForm(f => ({ ...f, fallback_profile_name: e.target.value }))}
                >
                  <option value="">none</option>
                  {(profiles.data ?? [])
                    .filter(p => p.name !== routeForm.profile_name)
                    .map(p => (
                      <option key={p.id} value={p.name}>{p.name}</option>
                    ))}
                </select>
              </div>
              <div className="space-y-1 col-span-6 md:col-span-3">
                <div className="label">match model (exact)</div>
                <input
                  className="field font-mono w-full text-[12px]"
                  placeholder="chat"
                  value={routeForm.match_model}
                  onChange={e => setRouteForm(f => ({ ...f, match_model: e.target.value }))}
                />
              </div>
              <div className="space-y-1 col-span-12 md:col-span-1">
                <div className="label">pri</div>
                <input
                  className="field font-mono w-full text-[12px] tnum text-right"
                  value={routeForm.priority}
                  onChange={e => setRouteForm(f => ({ ...f, priority: e.target.value }))}
                />
              </div>
            </div>
            {routeError && (
              <div className="text-err text-[11px] tracking-wider">{routeError}</div>
            )}
            <div className="flex items-center gap-3">
              <button
                className="btn-primary"
                disabled={
                  !routeForm.name.trim() ||
                  !routeForm.match_model.trim() ||
                  !routeForm.profile_name ||
                  createRoute.isPending
                }
                onClick={() => createRoute.mutate()}
              >
                {createRoute.isPending ? 'creating…' : 'create route'}
              </button>
              <span className="label">
                callable as <span className="text-dim">model: {routeForm.match_model || '<name>'}</span>
              </span>
            </div>
          </div>
        )}

        <table className="ditable">
          <thead>
            <tr>
              <th className="w-12">pri</th>
              <th>name</th>
              <th>match model</th>
              <th>profile</th>
              <th>fallback</th>
              <th>enabled</th>
              <th className="text-right">actions</th>
            </tr>
          </thead>
          <tbody>
            {(routes.data ?? []).length === 0 && (
              <tr>
                <td colSpan={7} className="!py-12 text-center text-mute">
                  {hasProfiles
                    ? 'no routes. create one above to expose a public model name.'
                    : 'no routes — and no profiles to route at yet.'}
                </td>
              </tr>
            )}
            {(routes.data ?? [])
              .slice()
              .sort((a, b) => a.priority - b.priority)
              .map(r => (
                <tr key={r.id}>
                  <td className="text-mute tnum">{r.priority}</td>
                  <td>{r.name}</td>
                  <td className="font-mono text-[12px]">{r.match_model}</td>
                  <td className="text-dim">{r.profile_name}</td>
                  <td className="text-mute">{r.fallback_profile_name ?? '—'}</td>
                  <td>
                    <span className={`dot ${r.enabled ? 'dot-ready' : 'dot-stopped'}`} />
                    <span className="text-dim">{r.enabled ? 'on' : 'off'}</span>
                  </td>
                  <td className="text-right">
                    <button
                      className="btn-link-danger disabled:opacity-40"
                      disabled={deleteRoute.isPending}
                      onClick={() => {
                        if (confirm(`delete route ${r.name}?`)) deleteRoute.mutate(r.name)
                      }}
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
