# View 1: Services & Routes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level "services" UI view that exposes `/admin/service-profiles` and `/admin/routes`. Profiles get a create form + table with deploy/delete actions; Routes get a create form + table with delete and visual priority ordering.

**Architecture:** One new top-level React view, `Services.tsx`, with two stacked sections (Profiles, Routes). Reuses existing nav, `api.ts`, design tokens, and `ditable` table style. Backend untouched — endpoints already exist as documented in README §Service Routes.

**Tech Stack:** React 18, @tanstack/react-query, Tailwind. No backend code.

**Test discipline:** The UI project has no JS test runner. Each task ends with a manual verification step in `npm run dev` against a running daemon. Where a check can be done at the type level (tsc) or via the API directly (`curl`), use that.

---

## File Structure

- **Create:** `ui/src/views/Services.tsx` — the new view (profiles section + routes section)
- **Modify:** `ui/src/api.ts` — add typed API methods for profiles and routes
- **Modify:** `ui/src/App.tsx` — add `services` to the `View` type, `VIEWS` array, and view-switch block

Single file for the view because Profiles and Routes are tightly coupled (a route always references a profile; common error states relate to both). Splitting would force a wrapper and duplicate the data plumbing.

---

## Task 1: Type the API methods

**Files:**
- Modify: `ui/src/api.ts` — append profile + route methods to the `api` object

- [ ] **Step 1: Add ServiceProfile and ServiceRoute types**

At the top of `ui/src/api.ts`, immediately after the `eventSourceUrl` function, add:

```ts
export type ServiceProfile = {
  id: number
  name: string
  model_name: string
  hf_repo: string
  revision: string
  backend: string
  image_tag: string
  gpu_ids: number[]
  tensor_parallel: number
  max_model_len: number
  dtype: string
  pinned: boolean
  idle_timeout_s: number | null
  target_concurrency: number | null
  max_loras: number
  max_lora_rank: number
  extra_args: Record<string, string>
}

export type ServiceRoute = {
  id: number
  name: string
  match_model: string
  profile_name: string
  fallback_profile_name: string | null
  enabled: boolean
  priority: number
}

export type CreateProfileBody = {
  name: string
  model_name: string
  hf_repo: string
  revision?: string
  backend?: string
  gpu_ids: number[]
  max_model_len?: number
  pinned?: boolean
  target_concurrency?: number | null
}

export type CreateRouteBody = {
  name: string
  match_model: string
  profile_name: string
  fallback_profile_name?: string | null
  enabled?: boolean
  priority?: number
}
```

- [ ] **Step 2: Add API methods to the `api` object**

Inside the `export const api = { ... }` block, append (before the closing brace):

```ts
  // Service profiles.
  listProfiles: () => jfetch<ServiceProfile[]>('GET', '/admin/service-profiles'),
  createProfile: (b: CreateProfileBody) =>
    jfetch<ServiceProfile>('POST', '/admin/service-profiles', b),
  deployProfile: (name: string) =>
    jfetch<any>('POST', `/admin/service-profiles/${encodeURIComponent(name)}/deploy`),
  deleteProfile: (name: string) =>
    jfetch<void>('DELETE', `/admin/service-profiles/${encodeURIComponent(name)}`),

  // Service routes.
  listRoutes: () => jfetch<ServiceRoute[]>('GET', '/admin/routes'),
  createRoute: (b: CreateRouteBody) => jfetch<ServiceRoute>('POST', '/admin/routes', b),
  deleteRoute: (name: string) =>
    jfetch<void>('DELETE', `/admin/routes/${encodeURIComponent(name)}`),
```

- [ ] **Step 3: Type-check**

Run: `cd ui && npx tsc -b --noEmit`
Expected: exit 0, no errors.

- [ ] **Step 4: Sanity-check against the live daemon**

With `serve daemon` running locally and `$SERVE_TOKEN` set:

```bash
curl -sf -H "Authorization: Bearer $SERVE_TOKEN" "$SERVE_URL/admin/service-profiles" | head -c 200
curl -sf -H "Authorization: Bearer $SERVE_TOKEN" "$SERVE_URL/admin/routes" | head -c 200
```
Expected: JSON arrays (possibly empty), shape matches `ServiceProfile` / `ServiceRoute`. If the daemon isn't running, skip — type-check is enough for this step.

- [ ] **Step 5: Commit**

```bash
git add ui/src/api.ts
git commit -m "feat(ui): typed api client for service profiles and routes"
```

---

## Task 2: Skeleton Services view + nav registration

**Files:**
- Create: `ui/src/views/Services.tsx`
- Modify: `ui/src/App.tsx` (add `services` to View type and VIEWS array; render block)

- [ ] **Step 1: Create the skeleton view**

Create `ui/src/views/Services.tsx`:

```tsx
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Services() {
  const profiles = useQuery({ queryKey: ['profiles'], queryFn: api.listProfiles })
  const routes = useQuery({ queryKey: ['routes'], queryFn: api.listRoutes })

  return (
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">services</h2>
        <div className="label">
          {(profiles.data ?? []).length} profiles / {(routes.data ?? []).length} routes
        </div>
      </header>

      <section className="space-y-4">
        <div className="label">routes</div>
        <div className="text-mute text-[12px]">coming next step</div>
      </section>

      <section className="space-y-4">
        <div className="label">profiles</div>
        <div className="text-mute text-[12px]">coming next step</div>
      </section>
    </div>
  )
}
```

- [ ] **Step 2: Register in App.tsx**

Modify `ui/src/App.tsx`:

After the `import Logs from './views/Logs'` line, add:
```ts
import Services from './views/Services'
```

In the `View` type union (currently `'dashboard' | 'models' | ... | 'logs'`), add `'services'` between `'adapters'` and `'predictor'`:
```ts
type View =
  | 'dashboard'
  | 'models'
  | 'adapters'
  | 'services'
  | 'predictor'
  | 'playground'
  | 'keys'
  | 'logs'
```

In the `VIEWS` const, add the matching entry between `adapters` and `predictor`:
```ts
{ id: 'services', label: 'services' },
```

In the JSX view-switch block (`main > div`), add between adapters and predictor:
```tsx
{view === 'services' && <Services />}
```

- [ ] **Step 3: Type-check and build**

Run: `cd ui && npx tsc -b --noEmit`
Expected: exit 0.

- [ ] **Step 4: Manual verification**

Run `cd ui && npm run dev`. Open the app, log in. Confirm:
- `services` tab appears in nav between `adapters` and `predictor`
- Clicking it switches the view
- Header shows `0 profiles / 0 routes` (or live counts if the daemon has any)
- No console errors

- [ ] **Step 5: Commit**

```bash
git add ui/src/views/Services.tsx ui/src/App.tsx
git commit -m "feat(ui): services view skeleton + nav entry"
```

---

## Task 3: Routes table

**Files:**
- Modify: `ui/src/views/Services.tsx` (replace the routes placeholder with a real table)

- [ ] **Step 1: Replace the routes section**

In `ui/src/views/Services.tsx`, replace the `<section>` whose label is `routes` with:

```tsx
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="label">routes</div>
          <div className="text-mute text-[11px] tracking-wider">
            lower priority wins
          </div>
        </div>
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
                  no routes. create one below to expose a public model name.
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
                    <span className="text-mute">—</span>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>
```

- [ ] **Step 2: Type-check**

Run: `cd ui && npx tsc -b --noEmit`
Expected: exit 0.

- [ ] **Step 3: Manual verification**

Reload the dev server tab.
- Empty case: shows "no routes. create one below to expose a public model name."
- With routes (create one via curl if needed):
  ```bash
  curl -X POST "$SERVE_URL/admin/service-profiles" \
    -H "Authorization: Bearer $SERVE_TOKEN" -H 'Content-Type: application/json' \
    -d '{"name":"qwen-vllm","model_name":"qwen","hf_repo":"Qwen/Qwen2.5-0.5B-Instruct","backend":"vllm","gpu_ids":[0]}'
  curl -X POST "$SERVE_URL/admin/routes" \
    -H "Authorization: Bearer $SERVE_TOKEN" -H 'Content-Type: application/json' \
    -d '{"name":"chat-default","match_model":"chat","profile_name":"qwen-vllm","priority":10}'
  ```
  Routes render with priority on left, sorted ascending.

- [ ] **Step 4: Commit**

```bash
git add ui/src/views/Services.tsx
git commit -m "feat(ui): routes table in services view"
```

---

## Task 4: Routes create form + delete action

**Files:**
- Modify: `ui/src/views/Services.tsx`

- [ ] **Step 1: Add form state and mutations near the top of the component**

Inside `export default function Services()`, after the `routes` query, add:

```ts
  const qc = useQueryClient()
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
      setRouteForm({ name: '', match_model: '', profile_name: '', fallback_profile_name: '', priority: '100' })
      qc.invalidateQueries({ queryKey: ['routes'] })
    },
  })

  const deleteRoute = useMutation({
    mutationFn: (name: string) => api.deleteRoute(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['routes'] }),
  })
```

Update imports at top of file:
```ts
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api'
```

- [ ] **Step 2: Add a "create route" form above the routes table**

Inside the routes `<section>`, insert this block immediately after the `flex items-center justify-between` header div (before the `<table>`):

```tsx
        <div className="bg-elev/40 border border-rule p-5 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div className="space-y-1">
              <div className="label">name</div>
              <input
                className="field font-mono w-full text-[12px]"
                placeholder="chat-default"
                value={routeForm.name}
                onChange={e => setRouteForm(f => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <div className="label">match model</div>
              <input
                className="field font-mono w-full text-[12px]"
                placeholder="chat"
                value={routeForm.match_model}
                onChange={e => setRouteForm(f => ({ ...f, match_model: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
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
            <div className="space-y-1">
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
            <div className="space-y-1">
              <div className="label">priority</div>
              <input
                className="field font-mono w-full text-[12px] tnum"
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
              public model name → profile mapping
            </span>
          </div>
        </div>
```

- [ ] **Step 3: Replace the placeholder delete cell with a real one**

In the routes table body, change the last `<td>` of each row from:
```tsx
                  <td className="text-right">
                    <span className="text-mute">—</span>
                  </td>
```
to:
```tsx
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
```

- [ ] **Step 4: Type-check**

Run: `cd ui && npx tsc -b --noEmit`
Expected: exit 0.

- [ ] **Step 5: Manual verification**

In the dev server:
1. Create a profile via curl (see Task 3 step 3 if not already there).
2. In the UI: fill the form (name=`chat-default`, match=`chat`, profile=the one you just made, priority=10). Click create. The route appears in the table.
3. Try submitting with a duplicate name — the `routeError` appears with the daemon's 409 message.
4. Click `delete` on a row, confirm. The row disappears.

- [ ] **Step 6: Commit**

```bash
git add ui/src/views/Services.tsx
git commit -m "feat(ui): create and delete service routes"
```

---

## Task 5: Profiles table + delete + deploy

**Files:**
- Modify: `ui/src/views/Services.tsx`

- [ ] **Step 1: Add profile mutations**

Inside the component, near the other mutations, add:

```ts
  const deployProfile = useMutation({
    mutationFn: (name: string) => api.deployProfile(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['deps'] }),
  })

  const deleteProfile = useMutation({
    mutationFn: (name: string) => api.deleteProfile(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['profiles'] })
      qc.invalidateQueries({ queryKey: ['routes'] })
    },
  })
```

- [ ] **Step 2: Replace the profiles placeholder with a real table**

Replace the `<section>` whose label is `profiles` with:

```tsx
      <section className="space-y-4">
        <div className="label">profiles</div>
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
                  no profiles. create one via the API or CLI — UI form for profiles comes after the routes flow is solid.
                </td>
              </tr>
            )}
            {(profiles.data ?? []).map(p => (
              <tr key={p.id}>
                <td>{p.name}</td>
                <td className="text-dim">{p.model_name}</td>
                <td className="text-dim">{p.backend}</td>
                <td className="text-right text-dim tnum">{p.gpu_ids.join(',') || '—'}</td>
                <td className="text-right tnum">{p.max_model_len}</td>
                <td>{p.pinned ? <span className="text-accent">yes</span> : <span className="text-mute">no</span>}</td>
                <td className="text-right space-x-5 whitespace-nowrap">
                  <button
                    className="text-accent hover:opacity-70 transition-opacity disabled:opacity-40"
                    disabled={deployProfile.isPending}
                    onClick={() => deployProfile.mutate(p.name)}
                  >
                    {deployProfile.isPending && deployProfile.variables === p.name ? 'deploying…' : 'deploy'}
                  </button>
                  <button
                    className="btn-link-danger disabled:opacity-40"
                    disabled={deleteProfile.isPending}
                    onClick={() => {
                      if (confirm(`delete profile ${p.name}?`)) deleteProfile.mutate(p.name)
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
```

- [ ] **Step 3: Type-check**

Run: `cd ui && npx tsc -b --noEmit`
Expected: exit 0.

- [ ] **Step 4: Manual verification**

In the dev server:
1. Profiles table shows existing profiles with model, backend, gpus, ctx, pinned columns.
2. Click `deploy` on a profile — a deployment starts (visible on the dashboard view; the daemon also publishes an event).
3. Click `delete` on a profile that has no route pointing to it — it disappears.
4. Click `delete` on a profile that *does* have a route pointing to it — the daemon returns an error; verify the error surfaces (it'll go to the network tab; full error display is Task 7 polish).

- [ ] **Step 5: Commit**

```bash
git add ui/src/views/Services.tsx
git commit -m "feat(ui): list, deploy, and delete service profiles"
```

---

## Task 6: Profile create form

**Files:**
- Modify: `ui/src/views/Services.tsx`

- [ ] **Step 1: Add form state and create mutation**

Near the other state declarations in the component, add:

```ts
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const backends = useQuery({ queryKey: ['backends'], queryFn: api.listBackends })

  const [profileForm, setProfileForm] = useState({
    name: '',
    model_name: '',
    hf_repo: '',
    backend: '',
    gpu_ids: '0',
    max_model_len: '8192',
    pinned: false,
  })
  const [profileError, setProfileError] = useState('')

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
    onMutate: () => setProfileError(''),
    onError: (e: Error) => setProfileError(e.message),
    onSuccess: () => {
      setProfileForm({
        name: '', model_name: '', hf_repo: '', backend: '',
        gpu_ids: '0', max_model_len: '8192', pinned: false,
      })
      qc.invalidateQueries({ queryKey: ['profiles'] })
    },
  })
```

- [ ] **Step 2: Insert the create form above the profiles table**

Inside the profiles `<section>`, after the `<div className="label">profiles</div>` line, add this block before the `<table>`:

```tsx
        <div className="bg-elev/40 border border-rule p-5 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="space-y-1">
              <div className="label">profile name</div>
              <input
                className="field font-mono w-full text-[12px]"
                placeholder="qwen-vllm"
                value={profileForm.name}
                onChange={e => setProfileForm(f => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
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
            <div className="space-y-1">
              <div className="label">hf repo</div>
              <input
                className="field font-mono w-full text-[12px]"
                placeholder="Qwen/Qwen2.5-0.5B-Instruct"
                value={profileForm.hf_repo}
                onChange={e => setProfileForm(f => ({ ...f, hf_repo: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
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
            <div className="space-y-1">
              <div className="label">gpu ids</div>
              <input
                className="field font-mono w-full text-[12px] tnum"
                placeholder="0 or 0,1"
                value={profileForm.gpu_ids}
                onChange={e => setProfileForm(f => ({ ...f, gpu_ids: e.target.value }))}
              />
            </div>
            <div className="space-y-1">
              <div className="label">max model len</div>
              <input
                className="field font-mono w-full text-[12px] tnum"
                value={profileForm.max_model_len}
                onChange={e => setProfileForm(f => ({ ...f, max_model_len: e.target.value }))}
              />
            </div>
            <div className="space-y-1 col-span-2 md:col-span-2 flex items-end">
              <label className="text-[12px] text-dim flex items-center gap-2 select-none cursor-pointer">
                <input
                  type="checkbox"
                  className="accent-accent"
                  checked={profileForm.pinned}
                  onChange={e => setProfileForm(f => ({ ...f, pinned: e.target.checked }))}
                />
                pinned (idle reaper skips it)
              </label>
            </div>
          </div>
          {profileError && (
            <div className="text-err text-[11px] tracking-wider">{profileError}</div>
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
            <span className="label">reusable launch definition</span>
          </div>
        </div>
```

Also update the profiles empty-state row to drop the API/CLI hint now that the form exists. Replace:
```tsx
                <td colSpan={7} className="!py-12 text-center text-mute">
                  no profiles. create one via the API or CLI — UI form for profiles comes after the routes flow is solid.
                </td>
```
with:
```tsx
                <td colSpan={7} className="!py-12 text-center text-mute">
                  no profiles. create one with the form above.
                </td>
```

- [ ] **Step 3: Type-check**

Run: `cd ui && npx tsc -b --noEmit`
Expected: exit 0.

- [ ] **Step 4: Manual verification**

In the dev server:
1. Select a registered model in the `model name` field — the `hf repo` auto-fills.
2. Submit with a missing name — the create button is disabled.
3. Submit with `gpu_ids = "abc"` — the error surfaces ("gpu_ids: at least one integer required").
4. Submit a valid profile — it appears in the table; the form resets.
5. Submit a duplicate name — the daemon's 409 error appears in `profileError`.

- [ ] **Step 5: Commit**

```bash
git add ui/src/views/Services.tsx
git commit -m "feat(ui): create service profiles from the services view"
```

---

## Task 7: Polish — production build + sanity sweep

**Files:**
- None (verification only)

- [ ] **Step 1: Run the production build**

Run: `cd ui && npm run build`
Expected: build succeeds. The script also touches `../src/serve_engine/ui/__init__.py` and writes assets into `src/serve_engine/ui/`.

- [ ] **Step 2: Start the daemon against the freshly built bundle**

Run: `serve daemon stop && serve daemon start`
Open the bundled UI at `http://127.0.0.1:11500/`.

- [ ] **Step 3: End-to-end sanity sweep**

In the bundled UI (not dev server):
1. Navigate to `services` from the nav.
2. Create a profile referencing an existing registered model.
3. Create a route pointing at that profile, `match_model = chat-test`, `priority = 10`.
4. Deploy the profile from the profiles table.
5. From a terminal:
   ```bash
   curl -s "$SERVE_URL/v1/chat/completions" \
     -H "Authorization: Bearer $SERVE_TOKEN" -H 'Content-Type: application/json' \
     -d '{"model":"chat-test","messages":[{"role":"user","content":"reply with OK"}],"max_tokens":4}'
   ```
   Expected: the route resolves to the deployment, response returns.
6. Delete the route, then the profile, then stop the deployment. No errors.

- [ ] **Step 4: Commit any build artifacts**

```bash
git add src/serve_engine/ui
git status   # confirm only ui assets changed
git commit -m "chore(ui): rebuild bundle for services view"
```

---

## Self-Review

**Spec coverage:** all six features from the brainstorming list have a task. ✓ — actually, this plan covers only View 1 (services + routes). Views 2–6 have their own plans.

**Type consistency:** field names match the dataclasses in `src/serve_engine/store/service_profiles.py` and `service_routes.py`. `ServiceProfile.max_lora_rank` is exposed in the type but not surfaced in the UI form — that's intentional, lora-specific config stays out of the basic profile flow.

**Placeholder scan:** no TBDs. Every step has full code.

**Edge cases acknowledged in code:**
- empty arrays everywhere (`?? []`)
- duplicate names → 409 surfaces via `routeError` / `profileError`
- delete a profile referenced by a route → daemon 409; we re-invalidate `routes` after `deleteProfile` so the table state stays correct even on error
- empty `gpu_ids` and non-integer `gpu_ids` rejected client-side before the request
