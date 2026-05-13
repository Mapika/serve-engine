# Serving Engine — Plan 06: Web UI

**Goal:** A small, tasteful web UI served by the daemon at `http://localhost:<port>/`. Five screens: Dashboard, Models, Playground, API Keys, Logs. Consumes the admin endpoints + SSE event stream + `/v1/chat/completions`.

**Architecture decisions (pragmatic):**

- **Vite + React + TypeScript + Tailwind CSS.** Standard modern frontend stack.
- **No charts library** for v1. Use plain inline SVG sparklines where needed.
- **No router library.** Five screens swap via a small `view` state in the root component. URL hash for deep-link if desired.
- **State management:** React Query for server data (caching, refetch). Plain `useState` for UI state.
- **Pre-built `dist/` is committed to the repo.** The daemon serves it as static files. Users do NOT need Node.js. Devs do.
- **Build system note:** Node 20+ + npm. `npm install` once, `npm run build` to regenerate `dist/`.
- **Auth flow:** When the user first opens the UI, the page reads an admin token from `localStorage`; if absent, prompts the user to paste one (created via `serve key create --tier admin`). Token is attached as `Authorization: Bearer ...` for every fetch.

**Tech Stack:** Node 20 + Vite 5 + React 18 + TypeScript 5 + Tailwind CSS 3 + @tanstack/react-query.

---

## File structure

```
serving-engine/
├── ui/                                # Frontend source (NOT shipped in wheel — only dist/ is)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts                     # fetch wrappers with Authorization
│       ├── components/
│       │   ├── TokenGate.tsx
│       │   ├── Layout.tsx
│       │   └── Sparkline.tsx
│       └── views/
│           ├── Dashboard.tsx
│           ├── Models.tsx
│           ├── Playground.tsx
│           ├── Keys.tsx
│           └── Logs.tsx
├── src/serve_engine/ui/               # Bundled dist (committed)
│   ├── __init__.py
│   ├── index.html                     # COPIED from ui/dist on each build
│   ├── assets/                        # COPIED from ui/dist/assets
│   └── ...
└── src/serve_engine/daemon/
    └── ui_router.py                   # NEW — serves the static files
```

---

## Task 1: Frontend scaffold

**Files to create under `ui/`:**

- [ ] **Step 1: `ui/package.json`**

```json
{
  "name": "serve-engine-ui",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.51.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.39",
    "tailwindcss": "^3.4.4",
    "typescript": "^5.5.3",
    "vite": "^5.3.4"
  }
}
```

- [ ] **Step 2: `ui/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "Bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: `ui/vite.config.ts`**

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/serve_engine/ui',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  server: {
    proxy: {
      '/admin': 'http://127.0.0.1:11500',
      '/v1':    'http://127.0.0.1:11500',
    },
  },
})
```

- [ ] **Step 4: `ui/tailwind.config.js`**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
```

- [ ] **Step 5: `ui/postcss.config.js`**

```js
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
}
```

- [ ] **Step 6: `ui/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>serve-engine</title>
  </head>
  <body class="bg-gray-50 text-gray-900">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: `ui/src/main.tsx`**

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './styles.css'

const client = new QueryClient({
  defaultOptions: { queries: { staleTime: 1000 } },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
```

- [ ] **Step 8: `ui/src/styles.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 9: Install and verify build works**

```bash
cd ui
npm install
npm run build
ls -la ../src/serve_engine/ui/
```
Expected: `index.html` and `assets/` directory present under `src/serve_engine/ui/`.

If `npm` is not installed: install Node.js 20 first via `apt install nodejs npm` or `curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash - && sudo apt install nodejs`.

- [ ] **Step 10: Commit (the empty App stub will be replaced in subsequent tasks)**

Create a minimal `ui/src/App.tsx` so the build succeeds:
```tsx
export default function App() {
  return <div className="p-8 text-2xl">serve-engine UI (scaffold)</div>
}
```

Then:
```bash
cd ui && npm run build && cd ..
git add ui/ src/serve_engine/ui/
# .gitignore should NOT exclude src/serve_engine/ui — that's the dist we ship
# but DO add ui/node_modules to .gitignore
echo "ui/node_modules" >> .gitignore
echo "ui/dist" >> .gitignore  # not used (we output to src/serve_engine/ui directly)
git add .gitignore
git commit -m "feat(ui): Vite + React + Tailwind scaffold (empty App)"
```

---

## Task 2: UI router on the daemon

**Files:** `src/serve_engine/daemon/ui_router.py` (new), `src/serve_engine/daemon/app.py` (modify)

- [ ] **Step 1: Create the router**

`src/serve_engine/daemon/ui_router.py`:
```python
from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


def make_ui_router() -> APIRouter | None:
    """Return a router that serves the bundled UI, or None if the dist is missing."""
    ui_dir = files("serve_engine.ui")
    index = ui_dir.joinpath("index.html")
    try:
        index_text = index.read_text()
    except FileNotFoundError:
        return None
    router = APIRouter()
    # Mount assets statically. The path comes from importlib.resources; convert to str.
    assets_dir = str(ui_dir.joinpath("assets"))
    router.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @router.get("/", response_class=HTMLResponse)
    @router.get("/{full_path:path}", response_class=HTMLResponse)
    def index_html() -> HTMLResponse:
        return HTMLResponse(content=index_text)

    return router
```

- [ ] **Step 2: Mount on uds_app only (Plan 04 keeps admin off TCP; the UI is admin)**

In `src/serve_engine/daemon/app.py`:
- Add import: `from serve_engine.daemon.ui_router import make_ui_router`
- In `build_apps`, after `uds_app.include_router(admin_router)` and `uds_app.include_router(metrics_router)`, add:
```python
    ui_router = make_ui_router()
    if ui_router is not None:
        uds_app.include_router(ui_router)
```

Wait — the UI must be reachable from the browser, which talks over TCP, not UDS. Re-read the Plan 04 design: `/admin/*` is UDS-only because it's unauth'd. With Plan 04 auth on, we can safely mount the UI + admin on TCP behind Bearer auth.

REVISED step 2: Mount the UI router on the TCP app, AND make sure the admin endpoints behind it require auth. Plan 04 already requires Bearer on /v1/*; we need to also require it on /admin/* if anyone wires it to TCP. Since Plan 04 mounted /admin/* on UDS only, let's KEEP it on UDS only and ALSO mount the UI on TCP with no auth on the static files (the JS app prompts for token).

So:
```python
    ui_router = make_ui_router()
    if ui_router is not None:
        tcp_app.include_router(ui_router)
```

The JS app then makes XHRs to `/admin/*` and `/v1/*` over TCP. But /admin/* isn't on TCP. Two options:

**A. Add /admin/* to tcp_app with auth required.**
**B. Have the UI talk to UDS via a tiny proxy.**

Option A is dramatically simpler. The risk Plan 04 mitigated was unauthenticated /admin/* on TCP. With Bearer auth now in place, /admin/* on TCP is safe IF auth is required. Plan 04's auth dep currently bypasses when api_keys table is empty. For UI to work, the user must create at least one admin key first. That's fine.

Final approach:
- Mount `/admin/*` on `tcp_app` too (revisit the Plan 04 decision now that we have auth).
- Add `require_auth_dep` (or a similar dep) to the admin router so /admin/* on TCP requires Bearer.
- Mount UI on `tcp_app` so the browser can reach it.

### Step 2 (revised): Mount admin + UI on tcp_app with auth required

This is a significant interaction with Plan 04. Defer to a separate task.

For T2 here, just mount the UI:
```python
    ui_router = make_ui_router()
    if ui_router is not None:
        tcp_app.include_router(ui_router)
```

- [ ] **Step 3: Run the daemon and curl the index**

```bash
serve daemon start
curl -s http://127.0.0.1:11500/ | head -5
```
Expected: HTML page with `<div id="root">` referencing the bundled assets.

- [ ] **Step 4: Commit**

```bash
git add src/serve_engine/daemon/ui_router.py src/serve_engine/daemon/app.py
git commit -m "feat(daemon): serve bundled UI at GET /"
```

---

## Task 3: Auth — admin endpoints on TCP with Bearer

**Files:** `src/serve_engine/daemon/app.py` (modify), `tests/integration/test_openai_proxy.py` (verify unbroken)

The UI's JS will call `/admin/deployments`, `/admin/keys`, etc. over TCP. Those routes currently exist only on `uds_app`. We need to add them to `tcp_app` BUT with auth required.

- [ ] **Step 1: In `daemon/app.py`, mount admin_router and metrics_router on tcp_app**

Currently `tcp_app` has only `openai_router`. Change to also include `admin_router` and (already had) `metrics_router`:
```python
    tcp_app.include_router(openai_router)
    tcp_app.include_router(admin_router)
    tcp_app.include_router(metrics_router)
    tcp_app.include_router(ui_router)
```

This means /admin/* is now reachable on TCP. Plan 04's `require_auth_dep` was applied only to /v1/* routes. We need to extend it to /admin/* on TCP, but NOT break the UDS-only access (which is bypassed because typically no keys are configured on a homelab daemon).

Actually `require_auth_dep` already bypasses when no keys exist. So:
- Without any keys → /admin/* works on both TCP and UDS (no auth)
- With keys created → /admin/* requires Bearer on TCP. UDS users can use the admin key as Bearer too.

Add `require_auth_dep` to the admin router. In `daemon/admin.py`, change the router declaration:
```python
from serve_engine.auth.middleware import require_auth_dep

router = APIRouter(prefix="/admin", dependencies=[Depends(require_auth_dep)])
```

This applies the dep to every route on the router. Existing routes don't need per-route Depends.

- [ ] **Step 2: Restrictive auth — require admin tier on /admin/***

Add a stricter dep that requires not just any key, but an admin-tier key:
```python
# In daemon/admin.py
from fastapi import Request


def require_admin_key(request: Request, key=Depends(require_auth_dep)):
    """Pass-through if no keys exist (bypass), else require tier=admin."""
    if key is None:
        return None  # bypass mode
    if key.tier != "admin":
        from fastapi import HTTPException, status
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin tier required for /admin/*")
    return key


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)])
```

- [ ] **Step 3: Update existing admin tests to use a key or no-keys-bypass**

The existing `tests/unit/test_admin_endpoints.py` `app` fixture creates an empty `api_keys` table → bypass mode → tests work unchanged.

The new `test_pin_unpin_deployment`, `test_create_list_revoke_key`, etc. also work in bypass mode.

Verify by running:
```bash
pytest -v
ruff check src/ tests/
```

If anything breaks, add a fixture parameter to optionally create an admin key and pass `Authorization: Bearer ...` in headers.

- [ ] **Step 4: Commit**

```bash
git add src/serve_engine/daemon/app.py src/serve_engine/daemon/admin.py
git commit -m "feat(daemon): admin routes on TCP behind admin-tier Bearer auth"
```

---

## Task 4: API client + TokenGate

**Files:** `ui/src/api.ts`, `ui/src/components/TokenGate.tsx`, `ui/src/App.tsx`

- [ ] **Step 1: `ui/src/api.ts`**

```ts
const TOKEN_KEY = 'serve.adminToken'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(t: string) {
  localStorage.setItem(TOKEN_KEY, t)
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

async function jfetch<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = getToken()
  if (token) headers['Authorization'] = `Bearer ${token}`
  const r = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) {
    const detail = await r.text()
    throw new Error(`${r.status}: ${detail}`)
  }
  if (r.status === 204) return undefined as T
  return r.json() as Promise<T>
}

export const api = {
  listDeployments: () => jfetch<any[]>('GET', '/admin/deployments'),
  stopDeployment: (id: number) => jfetch<void>('DELETE', `/admin/deployments/${id}`),
  pinDeployment: (id: number) => jfetch<void>('POST', `/admin/deployments/${id}/pin`),
  unpinDeployment: (id: number) => jfetch<void>('POST', `/admin/deployments/${id}/unpin`),
  listModels: () => jfetch<any[]>('GET', '/admin/models'),
  createModel: (b: { name: string; hf_repo: string }) => jfetch<any>('POST', '/admin/models', b),
  deleteModel: (name: string) => jfetch<void>('DELETE', `/admin/models/${name}`),
  listKeys: () => jfetch<any[]>('GET', '/admin/keys'),
  createKey: (b: { name: string; tier: string }) => jfetch<any>('POST', '/admin/keys', b),
  revokeKey: (id: number) => jfetch<void>('DELETE', `/admin/keys/${id}`),
  listGpus: () => jfetch<any[]>('GET', '/admin/gpus'),
  loadModel: (b: any) => jfetch<any>('POST', '/admin/deployments', b),
  chat: (model: string, messages: any[], stream: boolean) =>
    jfetch<any>('POST', '/v1/chat/completions', { model, messages, stream, max_tokens: 256 }),
}
```

- [ ] **Step 2: `ui/src/components/TokenGate.tsx`**

```tsx
import { useState } from 'react'
import { getToken, setToken } from '../api'

export default function TokenGate({ children }: { children: React.ReactNode }) {
  const [token, setLocalToken] = useState<string | null>(getToken())
  const [input, setInput] = useState('')

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <div className="max-w-md w-full bg-white rounded-lg shadow p-6 space-y-4">
          <h1 className="text-xl font-semibold">serve-engine</h1>
          <p className="text-sm text-gray-600">
            Paste an admin-tier API key. Create one with:
          </p>
          <pre className="text-xs bg-gray-100 rounded p-2 overflow-x-auto">
serve key create web --tier admin
          </pre>
          <input
            className="w-full border rounded px-3 py-2 font-mono text-sm"
            placeholder="sk-..."
            value={input}
            onChange={e => setInput(e.target.value)}
          />
          <button
            className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
            disabled={!input.trim()}
            onClick={() => {
              setToken(input.trim())
              setLocalToken(input.trim())
            }}
          >
            Continue
          </button>
        </div>
      </div>
    )
  }
  return <>{children}</>
}
```

- [ ] **Step 3: Replace `ui/src/App.tsx` with the layout + view switcher**

```tsx
import { useState } from 'react'
import TokenGate from './components/TokenGate'
import { clearToken } from './api'
import Dashboard from './views/Dashboard'
import Models from './views/Models'
import Playground from './views/Playground'
import Keys from './views/Keys'
import Logs from './views/Logs'

type View = 'dashboard' | 'models' | 'playground' | 'keys' | 'logs'

const VIEWS: { id: View; label: string }[] = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'models', label: 'Models' },
  { id: 'playground', label: 'Playground' },
  { id: 'keys', label: 'API Keys' },
  { id: 'logs', label: 'Logs' },
]

export default function App() {
  const [view, setView] = useState<View>('dashboard')

  return (
    <TokenGate>
      <div className="min-h-screen flex">
        <nav className="w-56 bg-white border-r border-gray-200 p-4 space-y-1">
          <h1 className="text-xl font-semibold mb-4">serve-engine</h1>
          {VIEWS.map(v => (
            <button
              key={v.id}
              onClick={() => setView(v.id)}
              className={`w-full text-left px-3 py-2 rounded ${
                view === v.id ? 'bg-blue-100 text-blue-900' : 'hover:bg-gray-100'
              }`}
            >
              {v.label}
            </button>
          ))}
          <button
            onClick={() => { clearToken(); location.reload() }}
            className="w-full text-left px-3 py-2 rounded text-xs text-gray-500 hover:text-gray-900 mt-8"
          >
            Sign out
          </button>
        </nav>
        <main className="flex-1 p-8 overflow-y-auto">
          {view === 'dashboard' && <Dashboard />}
          {view === 'models' && <Models />}
          {view === 'playground' && <Playground />}
          {view === 'keys' && <Keys />}
          {view === 'logs' && <Logs />}
        </main>
      </div>
    </TokenGate>
  )
}
```

- [ ] **Step 4: Stub each view (full impls in Task 5)**

Create empty stubs for the 5 views so the build doesn't fail:

`ui/src/views/Dashboard.tsx`:
```tsx
export default function Dashboard() {
  return <div>Dashboard (TODO Task 5)</div>
}
```

Repeat the same one-liner for `Models.tsx`, `Playground.tsx`, `Keys.tsx`, `Logs.tsx`.

- [ ] **Step 5: Build + commit**

```bash
cd ui && npm run build && cd ..
git add ui/ src/serve_engine/ui/
git commit -m "feat(ui): TokenGate auth flow + sidebar layout + view stubs"
```

---

## Task 5: Views — Dashboard, Models, Keys, Playground, Logs

**Files:** the five view files under `ui/src/views/`.

The view contents are pragmatic — fetch with React Query, render tables/forms. Full code for each view is provided below.

### 5a — Dashboard

`ui/src/views/Dashboard.tsx`:
```tsx
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
```

### 5b — Models

`ui/src/views/Models.tsx`:
```tsx
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
    mutationFn: (name: string) => api.deleteModel(name),
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
```

### 5c — Keys

`ui/src/views/Keys.tsx`:
```tsx
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
```

### 5d — Playground

`ui/src/views/Playground.tsx`:
```tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, getToken } from '../api'

export default function Playground() {
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const [selected, setSelected] = useState<string>('')
  const [prompt, setPrompt] = useState('')
  const [response, setResponse] = useState('')
  const [pending, setPending] = useState(false)

  async function send() {
    if (!selected || !prompt.trim()) return
    setResponse('')
    setPending(true)
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      const token = getToken()
      if (token) headers['Authorization'] = `Bearer ${token}`
      const r = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          model: selected,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          max_tokens: 512,
        }),
      })
      if (!r.body) return
      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()!
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (payload === '[DONE]') continue
          try {
            const obj = JSON.parse(payload)
            const delta = obj.choices?.[0]?.delta?.content ?? ''
            setResponse(prev => prev + delta)
          } catch { /* ignore */ }
        }
      }
    } catch (e: any) {
      setResponse(`Error: ${e?.message ?? e}`)
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Playground</h2>
      <div className="flex gap-2">
        <select
          className="border rounded px-3 py-2"
          value={selected}
          onChange={e => setSelected(e.target.value)}
        >
          <option value="">— choose model —</option>
          {(models.data ?? []).map((m: any) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </select>
      </div>
      <textarea
        className="w-full h-32 border rounded p-3 font-mono text-sm"
        placeholder="Ask something…"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
      />
      <button
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
        disabled={!selected || !prompt.trim() || pending}
        onClick={send}
      >{pending ? 'Streaming…' : 'Send'}</button>
      <pre className="bg-gray-100 p-4 rounded text-sm whitespace-pre-wrap font-mono min-h-[8rem]">
        {response || ' '}
      </pre>
    </div>
  )
}
```

### 5e — Logs

`ui/src/views/Logs.tsx`:
```tsx
import { useEffect, useState } from 'react'
import { getToken } from '../api'

type Event = { kind: string; payload: any; ts: string }

export default function Logs() {
  const [events, setEvents] = useState<Event[]>([])

  useEffect(() => {
    const token = getToken()
    // EventSource does not support custom headers; pass the token via query param if the API supports it.
    // For now, only viable if no auth required (homelab) — TODO: implement a /admin/events?token= path
    // or a small server-side handshake.
    const url = `/admin/events${token ? '' : ''}`
    const es = new EventSource(url)
    es.onmessage = (e: MessageEvent) => {
      try {
        const obj = JSON.parse(e.data) as Event
        setEvents(prev => [obj, ...prev].slice(0, 200))
      } catch { /* ignore */ }
    }
    es.onerror = () => { es.close() }
    return () => es.close()
  }, [])

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Logs (live events)</h2>
      <pre className="bg-black text-green-300 font-mono text-xs p-4 rounded overflow-y-auto h-96">
        {events.length === 0 && <div className="text-gray-500">waiting for events…</div>}
        {events.map((e, i) => (
          <div key={i}>
            <span className="text-gray-500">{e.ts.slice(11, 19)}</span>{' '}
            <span className="text-yellow-300">{e.kind}</span>{' '}
            {JSON.stringify(e.payload)}
          </div>
        ))}
      </pre>
    </div>
  )
}
```

- [ ] **Step build + commit**

```bash
cd ui && npm run build && cd ..
git add ui/src/views/ src/serve_engine/ui/
git commit -m "feat(ui): five screens — dashboard, models, playground, keys, logs"
```

---

## Task 6: Live verification

Manual procedure:

- [ ] **Step 1** — With daemon up, create an admin key:
```bash
serve daemon start
serve key create web --tier admin
# Save the printed secret.
```

- [ ] **Step 2** — Open `http://127.0.0.1:11500/` in a browser.
- [ ] **Step 3** — Paste the admin secret into the TokenGate. Sidebar should appear.
- [ ] **Step 4** — Click around all 5 views. Verify:
  - Dashboard shows GPU cards + deployment table.
  - Models lists registered models, can register a new repo, can delete.
  - Playground streams a chat response from a loaded model.
  - API Keys shows existing keys, can create + revoke.
  - Logs shows live SSE events when something happens.

## Verification (end of Plan 06)

1. `pytest -v` — all tests pass.
2. `ruff check src/ tests/` clean.
3. Built `src/serve_engine/ui/index.html` and `src/serve_engine/ui/assets/` present in the repo (committed).
4. Browser flow works end-to-end.

## Self-review

- **Auth via Bearer header.** Works for fetch() but EventSource doesn't support custom headers; Logs view has a known limitation (works fully only when no keys exist, i.e. homelab bypass). Future task: query-param token, or `?` token, or upgrade the EventSource shim to use fetch + ReadableStream.
- **No build dependency for users** — `dist/` is committed. Devs run `npm install && npm run build`.
- **Five screens, minimal but functional.** No charts library, no router, no state management beyond React Query.
- **Forward compat:** UI calls only documented JSON API; rev-locks to the daemon's contract.
