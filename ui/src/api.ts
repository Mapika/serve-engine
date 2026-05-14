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

export async function eventSourceUrl(path: string): Promise<string> {
  if (!getToken()) return path
  const ticket = await api.createStreamToken()
  const sep = path.includes('?') ? '&' : '?'
  return `${path}${sep}stream_token=${encodeURIComponent(ticket.token)}`
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
  createStreamToken: () => jfetch<{ token: string; expires_at: number }>('POST', '/admin/stream-token'),

  // Adapter endpoints.
  listAdapters: () => jfetch<any[]>('GET', '/admin/adapters'),
  createAdapter: (b: { name: string; base_model_name: string; hf_repo: string; revision?: string }) =>
    jfetch<any>('POST', '/admin/adapters', b),
  downloadAdapter: (name: string) => jfetch<any>('POST', `/admin/adapters/${name}/download`),
  addLocalAdapter: (b: { name: string; base_model_name: string; local_path: string }) =>
    jfetch<any>('POST', '/admin/adapters/local', b),
  deleteAdapter: (name: string, force = false) =>
    jfetch<void>('DELETE', `/admin/adapters/${name}${force ? '?force=true' : ''}`),
  hotLoadAdapter: (depId: number, name: string) =>
    jfetch<any>('POST', `/admin/deployments/${depId}/adapters/${name}`),
  hotUnloadAdapter: (depId: number, name: string) =>
    jfetch<void>('DELETE', `/admin/deployments/${depId}/adapters/${name}`),

  // Predictor endpoints.
  predictorCandidates: () => jfetch<any[]>('GET', '/admin/predictor/candidates'),
  predictorStats: () => jfetch<any>('GET', '/admin/predictor/stats'),
}
