import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, getToken } from '../api'

type LifecycleEvent = { kind: string; payload: any; ts: string }

export default function Logs() {
  const deps = useQuery({
    queryKey: ['deps'],
    queryFn: api.listDeployments,
    refetchInterval: 5000,
  })
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const [selected, setSelected] = useState<number | null>(null)
  const [lines, setLines] = useState<string[]>([])
  const [streaming, setStreaming] = useState(false)
  const [streamError, setStreamError] = useState('')
  const [events, setEvents] = useState<LifecycleEvent[]>([])
  const logRef = useRef<HTMLPreElement | null>(null)
  const stickToBottom = useRef(true)

  // Auto-select the most recently active deployment when the list loads.
  useEffect(() => {
    if (selected !== null) return
    const list = deps.data ?? []
    const active = list.find((d: any) => d.status === 'ready' || d.status === 'loading')
    if (active) {
      setSelected(active.id)
    } else if (list.length > 0) {
      setSelected(list[list.length - 1].id)
    }
  }, [deps.data, selected])

  // Stream container logs over SSE when a deployment is selected.
  useEffect(() => {
    if (selected === null) return
    setLines([])
    setStreamError('')
    setStreaming(true)
    const token = getToken()
    const url = token
      ? `/admin/deployments/${selected}/logs/stream?token=${encodeURIComponent(token)}`
      : `/admin/deployments/${selected}/logs/stream`
    const es = new EventSource(url)
    es.onmessage = (e: MessageEvent) => {
      setLines(prev => {
        const next = [...prev, e.data]
        // Cap history so the DOM doesn't grow unbounded.
        return next.length > 2000 ? next.slice(-2000) : next
      })
    }
    es.onerror = () => {
      setStreaming(false)
      setStreamError('stream closed (container may have stopped, or auth failed)')
      es.close()
    }
    return () => {
      setStreaming(false)
      es.close()
    }
  }, [selected])

  // Lifecycle event stream (separate from container logs).
  useEffect(() => {
    const token = getToken()
    const url = token
      ? `/admin/events?token=${encodeURIComponent(token)}`
      : '/admin/events'
    const es = new EventSource(url)
    es.onmessage = (e: MessageEvent) => {
      try {
        const obj = JSON.parse(e.data) as LifecycleEvent
        setEvents(prev => [obj, ...prev].slice(0, 50))
      } catch { /* ignore */ }
    }
    es.onerror = () => es.close()
    return () => es.close()
  }, [])

  // Auto-scroll while the user is at the bottom; pause when they scroll up.
  useEffect(() => {
    if (!logRef.current || !stickToBottom.current) return
    logRef.current.scrollTop = logRef.current.scrollHeight
  }, [lines])

  function onLogScroll(e: React.UIEvent<HTMLPreElement>) {
    const el = e.currentTarget
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight
    stickToBottom.current = dist < 50
  }

  const visibleDeps = (deps.data ?? []).filter(
    (d: any) => d.container_id !== null && d.container_id !== undefined,
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-2xl font-bold">Engine logs</h2>
        <select
          className="border rounded px-3 py-2 text-sm"
          value={selected ?? ''}
          onChange={e => setSelected(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">— select deployment —</option>
          {visibleDeps.map((d: any) => {
            const m = (models.data ?? []).find((m: any) => m.id === d.model_id)
            return (
              <option key={d.id} value={d.id}>
                #{d.id} · {m?.name ?? d.model_id} · {d.status}
              </option>
            )
          })}
        </select>
        {streaming && (
          <span className="text-xs text-green-700 bg-green-100 px-2 py-0.5 rounded">live</span>
        )}
        {streamError && (
          <span className="text-xs text-red-700 bg-red-100 px-2 py-0.5 rounded">{streamError}</span>
        )}
      </div>

      <pre
        ref={logRef}
        onScroll={onLogScroll}
        className="bg-black text-green-200 font-mono text-xs p-4 rounded overflow-y-auto h-[28rem] whitespace-pre-wrap"
      >
        {selected === null && (
          <div className="text-gray-500">select a deployment above to tail its container logs</div>
        )}
        {selected !== null && lines.length === 0 && streaming && (
          <div className="text-gray-500">waiting for output…</div>
        )}
        {lines.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
      </pre>

      <details className="bg-white border rounded">
        <summary className="cursor-pointer px-3 py-2 text-sm font-semibold select-none">
          Lifecycle events {events.length > 0 && <span className="text-gray-500">({events.length})</span>}
        </summary>
        <pre className="px-3 pb-3 text-xs font-mono whitespace-pre-wrap">
          {events.length === 0 && <div className="text-gray-500">no events yet</div>}
          {events.map((e, i) => (
            <div key={i}>
              <span className="text-gray-500">{e.ts.slice(11, 19)}</span>{' '}
              <span className="text-yellow-700">{e.kind}</span>{' '}
              <span className="text-gray-700">{JSON.stringify(e.payload)}</span>
            </div>
          ))}
        </pre>
      </details>
    </div>
  )
}
