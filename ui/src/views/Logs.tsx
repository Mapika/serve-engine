import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, eventSourceUrl } from '../api'

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
  const logRef = useRef<HTMLDivElement | null>(null)
  const stickToBottom = useRef(true)

  useEffect(() => {
    if (selected !== null) return
    const list = deps.data ?? []
    const active = list.find((d: any) => d.status === 'ready' || d.status === 'loading')
    if (active) setSelected(active.id)
    else if (list.length > 0) setSelected(list[list.length - 1].id)
  }, [deps.data, selected])

  useEffect(() => {
    if (selected === null) return
    setLines([])
    setStreamError('')
    setStreaming(true)
    let closed = false
    let es: EventSource | null = null
    eventSourceUrl(`/admin/deployments/${selected}/logs/stream`)
      .then(url => {
        if (closed) return
        es = new EventSource(url)
        es.onmessage = (e: MessageEvent) => {
          setLines(prev => {
            const next = [...prev, e.data]
            return next.length > 2000 ? next.slice(-2000) : next
          })
        }
        es.onerror = () => {
          setStreaming(false)
          setStreamError('stream closed (container stopped or auth failed)')
          es?.close()
        }
      })
      .catch((e: Error) => {
        setStreaming(false)
        setStreamError(e.message)
      })
    return () => {
      closed = true
      setStreaming(false)
      es?.close()
    }
  }, [selected])

  useEffect(() => {
    let closed = false
    let es: EventSource | null = null
    eventSourceUrl('/admin/events').then(url => {
      if (closed) return
      es = new EventSource(url)
      es.onmessage = (e: MessageEvent) => {
        try {
          const obj = JSON.parse(e.data) as LifecycleEvent
          setEvents(prev => [obj, ...prev].slice(0, 50))
        } catch { /* ignore */ }
      }
      es.onerror = () => es?.close()
    }).catch(() => {
      // Polling views still work; event feed is best-effort.
    })
    return () => {
      closed = true
      es?.close()
    }
  }, [])

  useEffect(() => {
    if (!logRef.current || !stickToBottom.current) return
    logRef.current.scrollTop = logRef.current.scrollHeight
  }, [lines])

  function onLogScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight
    stickToBottom.current = dist < 60
  }

  const visibleDeps = (deps.data ?? []).filter(
    (d: any) => d.container_id !== null && d.container_id !== undefined,
  )

  return (
    <div className="space-y-10">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">logs</h2>
        <div className="flex items-center gap-3">
          {streaming && (
            <span className="flex items-center text-[11px] tracking-wider text-accent">
              <span className="dot dot-loading" />live
            </span>
          )}
          {streamError && (
            <span className="text-err text-[11px] tracking-wider">{streamError}</span>
          )}
        </div>
      </header>

      <section className="space-y-3">
        <div className="label">container</div>
        <select
          className="field font-mono text-[13px] min-w-[420px]"
          value={selected ?? ''}
          onChange={e => setSelected(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">— select deployment —</option>
          {visibleDeps.map((d: any) => {
            const m = (models.data ?? []).find((m: any) => m.id === d.model_id)
            return (
              <option key={d.id} value={d.id}>
                #{d.id}  ·  {m?.name ?? d.model_id}  ·  {d.status}
              </option>
            )
          })}
        </select>
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="label">stdout</div>
          <div className="text-mute text-[10px] tracking-wider tnum">{lines.length} lines</div>
        </div>
        <div
          ref={logRef}
          onScroll={onLogScroll}
          className="bg-bg-page border border-rule font-mono text-[11.5px] leading-relaxed text-ink/80 overflow-y-auto h-[32rem] p-4"
          style={{ background: '#08080a' }}
        >
          {selected === null && (
            <div className="text-mute">select a deployment above to tail its container logs</div>
          )}
          {selected !== null && lines.length === 0 && streaming && (
            <div className="text-mute">waiting for output<span className="caret"></span></div>
          )}
          {lines.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
          ))}
        </div>
      </section>

      <details className="border border-rule">
        <summary className="cursor-pointer px-4 py-3 text-dim text-[11px] tracking-wider hover:text-ink transition-colors select-none flex items-center justify-between">
          <span>lifecycle events</span>
          <span className="text-mute tnum">{events.length}</span>
        </summary>
        <div className="px-4 pb-4 text-[11px] font-mono space-y-1 max-h-64 overflow-y-auto">
          {events.length === 0 && <div className="text-mute">no events yet</div>}
          {events.map((e, i) => (
            <div key={i} className="flex gap-3">
              <span className="text-mute tnum w-16 shrink-0">{e.ts.slice(11, 19)}</span>
              <span className="text-accent shrink-0">{e.kind}</span>
              <span className="text-dim break-all">{JSON.stringify(e.payload)}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}
