import { useEffect, useState } from 'react'

type Event = { kind: string; payload: any; ts: string }

export default function Logs() {
  const [events, setEvents] = useState<Event[]>([])

  useEffect(() => {
    // Note: EventSource does not support custom headers, so we cannot pass
    // a Bearer token here. /admin/events is admin-gated; this view will only
    // receive data when the api_keys table is empty (homelab bypass).
    // A future task will introduce a token-in-query-param variant.
    const es = new EventSource('/admin/events')
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
