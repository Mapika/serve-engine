import { useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, getToken } from '../api'

type Stats = {
  ttftMs: number | null
  totalMs: number | null
  tokens: number
  tps: number | null
}

export default function Playground() {
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const [selected, setSelected] = useState<string>('')
  const [prompt, setPrompt] = useState('')
  const [reasoning, setReasoning] = useState('')
  const [answer, setAnswer] = useState('')
  const [error, setError] = useState('')
  const [stats, setStats] = useState<Stats>({ ttftMs: null, totalMs: null, tokens: 0, tps: null })
  const [pending, setPending] = useState(false)
  const [showThinking, setShowThinking] = useState(true)
  const [maxTokens, setMaxTokens] = useState(4096)
  const abortRef = useRef<AbortController | null>(null)

  async function send() {
    if (!selected || !prompt.trim()) return
    setReasoning('')
    setAnswer('')
    setError('')
    setStats({ ttftMs: null, totalMs: null, tokens: 0, tps: null })
    setPending(true)
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    const t0 = performance.now()
    let firstTokenAt: number | null = null
    let tokenCount = 0

    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      const token = getToken()
      if (token) headers['Authorization'] = `Bearer ${token}`
      const r = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers,
        signal: ctrl.signal,
        body: JSON.stringify({
          model: selected,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
          max_tokens: maxTokens,
        }),
      })
      if (!r.ok) {
        const body = await r.text()
        setError(`${r.status}: ${body.slice(0, 500)}`)
        return
      }
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
            const delta = obj.choices?.[0]?.delta ?? {}
            const rd: string = delta.reasoning ?? delta.reasoning_content ?? ''
            const cd: string = delta.content ?? ''
            if ((rd || cd) && firstTokenAt === null) {
              firstTokenAt = performance.now()
              setStats(s => ({ ...s, ttftMs: Math.round(firstTokenAt! - t0) }))
            }
            if (rd) { setReasoning(prev => prev + rd); tokenCount += 1 }
            if (cd) { setAnswer(prev => prev + cd); tokenCount += 1 }
          } catch { /* skip malformed lines */ }
        }
      }
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(`error: ${e?.message ?? e}`)
    } finally {
      const t1 = performance.now()
      setStats(s => ({
        ...s,
        totalMs: Math.round(t1 - t0),
        tokens: tokenCount,
        tps: tokenCount && t1 > t0 ? Math.round((tokenCount / (t1 - t0)) * 10000) / 10 : null,
      }))
      setPending(false)
    }
  }

  function stop() { abortRef.current?.abort() }

  return (
    <div className="space-y-10">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">playground</h2>
        <div className="label">openai-compatible</div>
      </header>

      <section className="space-y-6">
        <div className="grid grid-cols-[1fr_180px] gap-4">
          <div className="space-y-2">
            <div className="label">model</div>
            <select
              className="field w-full font-mono"
              value={selected}
              onChange={e => setSelected(e.target.value)}
            >
              <option value="">choose model</option>
              {(models.data ?? []).map((m: any) => (
                <option key={m.name} value={m.name}>{m.name}</option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <div className="label">max tokens</div>
            <input
              type="number"
              className="field w-full font-mono tnum"
              value={maxTokens}
              min={32}
              max={32768}
              step={256}
              onChange={e => setMaxTokens(Number(e.target.value) || 4096)}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="label">prompt</div>
          <textarea
            className="field w-full font-mono text-[13px]"
            style={{ minHeight: '8rem' }}
            placeholder="ask something"
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && !pending) send()
            }}
          />
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                className="btn-primary"
                disabled={!selected || !prompt.trim() || pending}
                onClick={send}
              >
                {pending ? 'streaming...' : 'send'}
              </button>
              {pending && (
                <button className="btn" onClick={stop}>stop</button>
              )}
              <span className="label">Ctrl+Enter</span>
            </div>
            {(stats.ttftMs !== null || stats.totalMs !== null) && (
              <div className="flex gap-6 text-mute text-[11px] tnum">
                {stats.ttftMs !== null && (
                  <div><span className="text-dim">ttft</span> {stats.ttftMs}ms</div>
                )}
                {stats.totalMs !== null && (
                  <div><span className="text-dim">total</span> {stats.totalMs}ms</div>
                )}
                {stats.tps !== null && (
                  <div><span className="text-dim">throughput</span> {stats.tps} tok/s</div>
                )}
                {stats.tokens > 0 && (
                  <div><span className="text-dim">tokens</span> {stats.tokens}</div>
                )}
              </div>
            )}
          </div>
        </div>

        {error && (
          <pre className="text-err text-[12px] whitespace-pre-wrap font-mono border border-err/30 px-3 py-2">
            {error}
          </pre>
        )}

        {reasoning && (
          <details
            open={showThinking}
            onToggle={e => setShowThinking((e.target as HTMLDetailsElement).open)}
            className="border border-rule"
          >
            <summary className="cursor-pointer px-4 py-3 text-dim text-[11px] tracking-wider hover:text-ink transition-colors select-none flex items-center gap-2">
              <span className="dot dot-loading" style={{ width: 5, height: 5 }} />
              thinking
              <span className="text-mute tnum">{reasoning.length.toLocaleString()}c</span>
            </summary>
            <pre className="px-4 pb-4 text-[12px] text-dim whitespace-pre-wrap font-mono leading-relaxed">
              {reasoning}
            </pre>
          </details>
        )}

        <div className="space-y-2">
          {(answer || (pending && !reasoning) || (!pending && !reasoning && !answer && !error)) && (
            <div className="label">response</div>
          )}
          {(answer || (pending && !reasoning)) && (
            <pre className="text-[13px] whitespace-pre-wrap font-mono leading-relaxed border-l border-accent pl-4">
              {answer || (pending ? <span className="text-mute">waiting for tokens...</span> : '')}
            </pre>
          )}
        </div>
      </section>
    </div>
  )
}
