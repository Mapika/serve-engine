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
            // vLLM's --reasoning-parser=qwen3 emits reasoning under
            // delta.reasoning (some builds use delta.reasoning_content).
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
      if (e?.name !== 'AbortError') {
        setError(`error: ${e?.message ?? e}`)
      }
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

  function stop() {
    abortRef.current?.abort()
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Playground</h2>
      <div className="flex gap-2 items-center flex-wrap">
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
        <label className="flex items-center gap-2 text-sm">
          <span className="text-gray-600">max_tokens</span>
          <input
            type="number"
            className="border rounded px-2 py-1 w-24 font-mono"
            value={maxTokens}
            min={32}
            max={32768}
            step={256}
            onChange={e => setMaxTokens(Number(e.target.value) || 4096)}
          />
        </label>
      </div>
      <textarea
        className="w-full h-32 border rounded p-3 font-mono text-sm"
        placeholder="Ask something…"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
      />
      <div className="flex gap-2 items-center">
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          disabled={!selected || !prompt.trim() || pending}
          onClick={send}
        >{pending ? 'Streaming…' : 'Send'}</button>
        {pending && (
          <button
            className="bg-gray-200 px-4 py-2 rounded"
            onClick={stop}
          >Stop</button>
        )}
        {(stats.ttftMs !== null || stats.totalMs !== null) && (
          <span className="text-sm text-gray-600 font-mono ml-2">
            {stats.ttftMs !== null && <>ttft {stats.ttftMs}ms</>}
            {stats.totalMs !== null && <> · total {stats.totalMs}ms</>}
            {stats.tps !== null && <> · {stats.tps} tok/s</>}
            {stats.tokens > 0 && <> · {stats.tokens} tokens</>}
          </span>
        )}
      </div>

      {error && (
        <pre className="bg-red-50 border border-red-200 text-red-800 p-3 rounded text-sm whitespace-pre-wrap font-mono">
          {error}
        </pre>
      )}

      {reasoning && (
        <details
          open={showThinking}
          onToggle={e => setShowThinking((e.target as HTMLDetailsElement).open)}
          className="bg-gray-50 border border-gray-200 rounded"
        >
          <summary className="cursor-pointer px-3 py-2 text-sm text-gray-600 select-none">
            Thinking ({reasoning.length.toLocaleString()} chars)
          </summary>
          <pre className="px-3 pb-3 text-xs text-gray-600 whitespace-pre-wrap font-mono">
            {reasoning}
          </pre>
        </details>
      )}

      <pre className="bg-white border border-gray-200 p-4 rounded text-sm whitespace-pre-wrap font-mono min-h-[8rem]">
        {answer || (pending && !reasoning ? 'waiting for tokens…' : !pending && !reasoning && !answer && !error ? ' ' : answer)}
      </pre>
    </div>
  )
}
