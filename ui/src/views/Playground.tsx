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
