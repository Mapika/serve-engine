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
