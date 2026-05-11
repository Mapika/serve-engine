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
