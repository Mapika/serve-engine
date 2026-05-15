import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

export default function Services() {
  const profiles = useQuery({ queryKey: ['profiles'], queryFn: api.listProfiles })
  const routes = useQuery({ queryKey: ['routes'], queryFn: api.listRoutes })

  return (
    <div className="space-y-14">
      <header className="flex items-baseline justify-between">
        <h2 className="text-2xl font-light tracking-tightish caret">services</h2>
        <div className="label">
          {(profiles.data ?? []).length} profiles / {(routes.data ?? []).length} routes
        </div>
      </header>

      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="label">routes</div>
          <div className="text-mute text-[11px] tracking-wider">
            lower priority wins
          </div>
        </div>
        <table className="ditable">
          <thead>
            <tr>
              <th className="w-12">pri</th>
              <th>name</th>
              <th>match model</th>
              <th>profile</th>
              <th>fallback</th>
              <th>enabled</th>
              <th className="text-right">actions</th>
            </tr>
          </thead>
          <tbody>
            {(routes.data ?? []).length === 0 && (
              <tr>
                <td colSpan={7} className="!py-12 text-center text-mute">
                  no routes. create one below to expose a public model name.
                </td>
              </tr>
            )}
            {(routes.data ?? [])
              .slice()
              .sort((a, b) => a.priority - b.priority)
              .map(r => (
                <tr key={r.id}>
                  <td className="text-mute tnum">{r.priority}</td>
                  <td>{r.name}</td>
                  <td className="font-mono text-[12px]">{r.match_model}</td>
                  <td className="text-dim">{r.profile_name}</td>
                  <td className="text-mute">{r.fallback_profile_name ?? '—'}</td>
                  <td>
                    <span className={`dot ${r.enabled ? 'dot-ready' : 'dot-stopped'}`} />
                    <span className="text-dim">{r.enabled ? 'on' : 'off'}</span>
                  </td>
                  <td className="text-right">
                    <span className="text-mute">—</span>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </section>

      <section className="space-y-4">
        <div className="label">profiles</div>
        <div className="text-mute text-[12px]">coming next step</div>
      </section>
    </div>
  )
}
