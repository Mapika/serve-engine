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
        <div className="label">routes</div>
        <div className="text-mute text-[12px]">coming next step</div>
      </section>

      <section className="space-y-4">
        <div className="label">profiles</div>
        <div className="text-mute text-[12px]">coming next step</div>
      </section>
    </div>
  )
}
