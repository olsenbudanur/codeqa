import type { ReactNode } from 'react'
import { ArrowLeft, Activity, Boxes, Database, GitBranch, ScrollText } from 'lucide-react'
import { Toaster } from 'sonner'
import { navigate } from '@/lib/router'
import { HAS_API } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ThemeToggle } from '@/components/theme-toggle'
import { Wordmark } from '@/components/wordmark'
import { RunsPage } from './runs'
import { LivePage } from './live'
import { CheckpointsPage } from './checkpoints'
import { DataPage } from './data'
import { TracesPage } from './traces'

const NAV = [
  { href: '/workshop/live', label: 'Live', icon: Activity },
  { href: '/workshop/runs', label: 'Runs', icon: GitBranch },
  { href: '/workshop/checkpoints', label: 'Checkpoints', icon: Boxes },
  { href: '/workshop/data', label: 'Data', icon: Database },
  { href: '/workshop/traces', label: 'Traces', icon: ScrollText },
]

// Read-only pages over data/: what the model trained on, how the runs went, every episode on disk.
export function Workshop({ path, search }: { path: string; search: string }) {
  const rest = path.replace(/^\/workshop\/?/, '')
  const [section, ...parts] = rest.split('/').filter(Boolean)
  const params = new URLSearchParams(search)

  let body: ReactNode
  if (!HAS_API) {
    body = (
      <div className="mx-auto max-w-[60ch] px-6 py-16 text-sm">
        <p className="text-base font-medium">The workshop reads training runs, checkpoints and traces from the API.</p>
        <p className="mt-2 text-muted-foreground">
          Start it with <code className="rounded bg-muted px-1 font-mono text-[13px]">uv run uvicorn apps.api.server:app --port 8000</code> and run the web app with{' '}
          <code className="rounded bg-muted px-1 font-mono text-[13px]">VITE_API_URL=http://localhost:8000</code>.
        </p>
      </div>
    )
  } else if (section === 'live') body = <LivePage requested={params.get('run')} />
  else if (section === 'checkpoints') body = <CheckpointsPage />
  else if (section === 'data') body = <DataPage params={params} />
  else if (section === 'traces') body = <TracesPage parts={parts} params={params} />
  else if (section === 'runs') body = <RunsPage name={parts[0]} params={params} />
  else body = <LivePage requested={params.get('run')} />

  const active = section && NAV.some((n) => n.href.endsWith(section)) ? section : 'live'

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-dvh flex-col bg-canvas">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b bg-background px-3">
          <Button variant="ghost" size="icon" onClick={() => navigate('/app')} aria-label="Back to the workbench">
            <ArrowLeft />
          </Button>
          <Wordmark />
          <span className="h-4 w-px bg-border" aria-hidden />
          <span className="text-sm">Workshop</span>
          <nav className="ml-6 hidden items-center gap-1 md:flex" aria-label="Workshop sections">
            {NAV.map((n) => (
              <a
                key={n.href}
                href={n.href}
                onClick={(e) => {
                  e.preventDefault()
                  navigate(n.href)
                }}
                className={cn(
                  'flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition-colors hover:bg-accent',
                  n.href.endsWith(active) ? 'bg-accent text-foreground' : 'text-muted-foreground',
                )}
              >
                <n.icon className="size-3.5" aria-hidden />
                {n.label}
              </a>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-1">
            <ThemeToggle />
          </div>
        </header>
        <nav className="flex gap-1 overflow-x-auto border-b bg-background px-2 py-1.5 md:hidden" aria-label="Workshop sections">
          {NAV.map((n) => (
            <button key={n.href} type="button" onClick={() => navigate(n.href)} className={cn('rounded-md px-2.5 py-1 text-sm', n.href.endsWith(active) ? 'bg-accent' : 'text-muted-foreground')}>
              {n.label}
            </button>
          ))}
        </nav>
        <main className="min-h-0 flex-1 overflow-y-auto">{body}</main>
        <Toaster position="bottom-right" />
      </div>
    </TooltipProvider>
  )
}
