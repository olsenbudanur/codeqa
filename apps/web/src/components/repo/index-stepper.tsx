import { Check, CircleAlert } from 'lucide-react'
import type { RepoJobStatus, RepoStage } from '@/lib/contracts'
import { cn } from '@/lib/utils'

const STEPS: { stage: RepoStage; label: string; doing: string }[] = [
  { stage: 'snapshot', label: 'Snapshot', doing: 'Downloading the repository' },
  { stage: 'index', label: 'Index', doing: 'Parsing symbols' },
  { stage: 'summaries', label: 'Summaries', doing: 'Summarising directories' },
]

const ORDER: RepoStage[] = ['snapshot', 'index', 'summaries', 'ready']

export function IndexStepper({ status, repoName }: { status: RepoJobStatus; repoName: string }) {
  const at = ORDER.indexOf(status.stage)
  return (
    <div className="px-3 py-3" aria-live="polite">
      <p className="mb-2 truncate text-sm">
        {status.stage === 'error'
          ? `Could not index ${repoName}`
          : status.stage === 'ready'
            ? `${repoName} is ready`
            : `Indexing ${repoName}`}
      </p>
      <ol className="space-y-1.5">
        {STEPS.map((s, i) => {
          const done = status.stage !== 'error' && at > i
          const active = status.stage === s.stage
          const failed = status.stage === 'error' && at === -1 && i === 0
          return (
            <li key={s.stage} className="flex items-center gap-2 text-sm">
              <span
                className={cn(
                  'flex size-4 shrink-0 items-center justify-center rounded-full border',
                  done && 'border-verified bg-verified text-white',
                  active && 'border-foreground',
                  !done && !active && 'border-border',
                )}
                aria-hidden
              >
                {done && <Check className="size-3" strokeWidth={3} />}
                {active && <span className="size-1.5 animate-pulse rounded-full bg-foreground" />}
                {failed && <CircleAlert className="size-3 text-destructive" />}
              </span>
              <span className={cn(!done && !active && 'text-muted-foreground')}>
                {active ? s.doing : s.label}
              </span>
              {active && (
                <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                  {Math.round(status.progress * 100)}%
                </span>
              )}
            </li>
          )
        })}
      </ol>
      {status.stage === 'error' && status.message && (
        <p className="mt-2 text-xs text-destructive">{status.message}</p>
      )}
      <p className="mt-2 font-mono text-xs text-muted-foreground tabular-nums">{status.seconds.toFixed(0)} s</p>
    </div>
  )
}
