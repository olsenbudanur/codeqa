import { useEffect, useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { toast, Toaster } from 'sonner'
import { api } from '@/lib/api'
import { navigate } from '@/lib/router'
import { repoName } from '@/lib/repo'
import type { Profile, RepoSummary, Span } from '@/lib/contracts'
import type { Episode } from '@/state/episode'
import { cn } from '@/lib/utils'
import { useEpisode } from '@/hooks/use-episode'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ThemeToggle } from '@/components/theme-toggle'
import { ModelPicker } from '@/components/model-picker'
import { Wordmark } from '@/components/wordmark'
import { QuestionBox } from '@/components/ask/question-box'
import { Transcript } from '@/components/chat/transcript'
import { FileViewer } from '@/components/file/file-viewer'

// Same question, two models, side by side. Built for the demo: base vs trained.
export function Compare() {
  const [repos, setRepos] = useState<RepoSummary[]>([])
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [repoId, setRepoId] = useState<string>('')
  const [left, setLeft] = useState('')
  const [right, setRight] = useState('')
  const [openSpan, setOpenSpan] = useState<Span | null>(null)
  const [samples, setSamples] = useState<string[]>([])
  const a = useEpisode()
  const b = useEpisode()

  useEffect(() => {
    api.listRepos().then((list) => {
      setRepos(list)
      const flask = list.find((r) => r.repo_id.includes('flask') && r.stage === 'ready') ?? list.find((r) => r.stage === 'ready')
      if (flask) setRepoId((cur) => cur || flask.repo_id)
    }).catch((e: Error) => toast.error(e.message))
    api.listProfiles().then((p) => {
      setProfiles(p)
      const base = p.find((x) => x.kind === 'tinker' && /untrained/i.test(x.label ?? '')) ?? p[0]
      const trained = [...p].reverse().find((x) => x.kind === 'tinker' && /trained$/i.test(x.label ?? '')) ?? p[p.length - 1]
      setLeft((cur) => cur || base?.name || '')
      setRight((cur) => cur || trained?.name || '')
    }).catch((e: Error) => toast.error(e.message))
  }, [])

  const selected = repos.find((r) => r.repo_id === repoId) ?? null
  const running = a.episode.status === 'running' || b.episode.status === 'running'

  useEffect(() => {
    if (!repoId) return
    let cancelled = false
    api.suggestions(repoId).then((q) => {
      if (!cancelled) setSamples(q)
    }).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [repoId])

  const ask = (q: string) => {
    if (!repoId || !left || !right) return
    setOpenSpan(null)
    void a.ask(q, repoId, left)
    void b.ask(q, repoId, right)
  }
  const stop = () => {
    a.stop()
    b.stop()
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-dvh flex-col bg-canvas">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b bg-background px-3">
          <Button variant="ghost" size="icon" onClick={() => navigate('/app')} aria-label="Back to the workbench">
            <ArrowLeft />
          </Button>
          <Wordmark />
          <span className="h-4 w-px bg-border" aria-hidden />
          <span className="text-sm">Compare</span>
          <div className="ml-auto flex items-center gap-2">
            <Select value={repoId} onValueChange={setRepoId} disabled={running}>
              <SelectTrigger className="h-8 w-[220px]" aria-label="Repository">
                <SelectValue placeholder="Repository">{selected ? repoName(selected) : 'Repository'}</SelectValue>
              </SelectTrigger>
              <SelectContent align="end">
                {repos.filter((r) => r.stage === 'ready').map((r) => (
                  <SelectItem key={r.repo_id} value={r.repo_id}>
                    {repoName(r)} <span className="font-mono text-xs text-muted-foreground">{r.sha.slice(0, 7)}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <ThemeToggle />
          </div>
        </header>

        <main className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4 sm:p-6">
          <div className="mx-auto w-full max-w-[1400px]">
            <QuestionBox onAsk={ask} onStop={stop} running={running} disabled={!selected} showSamples={a.episode.status === 'idle'} samples={samples} />
          </div>
          {(a.episode.status !== 'idle' || b.episode.status !== 'idle') && (
            <div className="mx-auto w-full max-w-[1400px]">
              <CompareSummary
                left={a.episode}
                right={b.episode}
                leftLabel={profiles.find((p) => p.name === left)?.label ?? left}
                rightLabel={profiles.find((p) => p.name === right)?.label ?? right}
              />
            </div>
          )}
          <div className="mx-auto grid w-full max-w-[1400px] flex-1 gap-4 lg:grid-cols-2">
            {[
              { ep: a.episode, value: left, set: setLeft },
              { ep: b.episode, value: right, set: setRight },
            ].map((col, i) => (
              <section key={i} className="flex min-h-[420px] flex-col rounded-lg border bg-background" aria-label={`Model ${i + 1}`}>
                <div className="flex items-center gap-2 border-b px-3 py-2">
                  <ModelPicker profiles={profiles} value={col.value} onChange={col.set} disabled={running} />
                  {col.ep.stats && (
                    <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                      {col.ep.stats.tool_calls} calls, {(col.ep.stats.prompt_tokens / 1000).toFixed(1)}k tokens, {col.ep.stats.seconds.toFixed(1)} s
                    </span>
                  )}
                </div>
                {col.ep.status === 'idle' ? (
                  <p className="p-4 text-sm text-muted-foreground">Ask a question above; both models answer it at once.</p>
                ) : (
                  <Transcript episode={col.ep} onOpen={setOpenSpan} className="min-h-0 flex-1" />
                )}
                {col.ep.status === 'error' && <p className="border-t px-4 py-2 text-xs text-destructive">{col.ep.error}</p>}
              </section>
            ))}
          </div>
        </main>

        <Sheet open={!!openSpan} onOpenChange={(o) => !o && setOpenSpan(null)}>
          <SheetContent side="right" className="w-full p-0 sm:max-w-[640px]" showCloseButton={false}>
            <SheetTitle className="sr-only">File</SheetTitle>
            {selected && openSpan && (
              <FileViewer repoId={selected.repo_id} repoUrl={selected.url} sha={selected.sha} span={openSpan} onClose={() => setOpenSpan(null)} />
            )}
          </SheetContent>
        </Sheet>
        <Toaster position="bottom-right" />
      </div>
    </TooltipProvider>
  )
}


// Right minus left, per metric. Fewer calls, tokens and seconds are better; more verified citations are better.
function CompareSummary({ left, right, leftLabel, rightLabel }: { left: Episode; right: Episode; leftLabel: string; rightLabel: string }) {
  const rate = (v: { ok: number; all: number }) => (v.all ? (100 * v.ok) / v.all : 0)
  const verified = (e: Episode) => {
    const items = e.citations ?? []
    return { ok: items.filter((c) => c.verified).length, all: items.length }
  }
  const rows: { label: string; a?: number; b?: number; fmt: (n: number) => string; lowerIsBetter: boolean; text?: [string, string] }[] = [
    { label: 'Tool calls', a: left.stats?.tool_calls, b: right.stats?.tool_calls, fmt: (n) => String(n), lowerIsBetter: true },
    { label: 'Prompt tokens', a: left.stats?.prompt_tokens, b: right.stats?.prompt_tokens, fmt: (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)), lowerIsBetter: true },
    { label: 'Seconds', a: left.stats?.seconds, b: right.stats?.seconds, fmt: (n) => n.toFixed(1), lowerIsBetter: true },
    {
      // Compared as a rate: an answer with no citations counts as 0%.
      label: 'Citations verified',
      a: left.answer ? rate(verified(left)) : undefined,
      b: right.answer ? rate(verified(right)) : undefined,
      fmt: (n) => `${Math.round(n)} pts`,
      lowerIsBetter: false,
      text: [left.answer ? `${verified(left).ok} of ${verified(left).all} (${Math.round(rate(verified(left)))}%)` : '', right.answer ? `${verified(right).ok} of ${verified(right).all} (${Math.round(rate(verified(right)))}%)` : ''],
    },
  ]
  const pending = left.status === 'running' || right.status === 'running'
  return (
    <div className="rounded-lg border bg-background">
      <div className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] items-baseline gap-x-4 border-b px-4 py-2 text-xs text-muted-foreground sm:grid-cols-[10rem_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <span />
        <span className="truncate text-right sm:text-left">{leftLabel}</span>
        <span className="truncate text-right sm:text-left">{rightLabel}</span>
        <span className="text-right sm:text-left">Difference</span>
      </div>
      <dl className="divide-y">
        {rows.map((r) => {
          const has = r.a !== undefined && r.b !== undefined
          const delta = has ? r.b! - r.a! : undefined
          const better = delta === undefined || delta === 0 ? null : r.lowerIsBetter ? delta < 0 : delta > 0
          const pct = has && r.a ? Math.round((Math.abs(delta!) / r.a!) * 100) : null
          return (
            <div key={r.label} className="grid grid-cols-[minmax(0,1fr)_auto_auto_auto] items-baseline gap-x-4 px-4 py-2 font-mono text-[13px] tabular-nums sm:grid-cols-[10rem_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]">
              <dt className="font-sans text-sm">{r.label}</dt>
              <dd className="text-right sm:text-left">{r.text?.[0] || (r.a !== undefined ? r.fmt(r.a) : pending ? '…' : '–')}</dd>
              <dd className="text-right sm:text-left">{r.text?.[1] || (r.b !== undefined ? r.fmt(r.b) : pending ? '…' : '–')}</dd>
              <dd className={cn('text-right sm:text-left', better === true && 'text-verified', better === false && 'text-unverified')}>
                {delta === undefined ? (pending ? '…' : '–') : delta === 0 ? 'same' : `${delta > 0 ? '+' : '−'}${r.fmt(Math.abs(delta))}${pct !== null && r.lowerIsBetter ? ` (${pct}% ${delta < 0 ? 'fewer' : 'more'})` : ''}`}
              </dd>
            </div>
          )
        })}
      </dl>
    </div>
  )
}
