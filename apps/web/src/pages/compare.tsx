import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, CircleAlert, CircleCheck, Gavel, Plus, X } from 'lucide-react'
import { judge, type Verdict } from '@/lib/judge'
import { HAS_API } from '@/lib/workshop'
import { emptyEpisode, episodeReducer } from '@/state/episode'
import { BracketSpinner, ScanLine } from '@/components/working'
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

const MAX_COLS = 4

// Same question, several models, side by side. The first column is the baseline.
export function Compare() {
  const [repos, setRepos] = useState<RepoSummary[]>([])
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [repoId, setRepoId] = useState<string>('')
  const initial = new URLSearchParams(window.location.search)
  // Column profiles; the first is the baseline. Hooks are fixed at four, columns use the first N.
  const [cols, setCols] = useState<string[]>(() => {
    const fromUrl = (initial.get('models') ?? '').split(',').filter(Boolean)
    if (fromUrl.length >= 2) return fromUrl.slice(0, MAX_COLS)
    return [initial.get('left') ?? '', initial.get('right') ?? '']
  })
  const [openSpan, setOpenSpan] = useState<Span | null>(null)
  const [samples, setSamples] = useState<string[]>([])
  const e0 = useEpisode()
  const e1 = useEpisode()
  const e2 = useEpisode()
  const e3 = useEpisode()
  const eps = [e0, e1, e2, e3].slice(0, cols.length)
  const [ref, setRef] = useState<{ status: 'idle' | 'research' | 'judging' | 'done' | 'error'; episode: Episode; verdicts: Record<number, Verdict>; error?: string; model?: string; refCalls?: number; refSeconds?: number }>({ status: 'idle', episode: emptyEpisode, verdicts: {} })
  const judgeAbort = useRef<AbortController | null>(null)
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (ref.status !== 'research' && ref.status !== 'judging') return
    const t0 = Date.now()
    const t = window.setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 1000)
    return () => window.clearInterval(t)
  }, [ref.status])

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
      setCols((cur) => cur.map((c, i) => c || (i === 0 ? base?.name : trained?.name) || p[0]?.name || ''))
    }).catch((e: Error) => toast.error(e.message))
  }, [])

  const selected = repos.find((r) => r.repo_id === repoId) ?? null
  const running = eps.some((e) => e.episode.status === 'running')
  const idle = eps.every((e) => e.episode.status === 'idle')

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
    if (!repoId || cols.some((c) => !c)) return
    setOpenSpan(null)
    judgeAbort.current?.abort()
    setRef({ status: 'idle', episode: emptyEpisode, verdicts: {} })
    eps.forEach((e, i) => void e.ask(q, repoId, cols[i]))
  }
  const stop = () => eps.forEach((e) => e.stop())
  const allAnswered = !idle && !running && eps.every((e) => e.episode.status === 'done' || e.episode.status === 'error')
  const question = eps[0]?.episode.question ?? ''

  const runJudge = async () => {
    if (!repoId || !question) return
    judgeAbort.current?.abort()
    const ctrl = new AbortController()
    judgeAbort.current = ctrl
    setRef({ status: 'research', episode: episodeReducer(emptyEpisode, { type: 'start', question, repoId, profile: 'opus' }), verdicts: {} })
    const candidates = eps.map((e, i) => ({
      label: displayName(profiles, cols[i]),
      answer: e.episode.answer ?? '',
      citations: e.episode.citations ?? [],
      tool_calls: e.episode.stats?.tool_calls,
    }))
    try {
      for await (const f of judge(repoId, question, candidates, ctrl.signal)) {
        if (ctrl.signal.aborted) return
        setRef((r) => {
          if (f.type === 'phase' && f.phase === 'research') return { ...r, status: 'research', model: f.model }
          if (f.type === 'ref') return { ...r, episode: episodeReducer(r.episode, { type: 'event', event: f.event }) }
          if (f.type === 'phase' && f.phase === 'judging') return { ...r, status: 'judging', refCalls: f.ref_calls, refSeconds: f.ref_seconds, episode: episodeReducer(r.episode, { type: 'event', event: { type: 'done' } }) }
          if (f.type === 'verdict') return { ...r, verdicts: { ...r.verdicts, [f.index]: f } }
          if (f.type === 'error') return { ...r, status: 'error', error: f.message }
          if (f.type === 'done') return { ...r, status: r.status === 'error' ? 'error' : 'done' }
          return r
        })
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setRef((r) => ({ ...r, status: 'error', error: (e as Error).message }))
    }
  }

  const setCol = (i: number, name: string) => setCols((cur) => cur.map((c, j) => (j === i ? name : c)))
  const addCol = () => setCols((cur) => (cur.length < MAX_COLS ? [...cur, profiles.find((p) => !cur.includes(p.name))?.name ?? cur[cur.length - 1]] : cur))
  const removeCol = (i: number) => setCols((cur) => (cur.length > 2 ? cur.filter((_, j) => j !== i) : cur))

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-dvh flex-col bg-canvas">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b bg-background px-3">
          <Button variant="ghost" size="icon" onClick={() => navigate('/app')} aria-label="Back to the workbench">
            <ArrowLeft />
          </Button>
          <Wordmark home />
          <span className="h-4 w-px bg-border" aria-hidden />
          <span className="text-sm">Compare models</span>
          <div className="ml-auto flex items-center gap-2">
            {HAS_API && (
              <Button
                size="sm"
                onClick={() => void runJudge()}
                disabled={!allAnswered || ref.status === 'research' || ref.status === 'judging'}
                title={!allAnswered ? 'Ask a question first; the referee grades finished answers' : undefined}
              >
                <Gavel />
                {ref.status === 'research' || ref.status === 'judging' ? 'Judging' : 'Judge with Opus'}
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={addCol} disabled={running || cols.length >= MAX_COLS}>
              <Plus />
              Add model
            </Button>
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
            <div className="relative">
              <div className="composer-glow" aria-hidden />
              <div className="composer-frame bracket rounded-xl bg-background">
                <span className="bracket-corner tl" aria-hidden />
                <span className="bracket-corner tr" aria-hidden />
                <span className="bracket-corner bl" aria-hidden />
                <span className="bracket-corner br" aria-hidden />
                <div className="overflow-hidden rounded-xl">
                  <div className="flex items-center gap-3 border-b px-3.5 py-2 font-mono text-[11.5px] text-muted-foreground">
                    <span className="flex items-center gap-1.5">
                      <span className="size-1.5 rounded-full bg-verified" aria-hidden />
                      {selected ? repoName(selected) : 'no repository'}
                    </span>
                    <span className="ml-auto truncate">same question to {cols.length} models</span>
                  </div>
                  <QuestionBox onAsk={ask} onStop={stop} running={running} disabled={!selected} showSamples={idle} samples={samples} bare />
                </div>
              </div>
            </div>
          </div>
          {!idle && (
            <div className="mx-auto w-full max-w-[1400px]">
              <CompareSummary episodes={eps.map((e) => e.episode)} labels={cols.map((c) => displayName(profiles, c))} judge={ref.status === 'done' || ref.status === 'judging' ? ref.verdicts : undefined} />
            </div>
          )}
          {ref.status !== 'idle' && (
            <section className="mx-auto w-full max-w-[1400px] rounded-lg border bg-background" aria-label="Referee">
              <header className="flex flex-wrap items-center gap-3 border-b px-4 py-2">
                <Gavel className="size-4 text-verified" aria-hidden />
                <span className="text-sm font-medium">Referee: {ref.model ?? 'Opus'}</span>
                <span className="font-mono text-[11.5px] text-muted-foreground">
                  {ref.status === 'research' && 'researching the question itself through the same harness'}
                  {ref.status === 'judging' && `answered in ${ref.refCalls} calls, ${ref.refSeconds?.toFixed(1)} s; now grading each answer`}
                  {ref.status === 'done' && `answered in ${ref.refCalls} calls, ${ref.refSeconds?.toFixed(1)} s`}
                  {ref.status === 'error' && 'stopped'}
                </span>
              </header>
              {ref.status === 'error' && <p className="px-4 py-3 text-sm text-destructive">{ref.error}</p>}
              <div className={cn('grid gap-4 p-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]')}>
                <div className="min-h-0 rounded-md border">
                  <div className="border-b px-3 py-1.5 font-mono text-[11px] text-muted-foreground">the referee's own research</div>
                  {ref.status === 'research' && (
                    <div className="px-3 pt-2">
                      <ScanLine />
                    </div>
                  )}
                  <Transcript episode={ref.episode} onOpen={setOpenSpan} className="h-[360px]" />
                </div>
                <div className={cn('grid gap-3', cols.length >= 3 ? 'sm:grid-cols-2' : '')}>
                  {cols.map((c, i) => {
                    const v = ref.verdicts[i]
                    return (
                      <article key={i} className="rounded-md border p-3.5">
                        <div className="flex items-baseline gap-2">
                          <span className="truncate text-sm font-medium">{displayName(profiles, c)}</span>
                          {v?.score !== undefined && (
                            <span className={cn('ml-auto font-mono text-[22px] leading-none tabular-nums', v.score >= 6 ? 'text-verified' : v.score >= 3 ? 'text-unverified' : 'text-destructive')}>
                              {v.score}<span className="text-[12px] text-muted-foreground">/10</span>
                            </span>
                          )}
                        </div>
                        {!v && (
                          <p className="mt-2 flex items-center gap-2 font-mono text-[11.5px] text-muted-foreground">
                            <BracketSpinner /> {ref.status === 'research' ? 'waiting for the referee' : ref.status === 'judging' ? `grading, ${elapsed} s` : 'no verdict'}
                          </p>
                        )}
                        {v?.error && <p className="mt-2 text-xs text-destructive">{v.error}</p>}
                        {v && !v.error && (
                          <>
                            <p className={cn('mt-1.5 flex items-center gap-1.5 text-[12px]', v.correct ? 'text-verified' : 'text-destructive')}>
                              {v.correct ? <CircleCheck className="size-3.5" aria-hidden /> : <CircleAlert className="size-3.5" aria-hidden />}
                              {v.correct ? 'correct' : 'not correct'}
                            </p>
                            <p className="mt-2 text-[13px] leading-5">{v.summary}</p>
                            {v.strengths && v.strengths.length > 0 && (
                              <ul className="mt-2 space-y-0.5 text-[12px] text-muted-foreground">
                                {v.strengths.map((x, k) => <li key={k}>+ {x}</li>)}
                              </ul>
                            )}
                            {v.issues && v.issues.length > 0 && (
                              <ul className="mt-1 space-y-0.5 text-[12px] text-muted-foreground">
                                {v.issues.map((x, k) => <li key={k}>− {x}</li>)}
                              </ul>
                            )}
                          </>
                        )}
                      </article>
                    )
                  })}
                </div>
              </div>
            </section>
          )}

          <div className={cn('mx-auto grid w-full max-w-[1400px] flex-1 gap-4 lg:grid-cols-2', cols.length >= 3 && 'xl:grid-cols-3', cols.length >= 4 && '2xl:grid-cols-4')}>
            {eps.map((e, i) => ({ ep: e.episode, value: cols[i] })).map((col, i) => (
              <section key={i} className="flex min-h-[420px] flex-col rounded-lg border bg-background" aria-label={`Model ${i + 1}`}>
                <div className="flex items-center gap-2 border-b px-3 py-2">
                  {i === 0 && <span className="font-mono text-[11px] text-muted-foreground">baseline</span>}
                  <ModelPicker profiles={profiles} value={col.value} onChange={(v) => setCol(i, v)} disabled={running} />
                  {cols.length > 2 && (
                    <Button variant="ghost" size="icon" className="size-7" onClick={() => removeCol(i)} disabled={running} aria-label="Remove this model">
                      <X className="size-3.5" />
                    </Button>
                  )}
                  {col.ep.stats && (
                    <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                      {col.ep.stats.tool_calls} calls, {(col.ep.stats.prompt_tokens / 1000).toFixed(1)}k tokens
                    </span>
                  )}
                </div>
                {col.ep.status === 'running' && (
                  <div className="border-b px-3 py-2">
                    <ScanLine />
                    <p className="mt-1.5 flex items-center gap-2 font-mono text-[11.5px] text-muted-foreground">
                      <BracketSpinner />
                      {(() => { const n = col.ep.rows.filter((r) => r.kind === 'call').length; return n === 0 ? 'reading the repository map' : `researching, ${n} ${n === 1 ? 'call' : 'calls'} so far` })()}
                    </p>
                  </div>
                )}
                {col.ep.status === 'idle' ? (
                  <p className="p-4 text-sm text-muted-foreground">Ask a question above; every model answers it at once.</p>
                ) : (
                  <Transcript episode={col.ep} onOpen={setOpenSpan} className="min-h-0 flex-1" />
                )}
                {col.ep.format && !col.ep.format.ok && (
                  <p className="flex items-start gap-2 border-t px-4 py-2 text-xs text-destructive">
                    <CircleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
                    Format check failed: {col.ep.format.reason}. Scores 0 in training.
                  </p>
                )}
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


function displayName(profiles: Profile[], name: string): string {
  const p = profiles.find((x) => x.name === name)
  return p?.source === 'checkpoints' ? p.name : (p?.label ?? name)
}

// Each column against the baseline (first column). Fewer calls, tokens and seconds are better; more verified citations are better.
function CompareSummary({ episodes, labels, judge: verdicts }: { episodes: Episode[]; labels: string[]; judge?: Record<number, Verdict> }) {
  const rate = (v: { ok: number; all: number }) => (v.all ? (100 * v.ok) / v.all : 0)
  const verified = (e: Episode) => {
    const items = e.citations ?? []
    return { ok: items.filter((c) => c.verified).length, all: items.length }
  }
  const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n))
  // eps: differences below the display precision read as "same".
  const rows: { label: string; help: string; get: (e: Episode) => number | undefined; fmt: (n: number) => string; lowerIsBetter: boolean; text?: (e: Episode) => string; eps?: number }[] = [
    { label: 'Tool calls', help: 'lookups the agent made before answering; the budget depends on the question type', get: (e) => e.stats?.tool_calls, fmt: String, lowerIsBetter: true },
    { label: 'Prompt tokens', help: 'input tokens summed over every turn; the whole conversation is re-sent each turn, so this grows with calls and with how much each tool returned', get: (e) => e.stats?.prompt_tokens, fmt: fmtK, lowerIsBetter: true, eps: 50 },
    ...(verdicts
      ? [{
          label: 'Opus judge',
          help: 'score out of 10 from the referee, after it researched the question itself; higher is better',
          get: (e: Episode) => verdicts[episodes.indexOf(e)]?.score,
          fmt: (n: number) => String(n),
          text: (e: Episode) => { const v = verdicts[episodes.indexOf(e)]; return v?.score !== undefined ? `${v.score}/10` : v?.error ? 'failed' : '' },
          lowerIsBetter: false,
          eps: 0.5,
        }]
      : []),
    {
      label: 'Citations verified',
      help: 'share of [path:Lx-Ly] citations whose lines the agent actually read (or saw in a grep hit); not whether the claim is true',
      get: (e) => (e.answer ? rate(verified(e)) : undefined),
      fmt: (n) => `${Math.round(n)} pts`,
      lowerIsBetter: false,
      eps: 0.5,
      text: (e) => (e.answer ? `${verified(e).ok} of ${verified(e).all} (${Math.round(rate(verified(e)))}%)` : ''),
    },
  ]
  const pending = episodes.some((e) => e.status === 'running')
  const n = episodes.length
  const gridCols = { gridTemplateColumns: `16rem repeat(${n}, minmax(0, 1fr))` }
  return (
    <div className="overflow-x-auto rounded-lg border bg-background">
      <div className="grid min-w-[640px] items-baseline gap-x-4 border-b px-4 py-2 text-xs text-muted-foreground" style={gridCols}>
        <span />
        {labels.map((l, i) => (
          <span key={i} className="truncate">
            {l}
            {i === 0 && <span className="ml-1 font-mono text-[10px]">(baseline)</span>}
          </span>
        ))}
      </div>
      <dl className="divide-y">
        {rows.map((r) => {
          const vals = episodes.map((e) => r.get(e))
          if (vals.every((v) => v === undefined) && !pending) return null
          const a = vals[0]
          return (
            <div key={r.label} className="grid min-w-[640px] items-baseline gap-x-4 px-4 py-2 font-mono text-[13px] tabular-nums" style={gridCols}>
              <dt className="font-sans text-sm">
                {r.label}
                <span className="block text-[11px] leading-4 text-muted-foreground">{r.help}</span>
              </dt>
              {episodes.map((e, i) => {
                const v = vals[i]
                const shown = r.text?.(e) || (v !== undefined ? r.fmt(v) : pending ? '…' : '–')
                const raw = i > 0 && v !== undefined && a !== undefined ? v - a : undefined
                const delta = raw !== undefined && Math.abs(raw) < (r.eps ?? 0) ? 0 : raw
                const better = delta === undefined || delta === 0 ? null : r.lowerIsBetter ? delta < 0 : delta > 0
                const pct = delta !== undefined && a ? Math.round((Math.abs(delta) / a) * 100) : null
                return (
                  <dd key={i}>
                    <span>{shown}</span>
                    {delta !== undefined && (
                      <span className={cn('ml-2 text-[11.5px]', better === true && 'text-verified', better === false && 'text-unverified', better === null && 'text-muted-foreground')}>
                        {delta === 0 ? 'same' : `${delta > 0 ? '+' : '−'}${r.fmt(Math.abs(delta))}${pct !== null && r.lowerIsBetter ? ` (${pct}% ${delta < 0 ? 'fewer' : 'more'})` : ''}`}
                      </span>
                    )}
                  </dd>
                )
              })}
            </div>
          )
        })}
      </dl>
    </div>
  )
}
