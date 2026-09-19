import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronRight, CircleAlert } from 'lucide-react'
import { Dots } from '@/components/working'
import { ThinkingBlock, ThinkingToggleAll } from '@/components/research/thinking-block'
import type { Span } from '@/lib/contracts'
import { cn } from '@/lib/utils'
import type { Episode, LogRow, ToolRow } from '@/state/episode'
import { AnswerPanel } from '@/components/answer/answer-panel'
import { Wordmark } from '@/components/wordmark'

// Chat-shaped view of an episode: the question as the user's turn, then one
// assistant turn whose activity rows (thoughts and tool calls) stream in and
// collapse to a summary once the answer lands. The viewport is fixed-height
// and stays pinned to the newest content, so nothing outside it moves.
export function Transcript({ episode, onOpen, className }: { episode: Episode; onOpen: (s: Span) => void; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const reduced = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches, [])
  const [showSteps, setShowSteps] = useState(false)
  const answered = !!episode.answer
  const running = episode.status === 'running'
  const calls = episode.rows.filter((r): r is ToolRow => r.kind === 'call')

  useEffect(() => {
    if (answered) setShowSteps(false)
  }, [answered])

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const id = window.requestAnimationFrame(() => el.scrollTo({ top: el.scrollHeight, behavior: reduced ? 'auto' : 'smooth' }))
    return () => window.cancelAnimationFrame(id)
  }, [episode.rows.length, episode.answer, episode.stats, showSteps, reduced])

  return (
    <div ref={ref} className={cn('stage-scroll flex flex-col gap-4 overflow-y-auto px-4 py-3.5 sm:px-5', className)}>
      {episode.question && (
        <div className="flex justify-end">
          <p className="max-w-[85%] rounded-2xl rounded-br-md bg-accent px-3.5 py-2 text-[14.5px] leading-6">{episode.question}</p>
        </div>
      )}

      {(episode.rows.length > 0 || running || answered) && (
        <div className="flex gap-3">
          <span className="mt-1 shrink-0 font-mono text-[13px] text-verified" aria-hidden>[L]</span>
          <span className="sr-only"><Wordmark /></span>
          <div className="min-w-0 flex-1">
            {answered ? (
              <>
                <button
                  type="button"
                  onClick={() => setShowSteps((v) => !v)}
                  aria-expanded={showSteps}
                  className="row-in mb-2 inline-flex items-center gap-1.5 rounded-md py-0.5 text-[13px] text-muted-foreground hover:text-foreground"
                >
                  <Check className="size-3.5 text-verified" strokeWidth={3} aria-hidden />
                  Researched in {calls.length} {calls.length === 1 ? 'call' : 'calls'}
                  {typeof episode.stats?.seconds === 'number' ? `, ${episode.stats.seconds.toFixed(1)} s` : ''}
                  <ChevronRight className={cn('size-3.5 transition-transform', showSteps && 'rotate-90')} aria-hidden />
                </button>
                {showSteps && <Activity rows={episode.rows} running={false} onOpen={onOpen} className="mb-4" />}
                <div className="row-in">
                  <AnswerPanel episode={episode} onOpen={onOpen} compact bare />
                </div>
              </>
            ) : (
              <>
                {running && episode.budget && (
                  <p className="mb-1.5 font-mono text-[11.5px] text-muted-foreground">context {fmtK(episode.budget.context_tokens)} of {fmtK(episode.budget.context_cap)}, {episode.budget.messages_left} {episode.budget.messages_left === 1 ? 'message' : 'messages'} left</p>
                )}
                <Activity rows={episode.rows} running={running} onOpen={onOpen} />
                {episode.status === 'error' && (
                  <p role="alert" className="row-in mt-3 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-[13px]">
                    <CircleAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
                    <span>
                      <span className="font-medium">No answer.</span> {episode.error}
                      <span className="block text-xs text-muted-foreground">In training this scores 0 (and −0.1 with the no-answer penalty).</span>
                    </span>
                  </p>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

const VERBS: Record<ToolRow['name'], [string, string]> = {
  read_file: ['Reading', 'Read'],
  find_symbol: ['Looking up', 'Looked up'],
  grep: ['Searching for', 'Searched for'],
  list_dir: ['Listing', 'Listed'],
  overview: ['Getting the repository overview', 'Got the repository overview'],
  bash: ['Running', 'Ran'],
}

function describe(r: ToolRow): { text: string; span?: Span; code?: string } {
  const a = r.args
  const v = VERBS[r.name][r.result ? 1 : 0]
  switch (r.name) {
    case 'bash':
      return { text: v, code: String(a.command ?? '') }
    case 'read_file': {
      const path = String(a.path ?? '')
      const start = Number(a.start ?? 1)
      const end = Number(a.end ?? start)
      return { text: `${v} ${path}, lines ${start}–${end}`, span: { path, start, end } }
    }
    case 'find_symbol':
      return { text: `${v} ${a.kind ? String(a.kind) : 'symbol'} ${String(a.name ?? '')}` }
    case 'grep':
      return { text: `${v} ${String(a.pattern ?? '')}${a.file_pattern ? ` in ${a.file_pattern}` : ''}` }
    case 'list_dir':
      return { text: `${v} ${String(a.path ?? '/') || '/'}` }
    case 'overview':
      return { text: v }
  }
}

const fmtK = (n: number) => `${Math.round(n / 1000)}k`

function resultNote(r: ToolRow): string {
  if (r.name === 'read_file') {
    const n = Number(r.args.end ?? r.args.start ?? 1) - Number(r.args.start ?? 1) + 1
    return `${n} ${n === 1 ? 'line' : 'lines'}`
  }
  return r.result!.summary.replace(/^(\d+ hits?):\s*/, '$1, ')
}

function Activity({ rows, running, onOpen, className }: { rows: LogRow[]; running: boolean; onOpen: (s: Span) => void; className?: string }) {
  const [allOpen, setAllOpen] = useState(false)
  const [openIds, setOpenIds] = useState<Set<number>>(new Set())
  const thoughts = rows.filter((r) => r.kind === 'thinking').length
  const toggleOne = (id: number) => setOpenIds((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })
  return (
    <ol className={cn('space-y-2', className)} aria-label="Activity">
      {thoughts > 1 && (
        <li className="pl-[22px]">
          <ThinkingToggleAll count={thoughts} allOpen={allOpen} onToggle={() => setAllOpen((v) => !v)} />
        </li>
      )}
      {rows.length === 0 && running && (
        <li className="row-in flex items-center gap-2 text-[13px] text-muted-foreground">
          <Dot pending />
          Reading the repository map
        </li>
      )}
      {rows.map((r) => {
        if (r.kind === 'thinking') {
          return (
            <li key={r.id} className="row-in pl-[22px]">
              <ThinkingBlock text={r.text} open={allOpen || openIds.has(r.id)} onToggle={() => (allOpen ? setAllOpen(false) : toggleOne(r.id))} compact />
            </li>
          )
        }
        if (r.kind === 'notice') {
          return (
            <li key={r.id} className="row-in flex items-start gap-2 text-[13px] leading-5">
              <span className="mt-[5px] size-3.5 shrink-0" aria-hidden />
              <span className="text-unverified">Budget exhausted: the harness asked for the final answer.</span>
            </li>
          )
        }
        const d = describe(r)
        const pending = !r.result
        return (
          <li key={r.id} className="row-in flex items-start gap-2 text-[13px] leading-5">
            <Dot pending={pending} running={running} />
            <span className="min-w-0">
              {d.code !== undefined ? (
                <span>{d.text} <code className="rounded bg-muted px-1 py-0.5 font-mono text-[12px] break-all">{d.code}</code></span>
              ) : d.span ? (
                <button type="button" onClick={() => onOpen(d.span!)} className="text-left underline-offset-2 hover:underline">
                  {d.text}
                </button>
              ) : (
                <span>{d.text}</span>
              )}
              {r.result && <span className="ml-2 font-mono text-[11.5px] text-muted-foreground">{resultNote(r)}</span>}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

function Dot({ pending, running = true }: { pending: boolean; running?: boolean }) {
  return (
    <span className="mt-[5px] flex size-3.5 shrink-0 items-center justify-center" aria-hidden>
      {pending ? (
        running ? <Dots className="text-verified" /> : <span className="size-1.5 rounded-full bg-foreground/60" />
      ) : (
        <Check className="dot-done size-3.5 text-verified" strokeWidth={3} />
      )}
    </span>
  )
}
