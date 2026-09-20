import { useState } from 'react'
import { Check, ChevronRight } from 'lucide-react'
import { BracketSpinner, Dots } from '@/components/working'
import { ThinkingBlock, ThinkingToggleAll } from './thinking-block'
import { groupRounds, type LogRow, type ToolRow } from '@/state/episode'
import type { Span } from '@/lib/contracts'
import { cn } from '@/lib/utils'

// One sentence per tool call, written as what the agent is doing.
function describeCall(r: ToolRow): { verb: string; span?: Span; code?: string } {
  const a = r.args
  switch (r.name) {
    case 'read_file': {
      const path = String(a.path ?? '')
      const start = Number(a.start ?? 1)
      const end = Number(a.end ?? start)
      if (!Number.isFinite(start) || !Number.isFinite(end)) return { verb: `Read ${path} (malformed range: ${JSON.stringify(a.start)}…)` }
      return { verb: `Read ${path}, lines ${start}–${end}`, span: { path, start, end } }
    }
    case 'find_symbol': {
      const kind = a.kind ? String(a.kind) : 'symbol'
      const where = a.file_pattern ? ` in ${a.file_pattern}` : ''
      return { verb: `Look up ${kind} ${String(a.name ?? '')}${where}` }
    }
    case 'grep': {
      const where = a.file_pattern ? ` in ${a.file_pattern}` : ''
      return { verb: `Search for ${String(a.pattern ?? '')}${where}` }
    }
    case 'list_dir':
      return { verb: `List ${String(a.path ?? '/') || '/'}` }
    case 'overview':
      return { verb: 'Get the repository overview' }
    case 'bash':
      return { verb: `Run ${String(a.command ?? '')}`, code: String(a.command ?? '') }
    default:
      return { verb: String(r.name) }
  }
}


const fmt = new Intl.NumberFormat('en-US')

// Right-hand column: what came back, without repeating the verb.
function resultLabel(r: ToolRow): string {
  const res = r.result!
  if (r.name === 'read_file') {
    const start = Number(r.args.start ?? 1)
    const end = Number(r.args.end ?? start)
    const n = end - start + 1
    if (!Number.isFinite(n)) return res.summary
    return `${n} ${n === 1 ? 'line' : 'lines'}`
  }
  return res.summary
}

const secs = (s: number | undefined) => (s === undefined ? '' : `, ${s < 0.05 ? '<0.1' : s.toFixed(1)} s`)

export function Ledger({
  rows,
  running,
  budget,
  onOpen,
  compact,
  startStep = 1,
  callCount,
}: {
  rows: LogRow[]
  running: boolean
  budget?: number
  onOpen: (span: Span) => void
  compact?: boolean
  startStep?: number // step number of the first call in `rows`
  callCount?: number // total calls so far, when `rows` is a slice
}) {
  const calls = rows.filter((r): r is ToolRow => r.kind === 'call')
  const total = callCount ?? calls.length
  const thoughts = rows.filter((r) => r.kind === 'thinking').length
  const [allOpen, setAllOpen] = useState(false)
  const [openIds, setOpenIds] = useState<Set<number>>(new Set())
  const toggleOne = (id: number) => setOpenIds((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })
  return (
    <section aria-label="Research log">
      {!compact && (
      <header className="mb-2 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-medium">Research</h2>
        <span className="ml-auto"><ThinkingToggleAll count={thoughts} allOpen={allOpen} onToggle={() => setAllOpen((v) => !v)} /></span>
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {total} {total === 1 ? 'call' : 'calls'}
          {budget ? ` of ${budget}` : ''}
        </span>
      </header>
      )}
      <ol className={compact ? '' : 'border-t'}>
        {groupRounds(rows).map((r) => {
          if (r.kind === 'round')
            return (
              <RoundView key={`round-${r.turn}`} turn={r.turn} rows={r.rows} firstStep={startStep + calls.indexOf(r.rows[0])} running={running} compact={compact} onOpen={onOpen} />
            )
          if (r.kind === 'thinking')
            return (
              <li key={r.id} className="row-in border-b py-2">
                <ThinkingBlock text={r.text} open={allOpen || openIds.has(r.id)} onToggle={() => (allOpen ? setAllOpen(false) : toggleOne(r.id))} compact={compact} />
              </li>
            )
          if (r.kind === 'notice')
            return (
              <li key={r.id} className="row-in border-b py-2.5 text-[13px]">
                <span className="mr-2 rounded bg-unverified-soft px-1.5 py-0.5 font-mono text-[11px] text-unverified">{r.notice === 'forced_answer' ? 'budget exhausted' : r.notice}</span>
                <span className="text-muted-foreground">{r.notice === 'forced_answer' ? 'The harness asked for the final answer with no more tool calls.' : r.text}</span>
              </li>
            )
          const step = startStep + calls.indexOf(r)
          const d = describeCall(r)
          const pending = !r.result
          return <ToolRowView key={r.id} r={r} step={step} pending={pending} running={running} compact={compact} d={d} onOpen={onOpen} />
        })}
        {running && rows.length === 0 && (
          <li className="flex items-center gap-2 py-2.5 text-sm text-muted-foreground"><BracketSpinner />Reading the question</li>
        )}
      </ol>
    </section>
  )
}

// One message that carried several commands: a single row until expanded (the round is the unit the model acted in).
function RoundView({ turn, rows, firstStep, running, compact, onOpen }: { turn: number; rows: ToolRow[]; firstStep: number; running: boolean; compact?: boolean; onOpen: (span: Span) => void }) {
  const [open, setOpen] = useState(false)
  const done = rows.filter((r) => r.result).length
  const pending = done < rows.length
  const errors = rows.filter((r) => r.result?.summary.startsWith('ERROR')).length
  const preview = rows.map((r) => describeCall(r).code ?? describeCall(r).verb).join('  ·  ')
  return (
    <li className="row-in border-b py-2.5">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} className="grid w-full grid-cols-[1.5rem_minmax(0,1fr)_auto] gap-x-2 text-left">
        <span className="pt-px font-mono text-xs text-muted-foreground tabular-nums">{firstStep}–{firstStep + rows.length - 1}</span>
        <span className="min-w-0">
          <span className="flex items-center gap-1.5 text-sm">
            <ChevronRight className={cn('size-3.5 shrink-0 text-muted-foreground transition-transform', open && 'rotate-90')} aria-hidden />
            Ran {rows.length} commands in one message
            <span className="rounded border px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground">round {turn}</span>
          </span>
          {!open && <span className="mt-0.5 block truncate pl-5 font-mono text-[11.5px] text-muted-foreground" title={preview}>{preview}</span>}
        </span>
        <span className="pt-px text-right font-mono text-xs text-muted-foreground tabular-nums">
          {pending ? (running ? <span className="inline-flex items-center gap-1.5">{done} of {rows.length}<Dots className="text-verified" /></span> : `${done} of ${rows.length}`)
            : compact ? <Check className="size-3.5 text-verified" strokeWidth={3} aria-label="Done" /> : errors ? `${errors} error${errors > 1 ? 's' : ''}` : 'done'}
        </span>
      </button>
      {open && (
        <ol className="mt-2 border-l pl-3">
          {rows.map((r, i) => (
            <ToolRowView key={r.id} r={r} step={firstStep + i} pending={!r.result} running={running} compact={compact} d={describeCall(r)} onOpen={onOpen} />
          ))}
        </ol>
      )}
    </li>
  )
}

function ToolRowView({ r, step, pending, running, compact, d, onOpen }: { r: ToolRow; step: number; pending: boolean; running: boolean; compact?: boolean; d: { verb: string; span?: Span; code?: string }; onOpen: (span: Span) => void }) {
  const [open, setOpen] = useState(false)
  const expandable = !!r.result?.text
  return (
    <li className="row-in border-b py-2.5 last:border-0">
            <div className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] gap-x-2">
              <span className="pt-px font-mono text-xs text-muted-foreground tabular-nums">{step}</span>
              <div className="min-w-0">
                {d.code !== undefined ? (
                  <span className="text-sm">Run <code className="rounded bg-muted px-1 py-0.5 font-mono text-[12.5px] break-all">{d.code}</code></span>
                ) : d.span ? (
                  <button
                    type="button"
                    onClick={() => onOpen(d.span!)}
                    className="text-left text-sm underline-offset-2 hover:underline focus-visible:underline focus-visible:outline-none"
                  >
                    {d.verb}
                  </button>
                ) : (
                  <span className="text-sm">{d.verb}</span>
                )}
                {r.why && <p className="mt-0.5 text-xs text-muted-foreground">{r.why}</p>}
              </div>
              <span className="max-w-[18rem] truncate pt-px text-right font-mono text-xs text-muted-foreground tabular-nums" title={r.result?.summary}>
                {pending ? (
                  running ? <Dots className="text-verified" /> : <span className="inline-block size-2 rounded-full bg-foreground/60" aria-label="Pending" />
                ) : (
                  <span key="done" className="row-in inline-block">
                    {compact ? (
                      <Check className="size-3.5 text-verified" strokeWidth={3} aria-label="Done" />
                    ) : (
                      <>
                        <span className="max-sm:hidden">{resultLabel(r)}{secs(r.result!.seconds)}</span>
                        <span className="sm:hidden">{fmt.format(r.result!.chars)} chars</span>
                      </>
                    )}
                  </span>
                )}
              </span>
            </div>
            {expandable && (
              <div className="pl-[1.5rem]">
                <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} className="mt-1 font-mono text-[11px] text-muted-foreground hover:text-foreground">
                  {open ? 'hide result' : 'show result'}
                </button>
                {open && <pre className="mt-1 max-h-[280px] overflow-auto rounded bg-muted p-2 font-mono text-[11.5px] leading-4 whitespace-pre-wrap">{r.result!.text}</pre>}
              </div>
            )}
    </li>
  )
}
