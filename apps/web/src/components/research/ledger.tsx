import { useState } from 'react'
import { Check } from 'lucide-react'
import { BracketSpinner, Dots } from '@/components/working'
import { ThinkingBlock, ThinkingToggleAll } from './thinking-block'
import type { LogRow, ToolRow } from '@/state/episode'
import type { Span } from '@/lib/contracts'

// One sentence per tool call, written as what the agent is doing.
function describeCall(r: ToolRow): { verb: string; span?: Span } {
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
        {rows.map((r) => {
          if (r.kind === 'thinking')
            return (
              <li key={r.id} className="row-in border-b py-2">
                <ThinkingBlock text={r.text} open={allOpen || openIds.has(r.id)} onToggle={() => (allOpen ? setAllOpen(false) : toggleOne(r.id))} compact={compact} />
              </li>
            )
          const step = startStep + calls.indexOf(r)
          const d = describeCall(r)
          const pending = !r.result
          return (
            <ToolRowView key={r.id} r={r} step={step} pending={pending} running={running} compact={compact} d={d} onOpen={onOpen} />
          )
        })}
        {running && rows.length === 0 && (
          <li className="flex items-center gap-2 py-2.5 text-sm text-muted-foreground"><BracketSpinner />Reading the repository map</li>
        )}
      </ol>
    </section>
  )
}

function ToolRowView({ r, step, pending, running, compact, d, onOpen }: { r: ToolRow; step: number; pending: boolean; running: boolean; compact?: boolean; d: { verb: string; span?: Span }; onOpen: (span: Span) => void }) {
  const [open, setOpen] = useState(false)
  const expandable = !!r.result?.text
  return (
    <li className="row-in border-b py-2.5">
            <div className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] gap-x-2">
              <span className="pt-px font-mono text-xs text-muted-foreground tabular-nums">{step}</span>
              <div className="min-w-0">
                {d.span ? (
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
