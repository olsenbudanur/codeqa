import { useState } from 'react'
import { Check, ChevronRight } from 'lucide-react'
import type { LogRow, ToolRow } from '@/state/episode'
import type { Span } from '@/lib/contracts'
import { cn } from '@/lib/utils'

// One sentence per tool call, written as what the agent is doing.
function describeCall(r: ToolRow): { verb: string; span?: Span } {
  const a = r.args
  switch (r.name) {
    case 'read_file': {
      const path = String(a.path ?? '')
      const start = Number(a.start ?? 1)
      const end = Number(a.end ?? start)
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
    return `${n} ${n === 1 ? 'line' : 'lines'}`
  }
  return res.summary
}

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
  return (
    <section aria-label="Research log">
      {!compact && (
      <header className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-medium">Research</h2>
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {total} {total === 1 ? 'call' : 'calls'}
          {budget ? ` of ${budget}` : ''}
        </span>
      </header>
      )}
      <ol className={compact ? '' : 'border-t'}>
        {rows.map((r) => {
          if (r.kind === 'thinking') return <ThinkingRow key={r.id} text={r.text} />
          const step = startStep + calls.indexOf(r)
          const d = describeCall(r)
          const pending = !r.result
          return (
            <li key={r.id} className="row-in grid grid-cols-[1.5rem_minmax(0,1fr)_auto] gap-x-2 border-b py-2.5">
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
              <span className="pt-px text-right font-mono text-xs text-muted-foreground tabular-nums">
                {pending ? (
                  <span className={cn('inline-block size-2 rounded-full bg-foreground/60', running && 'animate-pulse')} aria-label="Running" />
                ) : (
                  <span key="done" className="row-in inline-block">
                    {compact ? (
                      <Check className="size-3.5 text-verified" strokeWidth={3} aria-label="Done" />
                    ) : (
                      <>
                        <span className="max-sm:hidden">{resultLabel(r)}</span>
                        <span className="sm:hidden">{fmt.format(r.result!.chars)} chars</span>
                      </>
                    )}
                  </span>
                )}
              </span>
            </li>
          )
        })}
        {running && rows.length === 0 && (
          <li className="py-2.5 text-sm text-muted-foreground">Reading the repository map</li>
        )}
      </ol>
    </section>
  )
}

function ThinkingRow({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  return (
    <li className="row-in border-b py-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="grid w-full grid-cols-[1.5rem_minmax(0,1fr)] gap-x-2 text-left text-xs text-muted-foreground"
      >
        <ChevronRight className={cn('size-3.5 transition-transform', open && 'rotate-90')} aria-hidden />
        <span className={cn(!open && 'truncate')}>{text}</span>
      </button>
    </li>
  )
}
