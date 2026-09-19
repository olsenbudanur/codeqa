import { useMemo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { CitationItem, Span } from '@/lib/contracts'
import { formatRange, linkifyCitations, parseCitations, parseCiteHref, spanKey } from '@/lib/citations'
import type { Episode, LogRow, ToolRow } from '@/state/episode'
import { filesRead } from '@/state/episode'
import { cn } from '@/lib/utils'
import { CircleAlert } from 'lucide-react'
import { CitationChip, type Verdict } from './citation-chip'

const fmt = new Intl.NumberFormat('en-US')

// Split the trailing "Sources:" block off the answer body.
function splitSources(md: string): { body: string; notes: Map<string, string> } {
  const idx = md.search(/^Sources:\s*$/m)
  const notes = new Map<string, string>()
  if (idx === -1) return { body: md, notes }
  const body = md.slice(0, idx).trimEnd()
  for (const line of md.slice(idx).split('\n').slice(1)) {
    const m = line.match(/^\s*[-*]\s*([^\s:]+):L(\d+)(?:-L(\d+))?\s+(.*)$/)
    if (m) notes.set(`${m[1]}:${m[2]}-${m[3] ?? m[2]}`, m[4].trim())
  }
  return { body, notes }
}

// Why a citation is (not) verified, from what the episode actually did.
export function explainCitation(s: Span, rows: LogRow[], read: Span[]): string | undefined {
  const sameFile = read.filter((r) => r.path === s.path)
  if (sameFile.some((r) => contains(r, s))) return undefined
  const seenIn = rows.filter((r): r is ToolRow => r.kind === 'call' && !!r.result && (r.name === 'find_symbol' || r.name === 'grep'))
    .filter((r) => new RegExp(`${s.path.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}:L${s.start}\b`).test((r.result!.text ?? '') + ' ' + r.result!.summary))
  const readTxt = sameFile.length ? `Read in this file: ${sameFile.map((r) => formatRange(r)).join(', ')}.` : 'Nothing in this file was read.'
  if (seenIn.length) {
    const tool = seenIn[0].name
    return `${readTxt} The location came from a ${tool} hit, which shows ${tool === 'find_symbol' ? 'the signature line' : 'the matching line'}, not the ${s.end > s.start ? 'range' : 'line'} itself.`
  }
  return readTxt
}

function contains(outer: Span, inner: Span) {
  return outer.path === inner.path && outer.start <= inner.start && outer.end >= inner.end
}

export function AnswerPanel({ episode, onOpen, compact, bare }: { episode: Episode; onOpen: (s: Span) => void; compact?: boolean; bare?: boolean }) {
  const { answer, citations, stats, status, rows, format } = episode
  const read = useMemo(() => filesRead(rows), [rows])

  const { body, notes } = useMemo(() => splitSources(answer ?? ''), [answer])
  const cited = useMemo(() => (answer ? parseCitations(answer) : []), [answer])

  // Verdict per citation: the server's check_citations result when it has
  // arrived, otherwise the same grounding rule computed locally from files read.
  const verdictFor = (s: Span): Verdict => {
    const fromServer = citations?.find((c: CitationItem) => c.path === s.path && c.start === s.start && c.end === s.end)
    if (fromServer) return fromServer.verified ? 'verified' : 'unverified'
    if (status === 'running' && !citations) return read.some((r) => contains(r, s)) ? 'verified' : 'pending'
    return read.some((r) => contains(r, s)) ? 'verified' : 'unverified'
  }

  if (!answer) return null

  const uniqueCited = dedupe(cited)
  // Read ranges not fully covered by a citation.
  const consulted = dedupe(read).filter((r) => !uniqueCited.some((c) => contains(c, r)))
  const nVerified = uniqueCited.filter((c) => verdictFor(c) === 'verified').length

  return (
    <section aria-label="Answer" className={compact ? '' : 'mt-8'}>
      <header className={cn('mb-2 flex items-baseline justify-between', bare && 'sr-only')}>
        <h2 className="text-sm font-medium">Answer</h2>
        {uniqueCited.length > 0 && (
          <span className={cn('font-mono text-xs tabular-nums', nVerified === uniqueCited.length ? 'text-verified' : 'text-unverified')}>
            {nVerified} of {uniqueCited.length} citations verified
          </span>
        )}
      </header>
      {format && !format.ok && (
        <div role="alert" className="mb-4 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2.5 text-sm">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
          <div>
            <p className="font-medium">This answer fails the format check, so in training it would score 0.</p>
            <p className="mt-0.5 text-xs text-muted-foreground">{explainFormat(format.reason)}</p>
          </div>
        </div>
      )}
      <div className={bare ? '' : 'border-t pt-4'}>
        <CitedMarkdown markdown={body} verdictFor={verdictFor} onOpen={onOpen} detailFor={(s) => explainCitation(s, rows, read)} />
      </div>

      {!compact && (uniqueCited.length > 0 || consulted.length > 0) && (
        <div className="mt-6 grid gap-6 sm:grid-cols-2">
          <SourceList title="Cited" spans={uniqueCited} notes={notes} verdictFor={verdictFor} onOpen={onOpen} />
          <SourceList title="Also read" spans={consulted} notes={notes} onOpen={onOpen} />
        </div>
      )}

      {stats && (
        <dl className={cn('flex flex-wrap gap-x-6 gap-y-2 font-mono text-xs text-muted-foreground tabular-nums', bare ? 'mt-4' : 'mt-6 border-t pt-3')}>
          <Stat label="tool calls" value={String(stats.tool_calls)} />
          <Stat label="prompt tokens" value={fmt.format(stats.prompt_tokens)} />
          <Stat label="completion tokens" value={fmt.format(stats.completion_tokens)} />
          <Stat label="seconds" value={stats.seconds.toFixed(1)} />
          {stats.model_seconds !== undefined && <Stat label="s in the model" value={stats.model_seconds.toFixed(1)} />}
          {stats.tool_seconds !== undefined && <Stat label="s in tools" value={stats.tool_seconds.toFixed(1)} />}
        </dl>
      )}
    </section>
  )
}

export function CitedMarkdown({
  markdown,
  verdictFor,
  onOpen,
  className,
  detailFor,
}: {
  markdown: string
  verdictFor: (s: Span) => Verdict
  onOpen: (s: Span) => void
  className?: string
  detailFor?: (s: Span) => string | undefined
}) {
  return (
    <div className={cn('answer-prose text-[15px] leading-7', className)}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        urlTransform={(u) => u}
        components={{
          a: ({ href, children }) => {
            const span = href ? parseCiteHref(href) : null
            if (span) return <CitationChip span={span} verdict={verdictFor(span)} onOpen={onOpen} detail={detailFor?.(span)} />
            return (
              <a href={href} target="_blank" rel="noreferrer" className="underline underline-offset-2">
                {children}
              </a>
            )
          },
          p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
          code: ({ children, className }) =>
            className ? (
              <code className={cn('font-mono text-[13px]', className)}>{children}</code>
            ) : (
              <code className="rounded bg-muted px-[0.3em] py-0.5 font-mono text-[13px]">{children}</code>
            ),
          pre: ({ children }) => <pre className="mb-3 overflow-x-auto rounded-md bg-muted p-3 text-[13px]">{children}</pre>,
          ul: ({ children }) => <ul className="mb-3 list-disc pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="mb-3 list-decimal pl-5">{children}</ol>,
        }}
      >
        {linkifyCitations(markdown)}
      </Markdown>
    </div>
  )
}

// The grader's gate reasons, in plain words.
function explainFormat(reason: string): string {
  if (/no \[path/.test(reason)) return 'No [path:Lstart-Lend] citation anywhere in the answer. Every claim needs one, in exactly that form.'
  if (/tokens > cap/.test(reason)) return `Too long: ${reason.replace('answer ', '')}. The cap is part of the prompt, so this counts as not following the format.`
  if (/no final answer/.test(reason)) return 'The agent never gave a final answer (it ran out of turns or budget).'
  if (/stop_reason=/.test(reason)) return `The episode ended abnormally (${reason.replace('stop_reason=', '')}): a malformed tool call or a truncated generation.`
  return reason
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <dd className="text-foreground">{value}</dd>
      <dt>{label}</dt>
    </div>
  )
}

function SourceList({
  title,
  spans,
  notes,
  verdictFor,
  onOpen,
}: {
  title: string
  spans: Span[]
  notes: Map<string, string>
  verdictFor?: (s: Span) => Verdict
  onOpen: (s: Span) => void
}) {
  return (
    <div>
      <h3 className="mb-1.5 text-xs text-muted-foreground">{title}</h3>
      {spans.length === 0 ? (
        <p className="text-xs text-muted-foreground">None</p>
      ) : (
        <ul className="space-y-1">
          {spans.map((s) => {
            const v = verdictFor?.(s)
            const note = notes.get(`${s.path}:${s.start}-${s.end}`)
            return (
              <li key={spanKey(s)} className="text-sm">
                <button
                  type="button"
                  onClick={() => onOpen(s)}
                  className="group flex w-full items-baseline gap-2 text-left hover:underline focus-visible:underline focus-visible:outline-none"
                >
                  <span
                    className={cn(
                      'mt-2 size-1.5 shrink-0 self-start rounded-full',
                      v === 'verified' && 'bg-verified',
                      v === 'unverified' && 'bg-unverified',
                      (!v || v === 'pending') && 'bg-border',
                    )}
                    aria-hidden
                  />
                  <span className="min-w-0">
                    <span className="font-mono text-[13px]">
                      {s.path} <span className="text-muted-foreground">{formatRange(s)}</span>
                    </span>
                    {note && <span className="block text-xs text-muted-foreground">{note}</span>}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

function dedupe(spans: Span[]): Span[] {
  const seen = new Set<string>()
  return spans.filter((s) => {
    const k = spanKey(s)
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}
