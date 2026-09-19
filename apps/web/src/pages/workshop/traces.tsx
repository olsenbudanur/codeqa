import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, Columns2 } from 'lucide-react'
import { navigate } from '@/lib/router'
import type { Span } from '@/lib/contracts'
import { formatRange } from '@/lib/citations'
import { fmtNum, fmtWhen, workshop, type TraceDetail, type TraceList } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { emptyEpisode, episodeReducer, type Episode } from '@/state/episode'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { Ledger } from '@/components/research/ledger'
import { AnswerPanel } from '@/components/answer/answer-panel'
import { FileViewer } from '@/components/file/file-viewer'
import { Chip, ErrorNote, Loading, Page, Panel } from '@/components/workshop/ui'
import { RawConversation } from '@/components/workshop/raw-conversation'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

export function TracesPage({ parts, params }: { parts: string[]; params: URLSearchParams }) {
  if (parts[0] === 'compare') return <TraceCompare a={params.get('a') ?? ''} b={params.get('b') ?? ''} />
  if (parts.length > 0) return <TraceDetailPage id={parts.join('/')} />
  return <TraceListPage params={params} />
}

// --- list ---------------------------------------------------------------------

function TraceListPage({ params }: { params: URLSearchParams }) {
  const [f, setF] = useState({
    q: params.get('q') ?? '',
    profile: params.get('profile') ?? '',
    run: params.get('run') ?? '',
    source: params.get('source') ?? '',
    type: params.get('type') ?? '',
    stop: params.get('stop') ?? '',
    has_citations: params.get('has_citations') ?? '',
  })
  const [data, setData] = useState<TraceList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [picked, setPicked] = useState<string[]>([])
  useEffect(() => {
    const t = window.setTimeout(() => {
      workshop
        .traces({ ...f, has_citations: f.has_citations === '' ? undefined : f.has_citations === 'yes', limit: 200 })
        .then(setData)
        .catch((e: Error) => setError(e.message))
    }, 150)
    return () => window.clearTimeout(t)
  }, [f])
  const set = (k: keyof typeof f) => (v: string) => setF((s) => ({ ...s, [k]: v === 'any' ? '' : v }))

  return (
    <Page
      title="Traces"
      wide
      actions={
        picked.length === 2 ? (
          <Button size="sm" onClick={() => navigate(`/workshop/traces/compare?a=${encodeURIComponent(picked[0])}&b=${encodeURIComponent(picked[1])}`)}>
            <Columns2 />
            Compare the two picked
          </Button>
        ) : (
          <span className="text-xs text-muted-foreground">Tick two traces to compare them</span>
        )
      }
    >
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Input value={f.q} onChange={(e) => setF((s) => ({ ...s, q: e.target.value }))} placeholder="Search question or task id" className="h-8 w-[240px]" />
        {data && (
          <>
            <Facet label="profile" value={f.profile} options={data.facets.profile} onChange={set('profile')} />
            <Facet label="run / set" value={f.run} options={data.facets.run} onChange={set('run')} />
            <Facet label="source" value={f.source} options={data.facets.source} onChange={set('source')} />
            <Facet label="type" value={f.type} options={data.facets.task_type} onChange={set('type')} />
            <Facet label="stop" value={f.stop} options={data.facets.stop_reason} onChange={set('stop')} />
            <Facet label="citations" value={f.has_citations} options={['yes', 'no']} onChange={set('has_citations')} />
          </>
        )}
      </div>
      {error && <ErrorNote error={error} />}
      {!data && !error && <Loading what="traces" />}
      {data && (
        <div className="overflow-x-auto rounded-lg border bg-background">
          <table className="w-full text-[13px]">
            <thead className="text-left text-xs text-muted-foreground">
              <tr className="border-b">
                <th className="w-8 px-3 py-2" />
                <th className="px-2 py-2 font-normal">Task</th>
                <th className="px-2 py-2 font-normal">Profile</th>
                <th className="px-2 py-2 font-normal">Run / set</th>
                <th className="px-2 py-2 font-normal">Source / type</th>
                <th className="px-2 py-2 font-normal">Stop</th>
                <th className="px-2 py-2 text-right font-normal">Turns</th>
                <th className="px-2 py-2 text-right font-normal">Calls</th>
                <th className="px-2 py-2 text-right font-normal">Prompt tok</th>
                <th className="px-2 py-2 text-right font-normal">Reward</th>
                <th className="px-2 py-2 font-normal">When</th>
              </tr>
            </thead>
            <tbody>
              {data.traces.map((t) => (
                <tr key={t.id} className="cursor-pointer border-b last:border-0 hover:bg-accent/60" onClick={() => navigate(`/workshop/traces/${t.id}`)}>
                  <td className="px-3 py-1.5" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      aria-label="Pick for comparison"
                      checked={picked.includes(t.id)}
                      onChange={(e) => setPicked((p) => (e.target.checked ? [...p, t.id].slice(-2) : p.filter((x) => x !== t.id)))}
                    />
                  </td>
                  <td className="max-w-[360px] px-2 py-1.5">
                    <span className="block truncate">{t.question || <span className="text-muted-foreground">(no question recorded)</span>}</span>
                    <span className="block truncate font-mono text-[11px] text-muted-foreground">{t.task_id}</span>
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[12px]">{t.profile}</td>
                  <td className="px-2 py-1.5 font-mono text-[12px] text-muted-foreground">{t.run}</td>
                  <td className="px-2 py-1.5 font-mono text-[12px] text-muted-foreground">{[t.source, t.task_type].filter(Boolean).join(' / ') || '–'}</td>
                  <td className="px-2 py-1.5"><Chip tone={t.stop_reason === 'answer' ? 'good' : t.stop_reason ? 'warn' : 'default'}>{t.stop_reason ?? '–'}</Chip></td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">{t.turns ?? '–'}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">{t.tool_calls ?? '–'}</td>
                  <td className="px-2 py-1.5 text-right font-mono tabular-nums">{t.prompt_tokens != null ? fmtNum(t.prompt_tokens, 0) : '–'}</td>
                  <td className={cn('px-2 py-1.5 text-right font-mono tabular-nums', (t.reward ?? 0) > 0 && 'text-verified')}>{t.reward === null ? <span className="text-muted-foreground">–</span> : fmtNum(t.reward, 2)}</td>
                  <td className="px-2 py-1.5 font-mono text-[11px] text-muted-foreground">{fmtWhen(t.mtime)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="border-t px-3 py-2 font-mono text-[11px] text-muted-foreground">
            {data.traces.length} of {data.total.toLocaleString()} shown
          </p>
        </div>
      )}
    </Page>
  )
}

function Facet({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <Select value={value || 'any'} onValueChange={onChange}>
      <SelectTrigger className="h-8 w-auto min-w-[120px]" aria-label={label}>
        <SelectValue>{value || `${label}: any`}</SelectValue>
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="any">any {label}</SelectItem>
        {options.map((o) => (
          <SelectItem key={o} value={o}>{o}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

// --- detail ---------------------------------------------------------------------

function toEpisode(t: TraceDetail): Episode {
  let ep = episodeReducer(emptyEpisode, { type: 'start', question: t.question, repoId: t.repo_id ?? '', profile: t.profile })
  for (const ev of t.events) ep = episodeReducer(ep, { type: 'event', event: ev })
  if (t.citations.length) ep = { ...ep, citations: t.citations.map((c) => ({ path: c.path, start: c.start, end: c.end, verified: c.verified })) }
  return ep
}

export function TraceView({ trace, onOpen, compact }: { trace: TraceDetail; onOpen: (s: Span) => void; compact?: boolean }) {
  const ep = useMemo(() => toEpisode(trace), [trace])
  return (
    <div>
      {!compact && <p className="mb-5 text-[16px] leading-snug font-medium">{trace.question || <span className="text-muted-foreground">(question not recorded in this rollout)</span>}</p>}
      <Ledger rows={ep.rows} running={false} onOpen={onOpen} budget={undefined} />
      <AnswerPanel episode={ep} onOpen={onOpen} />
      {!ep.answer && <p className="mt-6 border-t pt-3 text-sm text-muted-foreground">No final answer ({String(trace.stats.stop_reason ?? 'unknown')}).</p>}
    </div>
  )
}

const COMPONENTS: [string, string][] = [
  ['format_ok', 'format'],
  ['citations_parse', 'citations parse'],
  ['citations_exist', 'citations exist'],
  ['citations_grounded', 'citations grounded'],
  ['identifier_grounded', 'identifier grounded'],
  ['correctness', 'correctness'],
  ['efficiency', 'efficiency'],
]

export function GradePanel({ trace }: { trace: TraceDetail }) {
  const g = trace.grade
  return (
    <div className="space-y-4">
      <Panel title="Grade" aside={g.source}>
        <div className="flex items-baseline gap-3">
          <span className={cn('font-mono text-[26px] leading-none tabular-nums', (g.reward ?? 0) > 0 ? 'text-verified' : g.reward === 0 ? 'text-unverified' : 'text-muted-foreground')}>
            {g.reward === null || g.reward === undefined ? '–' : fmtNum(g.reward, 2)}
          </span>
          <span className="text-xs text-muted-foreground">reward</span>
          {g.gate_failed && <Chip tone="bad">gate: {g.gate_failed}</Chip>}
        </div>
        <dl className="mt-3 divide-y text-[12.5px]">
          {COMPONENTS.map(([k, label]) => {
            const v = g.components[k]
            return (
              <div key={k} className="flex items-center justify-between py-1">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className={cn('font-mono tabular-nums', v === null || v === undefined ? 'text-muted-foreground' : v >= 1 ? 'text-verified' : v <= 0 ? 'text-unverified' : '')}>
                  {v === null || v === undefined ? (k === 'correctness' && g.source === 'check_citations' ? 'not graded' : '–') : fmtNum(v, 2)}
                </dd>
              </div>
            )
          })}
        </dl>
        {g.notes && <p className="mt-3 font-mono text-[11.5px] text-muted-foreground">{g.notes}</p>}
      </Panel>
      <Panel title="Citations" aside={`${trace.citations.filter((c) => c.verified).length} of ${trace.citations.length} verified`}>
        {trace.citations.length === 0 ? (
          <p className="text-sm text-muted-foreground">None in the answer{trace.kind === 'rollout' ? ' (rollouts carry no repo, so none are checked)' : ''}.</p>
        ) : (
          <table className="w-full text-[12px]">
            <thead className="text-left text-muted-foreground">
              <tr>
                <th className="py-1 font-normal">citation</th>
                <th className="py-1 text-right font-normal">exists</th>
                <th className="py-1 text-right font-normal">grounded</th>
              </tr>
            </thead>
            <tbody>
              {trace.citations.map((c, i) => (
                <tr key={i} className="border-t">
                  <td className="py-1 font-mono">{c.path} <span className="text-muted-foreground">{formatRange(c)}</span></td>
                  <td className={cn('py-1 text-right font-mono', c.exists ? 'text-verified' : 'text-destructive')}>{c.exists ? 'yes' : 'no'}</td>
                  <td className={cn('py-1 text-right font-mono', c.grounded ? 'text-verified' : 'text-unverified')}>{c.grounded ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
      <Panel title="Stats">
        <dl className="grid grid-cols-2 gap-y-1 font-mono text-[12px]">
          {Object.entries(trace.stats)
            .filter(([k]) => k !== 'files_read')
            .map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted-foreground">{k}</dt>
                <dd className="text-right tabular-nums">{v === null || v === undefined ? '–' : typeof v === 'number' ? fmtNum(v, Number.isInteger(v) ? 0 : 2) : String(v)}</dd>
              </div>
            ))}
        </dl>
      </Panel>
    </div>
  )
}

function useTrace(id: string) {
  const [trace, setTrace] = useState<TraceDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setTrace(null)
    workshop.trace(id).then(setTrace).catch((e: Error) => setError(e.message))
  }, [id])
  return { trace, error }
}

function FileSheet({ repoId, span, onClose }: { repoId: string | null; span: Span | null; onClose: () => void }) {
  return (
    <Sheet open={!!span} onOpenChange={(o) => !o && onClose()}>
      <SheetContent side="right" className="w-full p-0 sm:max-w-[640px]" showCloseButton={false}>
        <SheetTitle className="sr-only">File</SheetTitle>
        {span && repoId ? <FileViewer repoId={repoId} span={span} onClose={onClose} /> : <p className="p-4 text-sm text-muted-foreground">This trace has no repository attached, so the file cannot be opened.</p>}
      </SheetContent>
    </Sheet>
  )
}

function TraceDetailPage({ id }: { id: string }) {
  const { trace, error } = useTrace(id)
  const [span, setSpan] = useState<Span | null>(null)
  const [others, setOthers] = useState<{ id: string; profile: string; run: string }[]>([])
  useEffect(() => {
    if (!trace || trace.kind === 'rollout') return
    workshop.traces({ task_id: trace.task_id, limit: 50 }).then((r) => setOthers(r.traces.filter((t) => t.id !== trace.id).map((t) => ({ id: t.id, profile: t.profile, run: t.run })))).catch(() => {})
  }, [trace])
  return (
    <Page
      title={
        <span className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[15px]">{trace?.task_id ?? id}</span>
          {trace && <Chip>{trace.profile}</Chip>}
          {trace && <Chip>{trace.run}</Chip>}
          {trace?.task?.source && <Chip>{trace.task.source} / {trace.task.task_type}</Chip>}
        </span>
      }
      wide
      actions={
        others.length > 0 && (
          <span className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            same task:
            {others.slice(0, 4).map((o) => (
              <Button key={o.id} variant="outline" size="sm" className="h-7" onClick={() => navigate(`/workshop/traces/compare?a=${encodeURIComponent(id)}&b=${encodeURIComponent(o.id)}`)}>
                vs {o.profile}
                <Columns2 />
              </Button>
            ))}
          </span>
        )
      }
    >
      {error && <ErrorNote error={error} />}
      {!trace && !error && <Loading what="trace" />}
      {trace && (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
          <Tabs defaultValue="log">
            <TabsList>
              <TabsTrigger value="log">Research log</TabsTrigger>
              <TabsTrigger value="raw">Everything the model saw{trace.messages ? ` (${trace.messages.length})` : ''}</TabsTrigger>
            </TabsList>
            <TabsContent value="log" className="mt-3">
              <div className="rounded-lg border bg-background p-5">
                <TraceView trace={trace} onOpen={setSpan} />
                {trace.task && trace.task.grading && (
                  <details className="mt-6 border-t pt-3 text-sm">
                    <summary className="cursor-pointer text-muted-foreground">Task gold</summary>
                    <pre className="mt-2 overflow-x-auto rounded bg-muted p-2 font-mono text-[11.5px]">{JSON.stringify(trace.task.grading, null, 1)}</pre>
                  </details>
                )}
              </div>
            </TabsContent>
            <TabsContent value="raw" className="mt-3">
              {trace.messages ? <RawConversation messages={trace.messages} note={trace.messages_note} /> : <p className="text-sm text-muted-foreground">No message log for this trace.</p>}
            </TabsContent>
          </Tabs>
          <GradePanel trace={trace} />
        </div>
      )}
      <FileSheet repoId={trace?.repo_id ?? null} span={span} onClose={() => setSpan(null)} />
    </Page>
  )
}

// --- compare: two traces on the same task, side by side ----------------------------

function TraceCompare({ a, b }: { a: string; b: string }) {
  const [data, setData] = useState<{ a: TraceDetail; b: TraceDetail } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<{ repoId: string | null; span: Span } | null>(null)
  useEffect(() => {
    if (!a || !b) return
    workshop.compare(a, b).then(setData).catch((e: Error) => setError(e.message))
  }, [a, b])
  const head = (t: TraceDetail) => `${t.stats.tool_calls ?? '–'} calls, ${fmtNum(t.stats.prompt_tokens as number, 0)} prompt tokens, ${t.stats.seconds ? `${fmtNum(t.stats.seconds as number, 1)} s` : 'no timing'}`
  return (
    <Page title="Same task, two episodes" wide actions={data && <Button variant="ghost" size="sm" onClick={() => navigate(`/workshop/traces?task_id=${encodeURIComponent(data.a.task_id)}`)}>All traces of this task <ArrowRight /></Button>}>
      {error && <ErrorNote error={error} />}
      {!data && !error && <Loading what="traces" />}
      {data && (
        <>
          <p className="mb-4 text-[16px] leading-snug font-medium">{data.a.question || data.b.question}</p>
          <div className="grid gap-4 lg:grid-cols-2">
            {[data.a, data.b].map((t) => (
              <section key={t.id} className="rounded-lg border bg-background">
                <header className="flex flex-wrap items-center gap-2 border-b px-4 py-2">
                  <span className="font-mono text-[13px]">{t.profile}</span>
                  <Chip>{t.run}</Chip>
                  <span className="ml-auto font-mono text-[11.5px] text-muted-foreground">{head(t)}</span>
                </header>
                <div className="p-4">
                  <TraceView trace={t} onOpen={(span) => setOpen({ repoId: t.repo_id, span })} compact />
                </div>
                <div className="border-t p-4">
                  <GradePanel trace={t} />
                </div>
              </section>
            ))}
          </div>
        </>
      )}
      <FileSheet repoId={open?.repoId ?? null} span={open?.span ?? null} onClose={() => setOpen(null)} />
    </Page>
  )
}
