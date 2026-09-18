import { useEffect, useState, type ReactNode } from 'react'
import { Shuffle } from 'lucide-react'
import { navigate } from '@/lib/router'
import { repoName } from '@/lib/repo'
import type { Span } from '@/lib/contracts'
import { formatRange } from '@/lib/citations'
import { fmtNum, fmtPct, workshop, type DataSummary, type Passrate, type TaskCard } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { Histogram } from '@/components/workshop/charts'
import { Chip, ErrorNote, Loading, Page, Panel, Stat } from '@/components/workshop/ui'
import { FileViewer } from '@/components/file/file-viewer'

const TYPES = ['locate', 'value', 'enumerate', 'trace', 'explain']

// What each task type asks and how the grader scores it (C5 grading, C7 grade).
const TYPE_HELP: Record<string, { asks: string; graded: string }> = {
  locate: { asks: 'Where is X defined? One file or symbol.', graded: 'exact match on the gold path or symbol; no judge' },
  value: { asks: 'What is the value of X? A literal.', graded: 'normalized string match on the gold literal; no judge' },
  enumerate: { asks: 'Which files/symbols do X? Several.', graded: 'F1 between the answer and the gold list; no judge' },
  trace: { asks: 'What happens when X is called? A chain.', graded: 'rubric items checked by the judge (Haiku)' },
  explain: { asks: 'How or why does X work?', graded: 'rubric items checked by the judge (Haiku)' },
}
const SOURCE_HELP: Record<string, string> = {
  structural: 'generated from the index: definitions, importers, call chains; gold is exact; runs on the docstring-stripped snapshot',
  codescout: 'SWE-rebench-code-search rows rewritten by Haiku; gold is the entity the issue was about',
  deepcodebench: 'Qodo DeepCodeBench questions; the facts list is the rubric',
  teacher: 'written by Claude while browsing the repo with the same tools; kept if a blind Haiku answered ≥ 0.5',
  sweqa: 'SWE-QA-Bench, held out, repos never trained on',
}

export function DataPage({ params }: { params: URLSearchParams }) {
  const [summary, setSummary] = useState<DataSummary | null>(null)
  const [passrate, setPassrate] = useState<Passrate | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    workshop.dataSummary().then(setSummary).catch((e: Error) => setError(e.message))
    workshop.passrate(10).then(setPassrate).catch((e: Error) => setError(e.message))
  }, [])

  return (
    <Page title="Data" wide>
      {error && <ErrorNote error={error} />}
      {!summary && !error && <Loading what="data" />}
      {summary && (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <Panel title="Tasks by source and type" aside="C5 records">
              <details className="mb-4 rounded-md border bg-muted/40 px-3 py-2 text-[12.5px]">
                <summary className="cursor-pointer text-muted-foreground">What the columns mean</summary>
                <dl className="mt-2 grid gap-x-6 gap-y-1.5 sm:grid-cols-[6rem_minmax(0,1fr)]">
                  {TYPES.map((t) => (
                    <div key={t} className="contents">
                      <dt className="font-mono">{t}</dt>
                      <dd><span>{TYPE_HELP[t].asks}</span> <span className="text-muted-foreground">Graded: {TYPE_HELP[t].graded}.</span></dd>
                    </div>
                  ))}
                </dl>
                <dl className="mt-3 grid gap-x-6 gap-y-1.5 border-t pt-2 sm:grid-cols-[6rem_minmax(0,1fr)]">
                  {Object.entries(SOURCE_HELP).map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt className="font-mono">{k}</dt>
                      <dd className="text-muted-foreground">{v}</dd>
                    </div>
                  ))}
                </dl>
              </details>
              <div className="space-y-4">
                {Object.entries(summary.counts).map(([file, bySource]) => {
                  const total = Object.values(bySource).reduce((a, m) => a + Object.values(m).reduce((x, y) => x + y, 0), 0)
                  return (
                    <div key={file}>
                      <div className="mb-1 flex items-baseline justify-between font-mono text-[12px]">
                        <span>{file}.jsonl</span>
                        <span className="text-muted-foreground">{total.toLocaleString()}</span>
                      </div>
                      <table className="w-full text-[12.5px]">
                        <thead className="text-left text-xs text-muted-foreground">
                          <tr>
                            <th className="py-1 font-normal">source</th>
                            {TYPES.map((t) => (
                              <th key={t} className="py-1 text-right font-normal" title={`${TYPE_HELP[t].asks} Graded: ${TYPE_HELP[t].graded}.`}>{t}</th>
                            ))}
                            <th className="py-1 text-right font-normal">total</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(bySource).sort().map(([src, m]) => (
                            <tr key={src} className="border-t">
                              <td className="py-1 font-mono">{src}</td>
                              {TYPES.map((t) => (
                                <td key={t} className="py-1 text-right font-mono tabular-nums">{m[t] ?? <span className="text-muted-foreground">·</span>}</td>
                              ))}
                              <td className="py-1 text-right font-mono tabular-nums">{Object.values(m).reduce((a, b) => a + b, 0)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )
                })}
              </div>
            </Panel>
            <div className="space-y-4">
              <Panel title="Difficulty at step 0" aside="base model, 2–4 samples per task">
                {passrate ? (
                  <>
                    <Histogram counts={passrate.histogram} window={passrate.window} total={passrate.n_scored} />
                    <div className="mt-4 flex flex-wrap gap-6">
                      <Stat label="too hard (base never right)" value={(passrate.buckets.too_hard ?? 0).toLocaleString()} />
                      <Stat label="kept" value={(passrate.buckets.kept ?? 0).toLocaleString()} tone="good" />
                      <Stat label="too easy" value={(passrate.buckets.too_easy ?? 0).toLocaleString()} />
                    </div>
                  </>
                ) : (
                  <Loading what="pass rates" />
                )}
              </Panel>
              {passrate && (
                <Panel title="Step-0 rates by source">
                  <table className="w-full text-[12.5px]">
                    <thead className="text-left text-xs text-muted-foreground">
                      <tr>
                        <th className="py-1 font-normal">source</th>
                        <th className="py-1 text-right font-normal">answered</th>
                        <th className="py-1 text-right font-normal">format</th>
                        <th className="py-1 text-right font-normal">found gold</th>
                        <th className="py-1 text-right font-normal">correct</th>
                        <th className="py-1 text-right font-normal">calls</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(passrate.per_source).sort().map(([src, m]) => (
                        <tr key={src} className="border-t">
                          <td className="py-1 font-mono">{src}</td>
                          <td className="py-1 text-right font-mono">{fmtPct(m.answered)}</td>
                          <td className="py-1 text-right font-mono">{fmtPct(m.format_ok)}</td>
                          <td className="py-1 text-right font-mono">{fmtPct(m.found)}</td>
                          <td className="py-1 text-right font-mono">{fmtPct(m.correct_lenient_all)}</td>
                          <td className="py-1 text-right font-mono">{fmtNum(m.tool_calls, 1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Panel>
              )}
            </div>
          </div>

          <TaskBrowser initial={params} />

          <Panel title="Repositories" aside={`${summary.repos.length} snapshots`}>
            <div className="max-h-[420px] overflow-auto">
              <table className="w-full text-[12.5px]">
                <thead className="sticky top-0 bg-background text-left text-xs text-muted-foreground">
                  <tr className="border-b">
                    <th className="py-1.5 font-normal">repo</th>
                    <th className="py-1.5 text-right font-normal">files</th>
                    <th className="py-1.5 text-right font-normal">lines</th>
                    <th className="py-1.5 text-right font-normal">symbols</th>
                    <th className="py-1.5 pl-3 font-normal">index</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.repos.map((r) => (
                    <tr key={r.repo_id} className="border-b last:border-0">
                      <td className="py-1.5 font-mono">{r.repo_id}</td>
                      <td className="py-1.5 text-right font-mono tabular-nums">{r.files.toLocaleString()}</td>
                      <td className="py-1.5 text-right font-mono tabular-nums">{r.lines.toLocaleString()}</td>
                      <td className="py-1.5 text-right font-mono tabular-nums">{r.symbols?.toLocaleString() ?? '–'}</td>
                      <td className="py-1.5 pl-3">
                        <span className="flex gap-1">
                          {r.map && <Chip tone="good">map</Chip>}
                          {r.summaries && <Chip tone="good">summaries</Chip>}
                          {r.nodoc && <Chip>nodoc</Chip>}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </Page>
  )
}

// Training data as cards, not a table.
function TaskBrowser({ initial }: { initial: URLSearchParams }) {
  const [q, setQ] = useState(initial.get('q') ?? '')
  const [source, setSource] = useState(initial.get('source') ?? 'all')
  const [type, setType] = useState(initial.get('type') ?? 'all')
  const [split, setSplit] = useState(initial.get('split') ?? 'train')
  const [bucket, setBucket] = useState(initial.get('bucket') ?? 'all')
  const [random, setRandom] = useState(0)
  const [result, setResult] = useState<{ total: number; tasks: TaskCard[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<{ repoId: string; span: Span } | null>(null)

  useEffect(() => {
    const t = window.setTimeout(() => {
      workshop
        .tasks({ q, source: source === 'all' ? undefined : source, type: type === 'all' ? undefined : type, split, bucket: bucket === 'all' ? undefined : bucket, limit: 12, random: random > 0 })
        .then(setResult)
        .catch((e: Error) => setError(e.message))
    }, 200)
    return () => window.clearTimeout(t)
  }, [q, source, type, split, bucket, random])

  return (
    <Panel title="Task browser" aside={result ? `${result.total.toLocaleString()} match` : ''}>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search questions" className="h-8 w-[240px]" />
        <Pick value={split} onChange={setSplit} options={['train', 'eval', 'raw']} label="Split" />
        <Pick value={source} onChange={setSource} options={['all', 'structural', 'codescout', 'deepcodebench', 'teacher', 'sweqa']} label="Source" />
        <Pick value={type} onChange={setType} options={['all', ...TYPES]} label="Type" />
        <Pick value={bucket} onChange={setBucket} options={['all', 'too_easy', 'kept', 'too_hard']} label="Pass-rate bucket" />
        <Button variant="outline" size="sm" onClick={() => setRandom((n) => n + 1)}>
          <Shuffle />
          Surprise me
        </Button>
      </div>
      {error && <ErrorNote error={error} />}
      {!result && !error && <Loading what="tasks" />}
      {result && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {result.tasks.map((t) => (
            <TaskCardView key={t.task_id} task={t} onOpen={(span) => setOpen({ repoId: t.repo_id, span })} />
          ))}
          {result.tasks.length === 0 && <p className="text-sm text-muted-foreground">No tasks match.</p>}
        </div>
      )}
      <Sheet open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
        <SheetContent side="right" className="w-full p-0 sm:max-w-[640px]" showCloseButton={false}>
          <SheetTitle className="sr-only">File</SheetTitle>
          {open && <FileViewer repoId={open.repoId} span={open.span} onClose={() => setOpen(null)} />}
        </SheetContent>
      </Sheet>
    </Panel>
  )
}

function Pick({ value, onChange, options, label }: { value: string; onChange: (v: string) => void; options: string[]; label: string }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-8 w-auto min-w-[120px]" aria-label={label}>
        <SelectValue>{value}</SelectValue>
      </SelectTrigger>
      <SelectContent>
        {options.map((o) => (
          <SelectItem key={o} value={o}>{o}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function Bar({ label, value }: { label: string; value: number | null | undefined }) {
  const v = value ?? 0
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="w-16 text-muted-foreground">{label}</span>
      <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <span className="block h-full rounded-full" style={{ width: `${Math.round(v * 100)}%`, background: 'var(--series-1)' }} />
      </span>
      <span className="w-9 text-right font-mono tabular-nums">{value === null || value === undefined ? '–' : fmtPct(value)}</span>
    </div>
  )
}

export function TaskCardView({ task, onOpen, compact }: { task: TaskCard; onOpen: (span: Span) => void; compact?: boolean }) {
  const g = task.grading
  const help = TYPE_HELP[task.task_type]
  const maxRubric = compact ? 3 : 8
  return (
    <article className="flex flex-col gap-3 rounded-lg border bg-background p-3.5">
      <p className="text-[14px] leading-5 font-medium">{task.question}</p>
      <div className="flex flex-wrap gap-1">
        <Chip>{repoName({ repo_id: task.repo_id })}</Chip>
        <span title={help ? `${help.asks} Graded: ${help.graded}.` : undefined}><Chip>{task.task_type}</Chip></span>
        <span title={SOURCE_HELP[task.source]}><Chip>{task.source}</Chip></span>
        {task.bucket && (
          <span title="Step-0 difficulty bucket: kept = the untrained model is right 10–90% of the time"><Chip tone={task.bucket === 'kept' ? 'good' : task.bucket === 'too_hard' ? 'bad' : 'warn'}>{task.bucket.replace('_', ' ')}</Chip></span>
        )}
      </div>

      <Section label="Correct answer" hint={help?.graded}>
        {!g.reference_answer && !g.rubric?.length && (g.expected_paths?.length > 0 || g.expected_symbols?.length > 0 || g.expected_literal) && (
          <p className="text-[12px] text-muted-foreground">
            No prose to compare against: the grader pulls {g.expected_literal ? 'the value' : g.expected_symbols?.length ? 'the symbols' : 'the file paths'} out of the answer and scores
            them {(g.expected_paths?.length ?? 0) + (g.expected_symbols?.length ?? 0) > 1 ? 'by F1 against this list' : 'by exact match'}.
          </p>
        )}
        {g.expected_paths?.length > 0 && (
          <Row label={g.expected_paths.length > 1 ? 'files (F1)' : 'file'}>
            {g.expected_paths.map((p) => (
              <CodeChip key={p} text={p} onClick={() => onOpen({ path: p, start: 1, end: 1 })} />
            ))}
          </Row>
        )}
        {g.expected_symbols?.length > 0 && (
          <Row label={g.expected_symbols.length > 1 ? 'symbols (F1)' : 'symbol'}>
            {g.expected_symbols.map((sym) => (
              <CodeChip key={sym} text={sym} onClick={() => onOpen({ path: sym.split(':')[0], start: 1, end: 1 })} />
            ))}
          </Row>
        )}
        {g.expected_literal && (
          <Row label="value">
            <CodeChip text={g.expected_literal} />
          </Row>
        )}
        {g.rubric?.length > 0 && (
          <Row label={`rubric, ${g.rubric.length} items`}>
            <ul className="w-full space-y-0.5 text-[12px]">
              {g.rubric.slice(0, maxRubric).map((r, i) => (
                <li key={i} className="flex gap-1.5">
                  <span className="mt-[5px] size-2 shrink-0 rounded-sm border" aria-hidden />
                  <span>{r}</span>
                </li>
              ))}
              {g.rubric.length > maxRubric && <li className="pl-3.5 text-muted-foreground">{g.rubric.length - maxRubric} more</li>}
            </ul>
          </Row>
        )}
        {g.reference_answer && (
          <Row label="reference answer">
            <details className="w-full text-[12px]">
              <summary className="cursor-pointer text-muted-foreground">show</summary>
              <p className="mt-1 whitespace-pre-wrap text-muted-foreground">{g.reference_answer}</p>
            </details>
          </Row>
        )}
        {!g.expected_paths?.length && !g.expected_symbols?.length && !g.expected_literal && !g.rubric?.length && !g.reference_answer && (
          <p className="text-[12px] text-muted-foreground">no gold recorded</p>
        )}
      </Section>

      {g.required_citations?.length > 0 && (
        <Section label="Must cite" hint="the answer has to point at these lines, and the agent has to have read them">
          <Row>
            {g.required_citations.map((c) => (
              <CodeChip key={`${c.path}${c.start}`} text={`${c.path} ${formatRange(c)}`} onClick={() => onOpen(c)} />
            ))}
          </Row>
        </Section>
      )}

      {task.passrate && (
        <Section label="Untrained model at step 0" hint={`${task.passrate.n ?? '?'} samples of qwen4b-base; decides the difficulty bucket`}>
          <div className="w-full space-y-1">
            <Bar label="answered" value={task.passrate.answered} />
            <Bar label="format ok" value={task.passrate.format_ok} />
            <Bar label="found gold" value={task.passrate.found} />
            <Bar label="correct" value={task.passrate.correct_lenient_all} />
          </div>
        </Section>
      )}

      <div className="mt-auto flex items-center gap-3 border-t pt-2 font-mono text-[11px] text-muted-foreground">
        <span>{task.task_id}</span>
        {task.budget ? <span>{task.budget.max_tool_calls} calls, {task.budget.max_answer_tokens} answer tokens</span> : <span>default budget for {task.task_type}</span>}
        {task.example_traces.length > 0 && (
          <button type="button" className="ml-auto text-foreground hover:underline" onClick={() => navigate(`/workshop/traces/${task.example_traces[0]}`)}>
            example episode
          </button>
        )}
      </div>
    </article>
  )
}

function Section({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div className="border-t pt-2">
      <p className="mb-1 flex items-baseline gap-2 text-[11px]">
        <span className="shrink-0 font-medium whitespace-nowrap">{label}</span>
        {hint && <span className="truncate text-muted-foreground" title={hint}>{hint}</span>}
      </p>
      <div className="space-y-1">{children}</div>
    </div>
  )
}

function Row({ label, children }: { label?: string; children: ReactNode }) {
  return (
    <div className="flex items-start gap-2">
      {label && <span className="w-[5.5rem] shrink-0 pt-0.5 font-mono text-[10.5px] text-muted-foreground">{label}</span>}
      <div className="flex min-w-0 flex-1 flex-wrap gap-1">{children}</div>
    </div>
  )
}

function CodeChip({ text, onClick }: { text: string; onClick?: () => void }) {
  const cls = 'max-w-full truncate rounded-md border px-1.5 py-0.5 font-mono text-[11.5px]'
  return onClick ? (
    <button type="button" onClick={onClick} className={cn(cls, 'hover:bg-accent')} title={text}>{text}</button>
  ) : (
    <span className={cls} title={text}>{text}</span>
  )
}
