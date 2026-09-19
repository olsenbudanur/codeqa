import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { ArrowLeft, ExternalLink, Play, Search } from 'lucide-react'
import { navigate } from '@/lib/router'
import { repoName } from '@/lib/repo'
import type { Span } from '@/lib/contracts'
import { formatRange } from '@/lib/citations'
import { workshop, type RepoOverview, type ToolRun, type ToolSpec } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { FileViewer } from '@/components/file/file-viewer'
import { Dots } from '@/components/working'
import { Chip, ErrorNote, Loading, Panel, Stat } from '@/components/workshop/ui'

// One repository as the agent sees it: the map, the files, and a console for its five tools.
export function RepoPage({ repoId, params }: { repoId: string; params: URLSearchParams }) {
  const [ov, setOv] = useState<RepoOverview | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [span, setSpan] = useState<Span | null>(null)
  useEffect(() => {
    setOv(null)
    workshop.repoOverview(repoId).then(setOv).catch((e: Error) => setError(e.message))
  }, [repoId])
  const tab = params.get('tab') ?? 'overview'

  return (
    <div className="mx-auto w-full max-w-[1400px] px-4 py-6 sm:px-6">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => navigate('/workshop/data')}>
          <ArrowLeft />
          Data
        </Button>
        <h1 className="font-mono text-[18px] font-medium">{repoName({ repo_id: repoId })}</h1>
        {ov?.sha && <Chip>{ov.sha.slice(0, 7)}</Chip>}
        {ov?.nodoc && <Chip tone="warn">docstrings stripped</Chip>}
        {ov?.url && (
          <a href={ov.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
            GitHub <ExternalLink className="size-3" />
          </a>
        )}
        <Button size="sm" className="ml-auto" onClick={() => navigate('/app')}>Ask about it</Button>
      </div>
      {error && <ErrorNote error={error} />}
      {!ov && !error && <Loading what="repository" />}
      {ov && (
        <Tabs value={tab} onValueChange={(v) => navigate(`/workshop/repos/${encodeURIComponent(repoId)}?tab=${v}`)}>
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="files">Files ({ov.n_files.toLocaleString()})</TabsTrigger>
            <TabsTrigger value="tools">Tool console</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="mt-4 space-y-4">
            <div className="flex flex-wrap gap-8 rounded-lg border bg-background px-4 py-4">
              <Stat label="files" value={ov.n_files.toLocaleString()} />
              <Stat label="lines" value={ov.lines.toLocaleString()} />
              <Stat label="symbols indexed" value={ov.symbols?.toLocaleString() ?? '–'} />
              <Stat label="directory summaries" value={String(ov.n_summaries)} />
              <Stat label="map lines" value={String(ov.map_lines)} />
            </div>
            <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
              <Panel title="Repository map" aside="the first thing the agent reads, verbatim">
                <pre className="max-h-[560px] overflow-auto font-mono text-[12px] leading-5 whitespace-pre">{ov.map || '(no map.txt)'}</pre>
              </Panel>
              <div className="space-y-4">
                <Panel title="Languages">
                  <Bars data={ov.languages} />
                </Panel>
                <Panel title="Top-level directories">
                  <Bars data={ov.top_dirs} />
                </Panel>
                {Object.keys(ov.dropped).length > 0 && (
                  <Panel title="Dropped at snapshot" aside="binaries, vendored and generated files">
                    <dl className="grid grid-cols-2 gap-y-1 font-mono text-[12px]">
                      {Object.entries(ov.dropped).map(([k, v]) => (
                        <div key={k} className="contents">
                          <dt className="text-muted-foreground">{k}</dt>
                          <dd className="text-right tabular-nums">{v.toLocaleString()}</dd>
                        </div>
                      ))}
                    </dl>
                  </Panel>
                )}
              </div>
            </div>
          </TabsContent>
          <TabsContent value="files" className="mt-4">
            <FileList ov={ov} onOpen={(p) => setSpan({ path: p, start: 1, end: 1 })} />
          </TabsContent>
          <TabsContent value="tools" className="mt-4">
            <ToolConsole repoId={repoId} onOpen={setSpan} />
          </TabsContent>
        </Tabs>
      )}
      <Sheet open={!!span} onOpenChange={(o) => !o && setSpan(null)}>
        <SheetContent side="right" className="w-full p-0 sm:max-w-[680px]" showCloseButton={false}>
          <SheetTitle className="sr-only">File</SheetTitle>
          {span && <FileViewer repoId={repoId} repoUrl={ov?.url ?? undefined} sha={ov?.sha ?? undefined} span={span} onClose={() => setSpan(null)} />}
        </SheetContent>
      </Sheet>
    </div>
  )
}

function Bars({ data }: { data: Record<string, number> }) {
  const max = Math.max(1, ...Object.values(data))
  return (
    <ul className="space-y-1">
      {Object.entries(data).map(([k, v]) => (
        <li key={k} className="flex items-center gap-2 text-[12px]">
          <span className="w-28 truncate font-mono text-muted-foreground" title={k}>{k}</span>
          <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
            <span className="block h-full rounded-full" style={{ width: `${(v / max) * 100}%`, background: 'var(--series-1)' }} />
          </span>
          <span className="w-12 text-right font-mono tabular-nums">{v.toLocaleString()}</span>
        </li>
      ))}
    </ul>
  )
}

function FileList({ ov, onOpen }: { ov: RepoOverview; onOpen: (path: string) => void }) {
  const [q, setQ] = useState('')
  const files = useMemo(() => {
    const ql = q.toLowerCase()
    return ov.files.filter((f) => !ql || f.path.toLowerCase().includes(ql)).slice(0, 500)
  }, [ov.files, q])
  return (
    <Panel title="Files" aside={`${files.length.toLocaleString()} shown${ov.files.length > 500 && files.length === 500 ? ', narrow the search for more' : ''}`}>
      <div className="mb-3 flex items-center gap-2">
        <Search className="size-4 text-muted-foreground" aria-hidden />
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter paths" className="h-8 max-w-[360px]" />
      </div>
      <ul className="max-h-[600px] divide-y overflow-auto rounded-md border">
        {files.map((f) => (
          <li key={f.path}>
            <button type="button" onClick={() => onOpen(f.path)} className="flex w-full items-center gap-3 px-3 py-1.5 text-left font-mono text-[12.5px] hover:bg-accent">
              <span className="min-w-0 flex-1 truncate">{f.path}</span>
              {f.lang && <span className="text-[11px] text-muted-foreground">{f.lang}</span>}
              <span className="w-14 text-right text-[11px] text-muted-foreground tabular-nums">{f.lines.toLocaleString()} L</span>
            </button>
          </li>
        ))}
      </ul>
    </Panel>
  )
}

// Call the agent's tools by hand and see exactly what it would see (same caps, same error text).
function ToolConsole({ repoId, onOpen }: { repoId: string; onOpen: (s: Span) => void }) {
  const [specs, setSpecs] = useState<ToolSpec[]>([])
  const [caps, setCaps] = useState<Record<string, number>>({})
  const [tool, setTool] = useState('find_symbol')
  const [args, setArgs] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [runs, setRuns] = useState<ToolRun[]>([])
  const [error, setError] = useState<string | null>(null)
  // Which agent's tool set: the default (lean) agent, or e.g. bash_v3, whose only tool is the read-only shell.
  const [variant, setVariant] = useState<string>('default')
  const [variants, setVariants] = useState<string[]>([])
  useEffect(() => {
    workshop.repoTools(repoId, variant).then((r) => {
      setSpecs(r.tools)
      setCaps(r.caps)
      setVariants(r.variants ?? [])
      if (!r.tools.some((t) => t.name === tool)) { setTool(r.tools[0]?.name ?? ''); setArgs({}) }
    }).catch((e: Error) => setError(e.message))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoId, variant])
  const spec = specs.find((s) => s.name === tool)

  const fieldType = (p: ToolSpec['parameters']['properties'][string]) => p.type ?? p.anyOf?.find((x) => x.type !== 'null')?.type ?? 'string'

  async function run(e?: FormEvent) {
    e?.preventDefault()
    if (!spec) return
    const built: Record<string, unknown> = {}
    for (const [k, p] of Object.entries(spec.parameters.properties)) {
      const raw = (args[k] ?? '').trim()
      if (raw === '') continue
      built[k] = fieldType(p) === 'integer' || fieldType(p) === 'number' ? Number(raw) : raw
    }
    setBusy(true)
    setError(null)
    try {
      const r = await workshop.runTool(repoId, tool, built, variant)
      setRuns((rs) => [r, ...rs].slice(0, 30))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
      <Panel title="Call a tool" aside="as the agent would">
        {variants.length > 1 && (
          <label className="mb-3 block text-[12px]">
            <span className="font-mono">agent</span>
            <select value={variant} onChange={(e) => setVariant(e.target.value)} className="mt-1 h-8 w-full rounded-md border bg-background px-2 font-mono text-[12.5px]">
              {variants.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        )}
        <div className="mb-3 flex flex-wrap gap-1.5">
          {specs.map((s) => (
            <button key={s.name} type="button" onClick={() => { setTool(s.name); setArgs({}) }} className={cn('rounded-full border px-2.5 py-1 font-mono text-[12px]', tool === s.name ? 'border-foreground bg-foreground text-background' : 'text-muted-foreground hover:text-foreground')}>
              {s.name}
            </button>
          ))}
        </div>
        {spec && (
          <form onSubmit={run} className="space-y-3">
            <p className="text-[12.5px] leading-5 text-muted-foreground">{spec.description}</p>
            {Object.entries(spec.parameters.properties).map(([k, p]) => (
              <label key={k} className="block text-[12px]">
                <span className="font-mono">{k}</span>
                {spec.parameters.required?.includes(k) && <span className="ml-1 text-muted-foreground">required</span>}
                <Input
                  value={args[k] ?? ''}
                  onChange={(e) => setArgs((a) => ({ ...a, [k]: e.target.value }))}
                  placeholder={p.description ?? ''}
                  inputMode={fieldType(p) === 'integer' ? 'numeric' : undefined}
                  className="mt-1 h-8 font-mono text-[12.5px]"
                />
              </label>
            ))}
            <Button type="submit" disabled={busy} className="w-full">
              {busy ? <Dots className="text-background" /> : <Play />}
              Run {tool}
            </Button>
          </form>
        )}
        {error && <div className="mt-3"><ErrorNote error={error} /></div>}
        {Object.keys(caps).length > 0 && (
          <details className="mt-4 text-[11.5px] text-muted-foreground">
            <summary className="cursor-pointer">Output caps the agent lives with</summary>
            <dl className="mt-1 grid grid-cols-2 gap-y-0.5 font-mono">
              {Object.entries(caps).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt>{k}</dt>
                  <dd className="text-right tabular-nums">{v}</dd>
                </div>
              ))}
            </dl>
          </details>
        )}
      </Panel>
      <div className="space-y-3">
        {runs.length === 0 && <p className="rounded-lg border bg-background px-4 py-6 text-sm text-muted-foreground">Outputs appear here, newest first, exactly as the agent receives them.</p>}
        {runs.map((r, i) => (
          <Panel
            key={i}
            title={
              <span className="font-mono text-[12.5px]">
                {r.name}({Object.entries(r.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})
              </span>
            }
            aside={`${r.chars.toLocaleString()} chars, ${r.seconds < 0.01 ? '<0.01' : r.seconds.toFixed(2)} s${r.error ? ', error' : ''}`}
          >
            <pre className={cn('max-h-[420px] overflow-auto font-mono text-[12px] leading-5 whitespace-pre-wrap', r.error && 'text-destructive')}>{r.output}</pre>
            {r.files_read.length > 0 && (
              <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[11.5px] text-muted-foreground">
                counts as read:
                {r.files_read.map((s, k) => (
                  <button key={k} type="button" onClick={() => onOpen(s)} className="rounded-md border px-1.5 py-0.5 font-mono text-verified hover:bg-accent">
                    {s.path} {formatRange(s)}
                  </button>
                ))}
              </p>
            )}
          </Panel>
        ))}
      </div>
    </div>
  )
}
