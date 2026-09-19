import { useEffect, useMemo, useRef, useState } from 'react'
import { CircleAlert, CircleCheck, Minus, TrendingDown, TrendingUp } from 'lucide-react'
import { toast } from 'sonner'
import { navigate } from '@/lib/router'
import { fmtNum, fmtPct, fmtWhen, workshop, type RunDetail, type RunRow } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { MetricChart, SERIES } from '@/components/workshop/charts'
import { Chip, ErrorNote, Loading, Panel, Stat } from '@/components/workshop/ui'
import { OptimizerPanel, plateau, withDerived } from './runs'
import { GateFunnelPanel, HeldOutPanel, SignalDensityPanel } from '@/components/workshop/panels'

const POLL_MS = 10_000

// One screen to leave open while a run trains. Picks the live run (or the newest), refreshes itself,
// and raises a toast when a new warning appears.
export function LivePage({ requested }: { requested: string | null }) {
  const [runs, setRuns] = useState<RunRow[]>([])
  const [name, setName] = useState<string | null>(requested)
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(() => Date.now())
  const seen = useRef<Set<string>>(new Set())

  // Which run: the requested one, else the live one, else the most recently updated.
  useEffect(() => {
    let alive = true
    const load = () =>
      workshop
        .runs()
        .then((rs) => {
          if (!alive) return
          setRuns(rs)
          if (!requested) {
            const live = rs.find((r) => r.live) ?? rs[0]
            if (live) setName((cur) => (cur && rs.some((r) => r.name === cur && r.live) ? cur : live.name))
          }
        })
        .catch((e: Error) => alive && setError(e.message))
    void load()
    const t = window.setInterval(load, 30_000)
    return () => {
      alive = false
      window.clearInterval(t)
    }
  }, [requested])

  // The run itself, every 10 s.
  useEffect(() => {
    if (!name) return
    let alive = true
    seen.current = new Set()
    const load = () =>
      workshop
        .run(name)
        .then((r) => {
          if (!alive) return
          setRun(r)
          setError(null)
          setTick(Date.now())
          for (const w of r.warnings) {
            if (!seen.current.has(w)) {
              if (seen.current.size > 0 || r.live) toast.warning(w, { duration: 12_000 })
              seen.current.add(w)
            }
          }
        })
        .catch((e: Error) => alive && setError(e.message))
    void load()
    const t = window.setInterval(load, POLL_MS)
    return () => {
      alive = false
      window.clearInterval(t)
    }
  }, [name])

  const rows = useMemo(() => (run ? withDerived(run) : []), [run])
  const p = useMemo(() => plateau(rows), [rows])
  const last = rows.at(-1)
  const prev = rows.at(-2)

  useEffect(() => {
    if (!run || !last) return
    document.title = `${run.live ? '● ' : ''}${run.name} · step ${last.step} · r ${fmtNum(last.reward as number, 3)}`
    return () => {
      document.title = 'Code Q&A'
    }
  }, [run, last])

  const hours = run?.started && run?.updated ? (run.updated - run.started) / 3600 : null
  const stepsPerHour = hours && hours > 0 && run ? run.steps / hours : null

  const delta = (k: string) => {
    if (!last || !prev) return null
    const a = last[k]
    const b = prev[k]
    return a === null || a === undefined || b === null || b === undefined ? null : (a as number) - (b as number)
  }
  const numbers: { label: string; key: string; fmt: (v: number) => string; upIsGood: boolean }[] = [
    { label: 'reward', key: 'reward', fmt: (v) => fmtNum(v, 3), upIsGood: true },
    { label: 'format ok', key: 'env/all/format_ok', fmt: fmtPct, upIsGood: true },
    { label: 'citations grounded', key: 'env/all/citations_grounded', fmt: fmtPct, upIsGood: true },
    { label: 'correctness', key: 'env/all/correctness', fmt: fmtPct, upIsGood: true },
    { label: 'tool calls', key: 'env/all/tool_calls', fmt: (v) => fmtNum(v, 1), upIsGood: false },
    { label: 'stopped at budget', key: 'env/all/stop_budget', fmt: fmtPct, upIsGood: false },
    { label: 'group reward std', key: 'env/all/group_reward_std', fmt: (v) => fmtNum(v, 3), upIsGood: true },
    { label: 'KL sampler vs trainer', key: 'optim/kl_sample_train_v1', fmt: (v) => v.toExponential(1), upIsGood: false },
    { label: 'entropy', key: 'optim/entropy', fmt: (v) => fmtNum(v, 3), upIsGood: true },
  ]

  return (
    <div className="mx-auto w-full max-w-[1500px] px-4 py-4 sm:px-6">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="flex items-center gap-2 text-[20px] font-medium tracking-tight">
          Live
          {run?.live ? <Chip tone="good">training</Chip> : run ? <Chip>idle since {fmtWhen(run.updated)}</Chip> : null}
        </h1>
        <Select value={name ?? ''} onValueChange={(v) => { setName(v); navigate(`/workshop/live?run=${encodeURIComponent(v)}`) }}>
          <SelectTrigger className="h-8 w-[200px]" aria-label="Run">
            <SelectValue>{name ?? 'pick a run'}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {runs.map((r) => (
              <SelectItem key={r.name} value={r.name}>
                {r.live ? '● ' : ''}{r.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="ml-auto font-mono text-[11.5px] text-muted-foreground">
          refreshed {new Date(tick).toLocaleTimeString()}, every {POLL_MS / 1000} s
        </span>
      </div>

      {error && <ErrorNote error={error} />}
      {!run && !error && <Loading what="run" />}
      {run && last && (
        <div className="grid gap-3 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <Panel title="Reward" aside={`step ${last.step} of ${run.steps}, ${run.config.group_size ?? '?'} × ${run.config.groups_per_batch ?? '?'} per step`}>
            <MetricChart
              data={rows}
              height={260}
              yDomain={[0, 'auto']}
              series={[
                { key: 'reward_band', label: '±1 s.e.', color: SERIES[0], kind: 'band' },
                { key: 'reward', label: 'reward', color: SERIES[0] },
                { key: 'reward_smooth', label: '5-step mean', color: SERIES[0], kind: 'dashed' },
                { key: 'eval_reward', label: 'held-out', color: SERIES[2], kind: 'points' },
              ]}
            />
            <div className="mt-3 flex flex-wrap gap-6 border-t pt-3">
              <Stat
                label={`gain, last ${p.lastN} vs previous ${p.prevN}`}
                value={
                  <span className="flex items-center gap-1.5">
                    {p.gain === null ? <Minus className="size-4" /> : p.gain > 0.01 ? <TrendingUp className="size-4" /> : p.gain < -0.01 ? <TrendingDown className="size-4" /> : <Minus className="size-4" />}
                    {p.gain === null ? 'n/a' : `${p.gain > 0 ? '+' : ''}${p.gain.toFixed(3)}`}
                  </span>
                }
                tone={p.gain === null ? undefined : p.gain > 0.01 ? 'good' : p.gain < -0.01 ? 'bad' : undefined}
              />
              <Stat label="best step" value={p.best ? `${fmtNum(p.best.v, 3)} @ ${p.best.step}` : '–'} />
              <Stat label="held-out at last eval" value={p.lastEval ? `${fmtNum(p.lastEval.eval_reward as number, 3)} @ ${p.lastEval.step}` : '–'} />
              <Stat label="steps per hour" value={stepsPerHour ? stepsPerHour.toFixed(1) : '–'} />
            </div>
          </Panel>

          <div className="grid gap-3">
            <Panel title="Health" aside={run.warnings.length ? `${run.warnings.length} warning${run.warnings.length > 1 ? 's' : ''}` : 'clear'}>
              {run.warnings.length === 0 ? (
                <p className="flex items-center gap-2 text-sm text-verified">
                  <CircleCheck className="size-4" aria-hidden />
                  Nothing alarming at step {last.step}.
                </p>
              ) : (
                <ul className="space-y-1.5">
                  {run.warnings.map((w) => (
                    <li key={w} className="flex items-start gap-2 text-sm">
                      <CircleAlert className="mt-0.5 size-4 shrink-0 text-unverified" aria-hidden />
                      <span>{w}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
            <Panel title="Last step" aside={prev ? `change vs step ${prev.step}` : ''}>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
                {numbers.map((n) => {
                  const v = last[n.key]
                  if (v === null || v === undefined) return null
                  const d = delta(n.key)
                  const good = d === null || Math.abs(d) < 1e-9 ? null : n.upIsGood ? d > 0 : d < 0
                  return (
                    <div key={n.key}>
                      <dd className="font-mono text-[17px] leading-none tabular-nums">{n.fmt(v as number)}</dd>
                      <dt className="mt-1 text-[11px] text-muted-foreground">
                        {n.label}
                        {d !== null && Math.abs(d) >= 1e-9 && (
                          <span className={cn('ml-1.5 font-mono', good ? 'text-verified' : 'text-unverified')}>
                            {d > 0 ? '+' : '−'}{n.fmt(Math.abs(d))}
                          </span>
                        )}
                      </dt>
                    </div>
                  )
                })}
              </dl>
            </Panel>
          </div>

          <div className="grid gap-3 xl:col-span-2 xl:grid-cols-3">
            <SignalDensityPanel rows={rows} compact />
            <GateFunnelPanel rows={rows} compact />
            <HeldOutPanel rows={rows} compact />
          </div>
          <div className="xl:col-span-2">
            <OptimizerPanel rows={rows} compact />
          </div>
        </div>
      )}
    </div>
  )
}
