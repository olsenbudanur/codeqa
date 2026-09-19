import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, CircleAlert, CircleCheck, Minus, TrendingDown, TrendingUp } from 'lucide-react'
import { navigate } from '@/lib/router'
import { fmtNum, fmtPct, fmtTokens, fmtWhen, workshop, type Iteration, type MetricRow, type RunDetail, type RunRow } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { MetricChart, SERIES, type Series } from '@/components/workshop/charts'
import { BreakdownPanel, EfficiencyPanel, FormatEmergencePanel, GateFunnelPanel, HeldOutPanel, ShapingPanel, SignalDensityPanel, ThroughputPanel, withDerived } from '@/components/workshop/panels'
export { withDerived }
import { Chip, ErrorNote, Loading, Page, Panel, Stat } from '@/components/workshop/ui'

export const REWARD = 'env/all/reward/total'
// Optimizer signals. Thresholds mirror codeqa/evals/monitor.py (KL > 0.05; entropy < 40% of step 0).
// Labels, captions and thresholds for the optimizer keys lane C logs (LOG 2026-09-20 00:50, metrics_patch.py).
const OPTIM: { key: string; label: string; caption?: string; ref?: (rows: MetricRow[]) => { y: number; label: string }[]; domain?: [number | 'auto', number | 'auto'] }[] = [
  { key: 'optim/kl_sample_train_v1', label: 'KL sampler vs trainer (v1)', caption: 'how far one update moved the policy from the weights that produced the batch', ref: () => [{ y: 0.05, label: 'lr too high' }] },
  { key: 'optim/kl_sample_train_v2', label: 'KL sampler vs trainer (v2)', caption: 'second-order estimate of the same' },
  { key: 'optim/post_kl', label: 'KL after the update', caption: 'only with --compute-post-kl' },
  { key: 'optim/entropy', label: 'Policy entropy', caption: 'falls as the policy sharpens; collapse risk below 40% of step 0', ref: (rows) => { const e0 = rows[0]?.['optim/entropy']; return e0 ? [{ y: 0.4 * e0, label: '40% of step 0' }] : [] } },
  { key: 'optim/nll', label: 'NLL of own samples', caption: 'the policy\'s confidence in what it sampled; falls as it sharpens' },
  { key: 'optim/advantage_std', label: 'Advantage std', caption: 'whether groups still disagree; 0 = no gradient' },
  { key: 'optim/frac_tokens_with_advantage', label: 'Tokens with a non-zero advantage', caption: 'share of action tokens that carry any signal', domain: [0, 1] },
  { key: 'optim/importance_ratio_mean', label: 'Importance ratio, mean', caption: 'exp(train logp − sample logp); ~1 in sync training', ref: () => [{ y: 1, label: 'on-policy' }] },
  { key: 'optim/importance_ratio_max', label: 'Importance ratio, max', caption: 'the worst token; spikes mean off-policy updates', ref: () => [{ y: 1, label: 'on-policy' }] },
  { key: 'optim/clip_fraction', label: 'Clip fraction', caption: 'share of tokens with |ratio − 1| > 0.2; rising = the classic PPO warning', ref: () => [{ y: 0.2, label: 'watch' }], domain: [0, 1] },
  // Not logged today: Tinker's optim_step returns no gradient statistics for LoRA runs (OptimStepResponse.metrics is empty
  // across every run on disk) and forward_backward returns log-probs, not gradients. Registered so it is charted with a
  // label and a threshold the moment lane C or Tinker provides it.
  { key: 'optim/grad_norm', label: 'Gradient norm', caption: 'size of the raw gradient before clipping; spikes precede collapse', ref: () => [{ y: 1, label: 'clip' }] },
  { key: 'optim/action_tokens', label: 'Action tokens per step', caption: 'how much generated text the update was computed on' },
  { key: 'optim/lr', label: 'Learning rate' },
]
const LOSS_KEYS = ['optim/loss', 'optim/loss_abs']
const OPTIM_PREFIXES = ['optim/', 'kl_ref/', 'loss/']


export function RunsPage({ name, params }: { name?: string; params: URLSearchParams }) {
  return name ? <RunDetailPage name={name} overlay={(params.get('vs') ?? '').split(',').filter(Boolean)} /> : <RunList />
}

function RunList() {
  const [runs, setRuns] = useState<RunRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    workshop.runs().then(setRuns).catch((e: Error) => setError(e.message))
  }, [])
  return (
    <Page title="Training runs">
      {error && <ErrorNote error={error} />}
      {!runs && !error && <Loading what="runs" />}
      {runs && (
        <div className="overflow-x-auto rounded-lg border bg-background">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted-foreground">
              <tr className="border-b">
                <th className="px-4 py-2 font-normal">Run</th>
                <th className="px-3 py-2 font-normal">Steps</th>
                <th className="px-3 py-2 font-normal">Last reward</th>
                <th className="px-3 py-2 font-normal">lr</th>
                <th className="px-3 py-2 font-normal">Group × batch</th>
                <th className="px-3 py-2 font-normal">Tasks</th>
                <th className="px-3 py-2 font-normal">Updated</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.name} className="cursor-pointer border-b last:border-0 hover:bg-accent/60" onClick={() => navigate(`/workshop/runs/${r.name}`)}>
                  <td className="px-4 py-2.5">
                    <span className="flex items-center gap-2 text-[13.5px]">
                      {r.live && <span className="size-1.5 animate-pulse rounded-full bg-verified" aria-label="Live" />}
                      {r.title}
                    </span>
                    <span className="block font-mono text-[11px] text-muted-foreground">{r.name}{r.hypothesis ? ` · ${r.hypothesis}` : ''}</span>
                  </td>
                  <td className="px-3 py-2.5 font-mono text-[13px] tabular-nums">{r.steps}{r.planned_steps ? <span className="text-muted-foreground"> / {r.planned_steps}</span> : null}</td>
                  <td className="px-3 py-2.5 font-mono text-[13px] tabular-nums">{fmtNum(r.last_reward, 3)}</td>
                  <td className="px-3 py-2.5 font-mono text-[13px]">{r.config.learning_rate?.toExponential(1) ?? '–'}</td>
                  <td className="px-3 py-2.5 font-mono text-[13px]">{r.config.group_size ?? '–'} × {r.config.groups_per_batch ?? '–'}</td>
                  <td className="max-w-[240px] truncate px-3 py-2.5 font-mono text-[12px] text-muted-foreground">{r.config.tasks_path?.replace(/^data\/tasks\//, '') ?? '–'}</td>
                  <td className="px-3 py-2.5 font-mono text-[12px] text-muted-foreground">{fmtWhen(r.updated)}</td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-sm text-muted-foreground">No runs under data/logs yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </Page>
  )
}

function useRun(name: string) {
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let alive = true
    let timer: number | null = null
    const load = () =>
      workshop
        .run(name)
        .then((r) => {
          if (!alive) return
          setRun(r)
          setError(null)
          // Poll while the run is live (metrics.jsonl changed in the last few minutes).
          if (r.live) timer = window.setTimeout(load, 10_000)
        })
        .catch((e: Error) => alive && setError(e.message))
    void load()
    return () => {
      alive = false
      if (timer) window.clearTimeout(timer)
    }
  }, [name])
  return { run, error }
}

export function plateau(rows: MetricRow[]) {
  const r = rows.map((m) => m.reward as number | null).filter((v): v is number => v !== null)
  const last = r.slice(-10)
  const prev = r.slice(-20, -10)
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)
  const gain = last.length && prev.length ? (mean(last) as number) - (mean(prev) as number) : null
  const best = rows.reduce<{ step: number; v: number } | null>((acc, m) => (m.reward !== null && m.reward !== undefined && (!acc || (m.reward as number) > acc.v) ? { step: m.step, v: m.reward as number } : acc), null)
  const lastEval = [...rows].reverse().find((m) => m.eval_reward !== null && m.eval_reward !== undefined)
  return { gain, best, lastEval, lastN: last.length, prevN: prev.length }
}

function RunDetailPage({ name, overlay }: { name: string; overlay: string[] }) {
  const { run, error } = useRun(name)
  const [runs, setRuns] = useState<RunRow[]>([])
  const [others, setOthers] = useState<RunDetail[]>([])
  const [toggles, setToggles] = useState<Record<string, boolean>>({})
  useEffect(() => {
    workshop.runs().then(setRuns).catch(() => {})
  }, [])
  const overlayKey = overlay.join(',')
  useEffect(() => {
    if (overlay.length === 0) {
      setOthers([])
      return
    }
    Promise.all(overlay.map((n) => workshop.run(n).catch(() => null))).then((rs) => setOthers(rs.filter((r): r is RunDetail => !!r)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlayKey])

  const rows = useMemo(() => (run ? withDerived(run) : []), [run])
  const merged = useMemo(() => {
    if (others.length === 0) return rows
    const maps = others.map((o) => new Map(withDerived(o).map((m) => [m.step, m])))
    const steps = new Set([...rows.map((m) => m.step), ...maps.flatMap((m) => [...m.keys()])])
    return [...steps].sort((a, b) => a - b).map((step) => {
      const base: Record<string, unknown> = { ...(rows.find((m) => m.step === step) ?? { step }) }
      others.forEach((_o, i) => {
        base[`o${i}_reward`] = maps[i].get(step)?.reward ?? null
        base[`o${i}_smooth`] = maps[i].get(step)?.reward_smooth ?? null
        base[`o${i}_correct`] = maps[i].get(step)?.correct ?? null
        base[`o${i}_tool_calls`] = maps[i].get(step)?.tool_calls ?? null
        base[`o${i}_prompt_tokens`] = maps[i].get(step)?.prompt_tokens ?? null
      })
      return base as MetricRow
    })
  }, [rows, others])

  const extraKeys = useMemo(() => {
    if (!run?.metrics.length) return [] as string[]
    const keys = new Set<string>()
    for (const m of run.metrics) for (const k of Object.keys(m)) if (/^env\/(?!all\/)[^/]+\/reward\/total$/.test(k)) keys.add(k)
    return [...keys].sort()
  }, [run])

  const p = useMemo(() => plateau(rows), [rows])
  const hours = run?.started && run?.updated ? (run.updated - run.started) / 3600 : null
  const stepsPerHour = hours && hours > 0.1 && run ? run.steps / hours : null   // under 6 min of history the rate is noise (a freshly synced dir has started ≈ updated)

  const rewardSeries: Series[] = [
    { key: 'reward_band', label: '±1 s.e.', color: SERIES[0], kind: 'band', follows: 'reward' },
    { key: 'reward', label: `${name} reward`, color: SERIES[0], kind: 'line', hidden: true },
    { key: 'reward_smooth', label: '5-step mean', color: SERIES[0], kind: 'dashed' },
    { key: 'eval_reward', label: 'held-out (fast)', color: SERIES[2], kind: 'points' },
    ...others.flatMap((o, i) => [
      { key: `o${i}_reward`, label: `${o.name} reward`, color: SERIES[[1, 3, 4, 6, 7][i % 5]], kind: 'line' as const, hidden: true },
      { key: `o${i}_smooth`, label: `${o.name} 5-step mean`, color: SERIES[[1, 3, 4, 6, 7][i % 5]], kind: 'dashed' as const },
    ]),
    ...extraKeys.filter((k) => toggles[k]).map((k, i) => ({ key: k, label: k.split('/')[1], color: SERIES[(5 + i) % SERIES.length], kind: 'dashed' as const })),
  ]

  const OVERLAY_COLORS = [1, 3, 4, 6, 7]
  const correctSeries: Series[] = [
    { key: 'correct', label: `${name} correct`, color: SERIES[1], kind: 'line', hidden: true },
    { key: 'correct_smooth', label: '5-step mean', color: SERIES[1], kind: 'dashed' },
    { key: 'eval_correct', label: 'held-out (fast)', color: SERIES[2], kind: 'points' },
    ...others.map((o, i) => ({ key: `o${i}_correct`, label: `${o.name} correct`, color: SERIES[OVERLAY_COLORS[(i + 1) % 5]], kind: 'line' as const })),
  ]
  const callsSeries: Series[] = [
    { key: 'tool_calls', label: `${name} tool calls`, color: SERIES[3], kind: 'line', hidden: true },
    { key: 'tool_calls_smooth', label: '5-step mean', color: SERIES[3], kind: 'dashed' },
    { key: 'eval_tool_calls', label: 'held-out (fast)', color: SERIES[2], kind: 'points' },
    ...others.map((o, i) => ({ key: `o${i}_tool_calls`, label: `${o.name} tool calls`, color: SERIES[OVERLAY_COLORS[(i + 2) % 5]], kind: 'line' as const })),
  ]

  const promptSeries: Series[] = [
    { key: 'prompt_tokens', label: `${name} prompt tokens`, color: SERIES[4], kind: 'line', hidden: true },
    { key: 'prompt_tokens_smooth', label: '5-step mean', color: SERIES[4], kind: 'dashed' },
    { key: 'eval_prompt_tokens', label: 'held-out (fast)', color: SERIES[2], kind: 'points' },
    ...others.map((o, i) => ({ key: `o${i}_prompt_tokens`, label: `${o.name} prompt tokens`, color: SERIES[OVERLAY_COLORS[(i + 3) % 5]], kind: 'line' as const, hidden: true })),
  ]
  const completionSeries: Series[] = [
    { key: 'completion_tokens', label: `${name} generated`, color: SERIES[6], kind: 'line', hidden: true },
    { key: 'completion_tokens_smooth', label: '5-step mean', color: SERIES[6], kind: 'dashed' },
    { key: 'eval_completion_tokens', label: 'held-out (fast)', color: SERIES[2], kind: 'points' },
  ]

  return (
    <Page
      title={
        <span className="flex flex-col">
          <span className="flex items-center gap-2">
            {run?.title ?? name}
            {run?.live && <Chip tone="good">live</Chip>}
          </span>
          <span className="font-mono text-[11.5px] font-normal text-muted-foreground">{name}{run?.hypothesis ? ` · ${run.hypothesis}` : ''}</span>
        </span>
      }
      wide
      actions={
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-xs text-muted-foreground">Overlay</span>
          {runs.filter((r) => r.name !== name).map((r) => {
            const on = overlay.includes(r.name)
            const next = on ? overlay.filter((x) => x !== r.name) : [...overlay, r.name]
            return (
              <button
                key={r.name}
                type="button"
                aria-pressed={on}
                title={r.title}
                onClick={() => navigate(`/workshop/runs/${name}${next.length ? `?vs=${encodeURIComponent(next.join(','))}` : ''}`)}
                className={cn('rounded-full border px-2 py-0.5 font-mono text-[11px]', on ? 'border-foreground bg-foreground text-background' : 'text-muted-foreground hover:text-foreground')}
              >
                {r.live ? '● ' : ''}{r.name}
              </button>
            )
          })}
        </div>
      }
    >
      {error && <ErrorNote error={error} />}
      {!run && !error && <Loading what="run" />}
      {run && (
        <div className="space-y-4">
          <Panel title="Reward growth" aside={`${run.steps} steps, ${run.config.group_size ?? '?'} × ${run.config.groups_per_batch ?? '?'} episodes per step`}>
            <MetricChart data={merged} series={rewardSeries} height={380} yDomain={[0, 'auto']} />
            <div className="mt-4 flex flex-wrap items-end gap-8 border-t pt-4">
              <Stat
                label={`gain, last ${p.lastN} vs previous ${p.prevN} steps`}
                value={
                  <span className="flex items-center gap-1.5">
                    {p.gain === null ? <Minus className="size-4" /> : p.gain > 0.01 ? <TrendingUp className="size-4" /> : p.gain < -0.01 ? <TrendingDown className="size-4" /> : <Minus className="size-4" />}
                    {p.gain === null ? 'n/a' : `${p.gain > 0 ? '+' : ''}${p.gain.toFixed(3)}`}
                  </span>
                }
                tone={p.gain === null ? undefined : p.gain > 0.01 ? 'good' : p.gain < -0.01 ? 'bad' : undefined}
              />
              <Stat label="best step" value={p.best ? `${fmtNum(p.best.v, 3)} @ ${p.best.step}` : '–'} />
              <Stat label="held-out reward at last eval" value={p.lastEval ? `${fmtNum(p.lastEval.eval_reward as number, 3)} @ ${p.lastEval.step}` : '–'} />
              <Stat label="steps per hour" value={stepsPerHour ? stepsPerHour.toFixed(1) : '–'} />
              {extraKeys.length > 0 && (
                <div className="ml-auto flex flex-wrap gap-1.5">
                  {extraKeys.map((k) => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setToggles((t) => ({ ...t, [k]: !t[k] }))}
                      className={cn('rounded-full border px-2 py-0.5 font-mono text-[11px]', toggles[k] ? 'border-foreground text-foreground' : 'text-muted-foreground')}
                      aria-pressed={!!toggles[k]}
                    >
                      {k.split('/')[1]}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </Panel>
          <div className="grid gap-4">
            <Panel title="Correctness per step" aside="share of episodes judged correct; held-out as points">
              <MetricChart data={merged} series={correctSeries} height={380} yDomain={[0, 1]} yFormat={fmtPct} />
              <div className="mt-3 flex flex-wrap gap-6 border-t pt-3">
                <Stat label="last step" value={rows.at(-1)?.correct !== null && rows.at(-1)?.correct !== undefined ? fmtPct(rows.at(-1)!.correct as number) : '–'} />
                <Stat label="best step" value={(() => { const b = rows.reduce<MetricRow | null>((acc, m) => (m.correct !== null && m.correct !== undefined && (!acc || (m.correct as number) > (acc.correct as number)) ? m : acc), null); return b ? `${fmtPct(b.correct as number)} @ ${b.step}` : '–' })()} />
              </div>
            </Panel>
            <Panel title="Tool calls per step" aside="mean tool calls per episode; held-out as points">
              <MetricChart data={merged} series={callsSeries} height={380} yDomain={[0, 'auto']} />
              <div className="mt-3 flex flex-wrap gap-6 border-t pt-3">
                <Stat label="last step" value={fmtNum(rows.at(-1)?.tool_calls as number, 2)} />
                <Stat label="step 0" value={fmtNum(rows[0]?.tool_calls as number, 2)} />
              </div>
            </Panel>
            <Panel title="Prompt tokens per episode" aside="everything the model read across its turns; the cost driver">
              <MetricChart data={merged} series={promptSeries} height={380} yDomain={[0, 'auto']} yFormat={fmtTokens} />
              <div className="mt-3 flex flex-wrap gap-6 border-t pt-3">
                <Stat label="last step" value={fmtTokens(rows.at(-1)?.prompt_tokens as number)} />
                <Stat label="step 0" value={fmtTokens(rows[0]?.prompt_tokens as number)} />
                <Stat label="final context" value={rows.at(-1)?.context_tokens !== null && rows.at(-1)?.context_tokens !== undefined ? `${fmtTokens(rows.at(-1)!.context_tokens as number)}${run.config.max_context ? ` of ${fmtTokens(run.config.max_context)}` : ''}` : '–'} />
              </div>
            </Panel>
            <Panel title="Generated tokens per episode" aside="thinking, tool calls and the answer">
              <MetricChart data={merged} series={completionSeries} height={380} yDomain={[0, 'auto']} yFormat={fmtTokens} />
              <div className="mt-3 flex flex-wrap gap-6 border-t pt-3">
                <Stat label="last step" value={fmtTokens(rows.at(-1)?.completion_tokens as number)} />
                <Stat label="step 0" value={fmtTokens(rows[0]?.completion_tokens as number)} />
                <Stat label="cap" value={run.config.max_tokens ? fmtTokens(run.config.max_tokens) : '–'} />
              </div>
            </Panel>
          </div>

          <Panel title="Health" aside="same checks as codeqa.evals.monitor">
            {run.warnings.length === 0 ? (
              <p className="flex items-center gap-2 text-sm text-verified">
                <CircleCheck className="size-4" aria-hidden />
                Nothing alarming at step {run.metrics.at(-1)?.step ?? '–'}: group variance, tool diversity, stall rate, format gate, KL and entropy are all within bounds.
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

          <SignalDensityPanel rows={rows} />
          <GateFunnelPanel rows={rows} />
          <FormatEmergencePanel rows={rows} />
          <ShapingPanel rows={rows} />
          <BreakdownPanel rows={rows} />
          <EfficiencyPanel rows={rows} />
          <ThroughputPanel rows={rows} run={run} />
          <HeldOutPanel rows={rows} />
          <OptimizerPanel rows={rows} />

          <RolloutBrowser run={run} />

          <Panel title="Configuration">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 font-mono text-[12.5px] sm:grid-cols-3 lg:grid-cols-4">
              {Object.entries(run.config).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="truncate">{v === null || v === undefined ? '–' : String(v)}</dd>
                </div>
              ))}
            </dl>
            {run.checkpoints.length > 0 && (
              <p className="mt-3 font-mono text-[12px] text-muted-foreground">checkpoints: {run.checkpoints.map((c) => c.name).join(', ')}</p>
            )}
          </Panel>
        </div>
      )}
    </Page>
  )
}

export function OptimizerPanel({ rows, compact }: { rows: MetricRow[]; compact?: boolean }) {
  const present = (k: string) => rows.some((m) => m[k] !== undefined && m[k] !== null)
  const known = OPTIM.filter((o) => present(o.key))
  const extra = [...new Set(rows.flatMap((m) => Object.keys(m)))]
    .filter((k) => OPTIM_PREFIXES.some((p) => k.startsWith(p)) && !OPTIM.some((o) => o.key === k) && !LOSS_KEYS.includes(k) && present(k))
    .sort()
  const hasLoss = LOSS_KEYS.some(present)
  if (known.length + extra.length === 0 && !hasLoss) return null
  return (
    <Panel title="Optimizer" aside="how far each update moved the policy; gradient norms are not exposed by Tinker">
      {hasLoss && (
        <div className="mb-5">
          <p className="mb-1 text-[13px]">Loss</p>
          <p className="mb-2 text-[11.5px] leading-4 text-muted-foreground">
            The importance-sampling policy-gradient surrogate, −mean(ratio × advantage). Advantages are centred within each group, so it sits near zero by design: drift and sign changes are informative, the level is not, and it will not fall like a supervised loss. |loss| is the magnitude of the learning signal; zero means there was nothing to learn from.
          </p>
          <MetricChart
            data={rows}
            height={compact ? 200 : 300}
            refLines={[{ y: 0, label: '' }]}
            series={[
              { key: 'optim/loss', label: 'loss (surrogate)', color: SERIES[1] },
              { key: 'optim/loss_abs', label: '|loss| (signal magnitude)', color: SERIES[0] },
            ]}
          />
        </div>
      )}
      {!hasLoss && rows.length > 0 && (
        <p className="mb-4 rounded-md border border-dashed px-3 py-2 text-[12px] text-muted-foreground">
          No loss for this run: it was launched before the trainer logged one. Runs launched after 2026-09-20 00:50 carry `optim/loss` and `optim/loss_abs`.
        </p>
      )}
      <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-3">
        {known.map((o) => (
          <div key={o.key}>
            <p className="text-[13px]">{o.label}</p>
            {o.caption && <p className="mb-1 truncate text-[11px] text-muted-foreground" title={o.caption}>{o.caption}</p>}
            <MetricChart data={rows} height={compact ? 160 : 220} legend={false} yDomain={o.domain} series={[{ key: o.key, label: o.label, color: SERIES[1] }]} refLines={o.ref?.(rows) ?? []} />
          </div>
        ))}
        {extra.map((k) => (
          <div key={k}>
            <p className="mb-1 font-mono text-[12px]">{k}</p>
            <MetricChart data={rows} height={compact ? 160 : 220} legend={false} series={[{ key: k, label: k, color: SERIES[1] }]} />
          </div>
        ))}
      </div>
    </Panel>
  )
}

function RolloutBrowser({ run }: { run: RunDetail }) {
  const [n, setN] = useState<number | null>(run.iterations.at(-1) ?? null)
  const [it, setIt] = useState<Iteration | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (n === null) return
    setIt(null)
    workshop.iteration(run.name, n).then(setIt).catch((e: Error) => setError(e.message))
  }, [run.name, n])
  if (run.iterations.length === 0) return null
  return (
    <Panel
      title="Rollouts"
      aside={
        <span className="flex items-center gap-2">
          iteration
          <Select value={String(n)} onValueChange={(v) => setN(Number(v))}>
            <SelectTrigger className="h-7 w-[90px]" aria-label="Iteration">
              <SelectValue>{n}</SelectValue>
            </SelectTrigger>
            <SelectContent align="end">
              {run.iterations.map((i) => (
                <SelectItem key={i} value={String(i)}>{i}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </span>
      }
    >
      {error && <ErrorNote error={error} />}
      {!it && !error && <Loading what="rollouts" />}
      {it && (
        <div className="space-y-4">
          {it.groups.map((g) => (
            <div key={g.group_idx}>
              <div className="mb-1 flex items-baseline gap-3 font-mono text-[12px] text-muted-foreground">
                <span>group {g.group_idx}</span>
                <span>{g.trajectories[0]?.tags.join(' / ')}</span>
                <span className="ml-auto">mean reward {fmtNum(g.trajectories.reduce((a, t) => a + (t.reward ?? 0), 0) / Math.max(1, g.trajectories.length), 2)}</span>
              </div>
              <div className="overflow-x-auto rounded-md border">
                <table className="w-full text-[13px]">
                  <tbody>
                    {g.trajectories.map((t) => (
                      <tr key={t.id} className="cursor-pointer border-b last:border-0 hover:bg-accent/60" onClick={() => navigate(`/workshop/traces/${t.id}`)}>
                        <td className={cn('w-14 px-3 py-1.5 font-mono tabular-nums', (t.reward ?? 0) > 0 ? 'text-verified' : 'text-muted-foreground')}>{fmtNum(t.reward, 2)}</td>
                        <td className="px-2 py-1.5">
                          <span className="flex flex-wrap gap-1">
                            {t.tool_sequence.map((s, i) => (
                              <Chip key={i}>{s}</Chip>
                            ))}
                            {t.tool_sequence.length === 0 && <Chip>no tools</Chip>}
                          </span>
                        </td>
                        <td className="px-2 py-1.5 font-mono text-[11.5px] text-muted-foreground">{t.stop ?? '–'}</td>
                        <td className="max-w-[420px] truncate px-2 py-1.5 text-muted-foreground">{t.answer_excerpt}</td>
                        <td className="px-2 py-1.5 text-muted-foreground"><ArrowRight className="size-3.5" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
          <Button variant="ghost" size="sm" onClick={() => navigate(`/workshop/traces?run=${encodeURIComponent(`train:${run.name}`)}`)}>
            All rollouts of this run in Traces
            <ArrowRight />
          </Button>
        </div>
      )}
    </Panel>
  )
}
