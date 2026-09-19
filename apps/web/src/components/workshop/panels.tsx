// Training panels in lane C's order (LOG 2026-09-19 17:30). Each takes derived metric rows and renders nothing
// when its keys are absent, so smoke runs and run one share one page.
import { fmtNum, fmtPct, type MetricRow, type RunDetail } from '@/lib/workshop'
import { MetricChart, SERIES, StackedBars, rollingMean, type Series } from './charts'
import { Panel, Stat } from './ui'

const E = (k: string) => `env/all/${k}`
const EV = (k: string) => `eval/fast/env/all/${k}`
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const has = (rows: MetricRow[], k: string) => rows.some((m) => num(m[k]) !== null)

export function withDerived(run: RunDetail): MetricRow[] {
  const cfg = run.config
  const n = (cfg.groups_per_batch ?? 0) * (cfg.group_size ?? 0)
  const rewards = run.metrics.map((m) => num(m[E('reward/total')]) ?? num(m[E('reward')]))
  const smooth = rollingMean(rewards, 5)
  const per = (a: number | null, b: number | null) => (a !== null && b !== null && b > 0 ? a / b : null)
  return run.metrics.map((m, i) => {
    const std = num(m[E('group_reward_std')])
    const se = std !== null && n > 0 ? std / Math.sqrt(n) : null
    const r = rewards[i]
    const gates = ['format', 'citations', 'grounding', 'budget', 'judge_error'].map((g) => num(m[E(`gate_${g}`)]) ?? 0)
    const correct = num(m[E('correct')])
    const evCorrect = num(m[EV('correct')])
    const sample = num(m['time/policy_sample:total'])
    const grade = num(m['time/compute_group_rewards:total'])
    return {
      ...m,
      reward: r,
      reward_smooth: smooth[i],
      reward_band: r !== null && se !== null ? ([r - se, r + se] as unknown as number) : null,
      eval_reward: num(m[EV('reward/total')]) ?? num(m[EV('reward')]),
      stalled: num(m[E('stalled')]) ?? ((num(m[E('stop_budget')]) ?? 0) + (num(m[E('stop_max_turns')]) ?? 0)),
      // the "format" gate splits by stop reason: ran out of tool calls, ran out of turns, context overflow, bad tool syntax,
      // and whatever is left (pasted tool output / driver errors)
      gate_out_of_calls: num(m[E('gate_format')]) !== null ? (num(m[E('stop_budget')]) ?? 0) : null,
      gate_out_of_turns: num(m[E('gate_format')]) !== null ? (num(m[E('stop_max_turns')]) ?? 0) : null,
      gate_overflow: num(m[E('gate_format')]) !== null ? (num(m[E('stop_overflow')]) ?? 0) : null,
      gate_bad_syntax: num(m[E('gate_format')]) !== null ? (num(m[E('stop_parse_error')]) ?? 0) : null,
      gate_other_format: num(m[E('gate_format')]) !== null
        ? Math.max(0, (num(m[E('gate_format')]) ?? 0) - (num(m[E('stop_budget')]) ?? 0) - (num(m[E('stop_max_turns')]) ?? 0)
            - (num(m[E('stop_overflow')]) ?? 0) - (num(m[E('stop_parse_error')]) ?? 0))
        : null,
      reached_grading: has([m], E('gate_format')) ? Math.max(0, 1 - gates.reduce((a, b) => a + b, 0)) : null,
      tool_calls_per_correct: per(num(m[E('tool_calls')]), correct),
      prompt_tokens_per_correct: per(num(m[E('prompt_tokens')]), correct),
      eval_tool_calls_per_correct: per(num(m[EV('tool_calls')]), evCorrect),
      sampling_share: sample !== null && grade !== null && sample + grade > 0 ? sample / (sample + grade) : null,
    }
  })
}

// 1. Signal density: what share of groups carry a gradient at all.
export function SignalDensityPanel({ rows, compact }: { rows: MetricRow[]; compact?: boolean }) {
  if (!has(rows, E('by_group/frac_mixed'))) return null
  const h = compact ? 180 : 300
  return (
    <Panel title="Signal density" aside="collapse early warning: groups with mixed reward carry the gradient">
      <div className={compact ? '' : 'grid gap-4 md:grid-cols-2'}>
        <div>
          {!compact && <p className="mb-1 text-[13px]">Groups by reward pattern</p>}
          <MetricChart
            data={rows}
            height={h}
            yDomain={[0, 1]}
            series={[
              { key: E('by_group/frac_mixed'), label: 'mixed (learning signal)', color: SERIES[2], kind: 'area' },
              { key: E('by_group/frac_all_bad'), label: 'all zero', color: SERIES[1], kind: 'area' },
              { key: E('by_group/frac_all_good'), label: 'all good', color: SERIES[0], kind: 'area' },
            ]}
          />
        </div>
        {!compact && (
          <div>
            <p className="mb-1 text-[13px]">Behaviour diversity</p>
            <MetricChart
              data={rows}
              height={h}
              yDomain={[0, 'auto']}
              refLines={[{ y: 1.5, label: 'collapsed' }]}
              series={[
                { key: E('unique_tool_sequences_per_group'), label: 'unique tool sequences per group', color: SERIES[0] },
                { key: E('group_reward_std'), label: 'reward std within group', color: SERIES[1], kind: 'dashed' },
              ]}
            />
          </div>
        )}
      </div>
    </Panel>
  )
}

// 2. Gate funnel: where episodes drop before correctness is even scored.
export function GateFunnelPanel({ rows, compact }: { rows: MetricRow[]; compact?: boolean }) {
  if (!has(rows, E('gate_format'))) return null
  return (
    <Panel title="Why episodes scored zero" aside="share of episodes by how they ended before correctness was scored">
      <StackedBars
        data={rows}
        height={compact ? 280 : 460}
        series={[
          { key: 'gate_out_of_calls', label: 'out of tool calls', color: SERIES[1] },
          { key: 'gate_out_of_turns', label: 'out of turns', color: SERIES[5] },
          { key: 'gate_overflow', label: 'context overflow', color: SERIES[0] },
          { key: 'gate_bad_syntax', label: 'bad tool syntax', color: SERIES[6] },
          { key: 'gate_other_format', label: 'pasted output', color: SERIES[7] },
          { key: E('gate_citations'), label: 'no citations', color: SERIES[3] },
          { key: E('gate_grounding'), label: 'unread citations', color: SERIES[4] },
          { key: E('gate_judge_error'), label: 'judge down', color: SERIES[5] },
          { key: 'reached_grading', label: 'reached grading', color: SERIES[2] },
        ]}
      />
    </Panel>
  )
}

// 3. Format emergence: does the model write bracket citations, and are they real.
export function FormatEmergencePanel({ rows, compact }: { rows: MetricRow[]; compact?: boolean }) {
  if (!has(rows, E('citations_parse'))) return null
  return (
    <Panel title="Format emergence" aside="citations_parse is the wall the step-0 probe found">
      <MetricChart
        data={rows}
        height={compact ? 200 : 320}
        yDomain={[0, 1]}
        series={[
          { key: E('citations_parse'), label: 'citations present', color: SERIES[0] },
          { key: E('format_ok'), label: 'format ok', color: SERIES[1] },
          { key: E('citations_exist'), label: 'citations exist', color: SERIES[2] },
          { key: E('citations_grounded'), label: 'citations grounded', color: SERIES[3] },
          { key: E('identifier_grounded'), label: 'identifier grounded', color: SERIES[6], kind: 'dashed' },
        ]}
      />
    </Panel>
  )
}

// 4. Shaped vs unshaped reward, plus the shaping rates and stalls.
export function ShapingPanel({ rows }: { rows: MetricRow[] }) {
  const shaped = has(rows, E('reward_shaped'))
  const rates: Series[] = [
    { key: 'stalled', label: 'stalled (budget + max turns)', color: SERIES[1] },
    ...(has(rows, E('no_answer_penalty')) ? [{ key: E('no_answer_penalty'), label: 'no-answer penalty applied', color: SERIES[3] }] : []),
    ...(has(rows, E('grounded_credit')) ? [{ key: E('grounded_credit'), label: 'grounded credit applied', color: SERIES[2] }] : []),
    { key: E('stop_answer'), label: 'answered', color: SERIES[0], kind: 'dashed' },
  ]
  return (
    <Panel title="Reward shaping and stalls" aside={shaped ? 'shaped is what trains; unshaped is what we report' : 'no shaping keys in this run (pre run-one)'}>
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-[13px]">Reward</p>
          <MetricChart
            data={rows}
            height={280}
            series={[
              { key: 'reward', label: 'unshaped (reported)', color: SERIES[0] },
              ...(shaped ? [{ key: E('reward_shaped'), label: 'shaped (trains)', color: SERIES[1] }] : []),
            ]}
          />
        </div>
        <div>
          <p className="mb-1 text-[13px]">Rates</p>
          <MetricChart data={rows} height={280} yDomain={[0, 1]} series={rates} />
        </div>
      </div>
    </Panel>
  )
}

// 5. Per-source and per-type reward.
export function BreakdownPanel({ rows }: { rows: MetricRow[] }) {
  const keys = [...new Set(rows.flatMap((m) => Object.keys(m)))]
  const bySrc = ['codescout', 'deepcodebench', 'structural', 'teacher', 'sweqa'].filter((s) => keys.includes(`env/${s}/reward`))
  const byType = ['locate', 'value', 'enumerate', 'trace', 'explain'].filter((t) => keys.includes(`env/${t}/reward`))
  if (bySrc.length + byType.length === 0) return null
  return (
    <Panel title="Reward by source and task type" aside="which data the policy is learning on">
      <div className="grid gap-4 md:grid-cols-2">
        {bySrc.length > 0 && (
          <div>
            <p className="mb-1 text-[13px]">By source</p>
            <MetricChart data={rows} height={280} yDomain={[0, 'auto']} series={bySrc.map((s, i) => ({ key: `env/${s}/reward`, label: s, color: SERIES[i] }))} />
          </div>
        )}
        {byType.length > 0 && (
          <div>
            <p className="mb-1 text-[13px]">By task type</p>
            <MetricChart data={rows} height={280} yDomain={[0, 'auto']} series={byType.map((t, i) => ({ key: `env/${t}/reward`, label: t, color: SERIES[i] }))} />
          </div>
        )}
      </div>
    </Panel>
  )
}

// 6. Efficiency: the run-two headline, with the per-correct ratios.
export function EfficiencyPanel({ rows }: { rows: MetricRow[] }) {
  const items: { key: string; label: string; ev?: string }[] = [
    { key: E('tool_calls'), label: 'tool calls', ev: EV('tool_calls') },
    { key: E('prompt_tokens'), label: 'prompt tokens (cumulative per episode)' },
    { key: E('answer_tokens'), label: 'answer tokens' },
    { key: E('turns'), label: 'turns' },
    { key: E('tool_errors'), label: 'tool errors' },
    { key: E('redundant_reads'), label: 'redundant reads' },
    { key: 'tool_calls_per_correct', label: 'tool calls per correct answer', ev: 'eval_tool_calls_per_correct' },
    { key: 'prompt_tokens_per_correct', label: 'prompt tokens per correct answer' },
  ].filter((i) => has(rows, i.key))
  if (items.length === 0) return null
  return (
    <Panel title="Efficiency" aside={has(rows, 'tool_calls_per_correct') ? 'per-correct ratios use the correct key, not mean reward' : 'per-correct ratios appear once the run logs correct'}>
      <div className="grid gap-5 md:grid-cols-2 2xl:grid-cols-3">
        {items.map((it) => (
          <div key={it.key}>
            <p className="mb-1 text-[13px]">{it.label}</p>
            <MetricChart
              data={rows}
              height={220}
              legend={false}
              yDomain={[0, 'auto']}
              series={[{ key: it.key, label: 'train', color: SERIES[0] }, ...(it.ev && has(rows, it.ev) ? [{ key: it.ev, label: 'held-out', color: SERIES[2], kind: 'points' as const }] : [])]}
            />
          </div>
        ))}
      </div>
    </Panel>
  )
}

// 7. Judge health and throughput.
export function ThroughputPanel({ rows, run }: { rows: MetricRow[]; run: RunDetail }) {
  const last = rows.at(-1)
  if (!last) return null
  const totals = rows.map((m) => num(m['time/total'])).filter((v): v is number => v !== null)
  const meanStep = totals.length ? totals.reduce((a, b) => a + b, 0) / totals.length : null
  const done = num(last['progress/done_frac'])
  const remainingSteps = done && done > 0 ? Math.round((rows.length * (1 - done)) / done) : null
  const eta = meanStep !== null && remainingSteps !== null ? remainingSteps * meanStep : null
  const stepsPerHour = meanStep ? 3600 / meanStep : null
  const fmtDur = (s: number) => (s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`)
  return (
    <Panel title="Judge health and throughput" aside={run.live ? 'live' : 'finished'}>
      <div className="mb-4 flex flex-wrap gap-6">
        <Stat label="progress" value={done !== null ? fmtPct(done) : '–'} />
        <Stat label="seconds per step (mean)" value={meanStep !== null ? fmtNum(meanStep, 0) : '–'} />
        <Stat label="steps per hour" value={stepsPerHour !== null ? stepsPerHour.toFixed(1) : '–'} />
        <Stat label="remaining, at this pace" value={eta !== null ? `${remainingSteps} steps, ${fmtDur(eta)}` : '–'} />
        <Stat label="judge error rate, last step" value={fmtPct(num(last[E('judge_error_rate')]))} tone={(num(last[E('judge_error_rate')]) ?? 0) > 0.1 ? 'bad' : undefined} />
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {has(rows, E('judge_error_rate')) && (
          <div>
            <p className="mb-1 text-[13px]">Judge errors</p>
            <MetricChart
              data={rows}
              height={240}
              yDomain={[0, 1]}
              series={[
                { key: E('judge_error_rate'), label: 'samples with a judge error', color: SERIES[1] },
                { key: E('group_all_judge_errors'), label: 'groups where every sample errored', color: SERIES[7], kind: 'dashed' },
              ]}
            />
          </div>
        )}
        {has(rows, 'time/policy_sample:total') && (
          <div>
            <p className="mb-1 text-[13px]">Time per step</p>
            <MetricChart
              data={rows}
              height={240}
              yDomain={[0, 'auto']}
              series={[
                { key: 'time/policy_sample:total', label: 'sampling', color: SERIES[0] },
                { key: 'time/compute_group_rewards:total', label: 'grading', color: SERIES[1] },
                { key: 'time/train_step', label: 'train step', color: SERIES[2] },
                { key: 'time/total', label: 'total', color: SERIES[3], kind: 'dashed' },
              ]}
            />
          </div>
        )}
      </div>
    </Panel>
  )
}

// 8. Held-out block: the talk's four numbers at the latest eval, and their curves.
export function HeldOutPanel({ rows, compact }: { rows: MetricRow[]; compact?: boolean }) {
  const evalRows = rows.filter((m) => num(m['eval_reward']) !== null)
  if (evalRows.length === 0) return null
  const last = evalRows.at(-1)!
  const first = evalRows[0]
  const d = (k: string) => {
    const a = num(last[k])
    const b = num(first[k])
    return a !== null && b !== null && evalRows.length > 1 ? a - b : null
  }
  const tiles: { label: string; key: string; fmt: (v: number) => string; upIsGood: boolean }[] = [
    { label: 'reward', key: 'eval_reward', fmt: (v) => fmtNum(v, 3), upIsGood: true },
    { label: 'correctness', key: EV('correctness'), fmt: fmtPct, upIsGood: true },
    { label: 'citations grounded', key: EV('citations_grounded'), fmt: fmtPct, upIsGood: true },
    { label: 'tool calls per correct', key: 'eval_tool_calls_per_correct', fmt: (v) => fmtNum(v, 1), upIsGood: false },
  ]
  return (
    <Panel title="Held-out (eval/fast)" aside={`latest at step ${last.step}${evalRows.length > 1 ? `, change since step ${first.step}` : ''}`}>
      <div className="flex flex-wrap gap-6">
        {tiles.map((t) => {
          const v = num(last[t.key])
          const dv = d(t.key)
          const good = dv === null || Math.abs(dv) < 1e-9 ? undefined : (t.upIsGood ? dv > 0 : dv < 0) ? 'good' : 'bad'
          return (
            <Stat
              key={t.key}
              label={t.label}
              tone={good}
              value={
                <span className="flex items-baseline gap-2">
                  <span className="text-foreground">{v === null ? '–' : t.fmt(v)}</span>
                  {dv !== null && Math.abs(dv) >= 1e-9 && <span className="text-[12px]">{dv > 0 ? '+' : '−'}{t.fmt(Math.abs(dv))}</span>}
                </span>
              }
            />
          )
        })}
      </div>
      {!compact && (
        <div className="mt-4 grid gap-5 md:grid-cols-2 2xl:grid-cols-3">
          {[
            { key: 'eval_reward', label: 'reward' },
            { key: EV('correctness'), label: 'correctness', domain: [0, 1] as [number, number] },
            { key: EV('citations_grounded'), label: 'citations grounded', domain: [0, 1] as [number, number] },
            { key: EV('format_ok'), label: 'format ok', domain: [0, 1] as [number, number] },
            { key: EV('citations_parse'), label: 'citations present', domain: [0, 1] as [number, number] },
            { key: EV('tool_calls'), label: 'tool calls' },
            { key: 'eval_tool_calls_per_correct', label: 'tool calls per correct' },
            { key: EV('stalled'), label: 'stalled', domain: [0, 1] as [number, number] },
          ]
            .filter((c) => has(rows, c.key))
            .map((c) => (
              <div key={c.key}>
                <p className="mb-1 text-[13px]">{c.label}</p>
                <MetricChart data={evalRows} height={220} legend={false} yDomain={c.domain ?? [0, 'auto']} series={[{ key: c.key, label: 'held-out', color: SERIES[2] }]} />
              </div>
            ))}
        </div>
      )}
    </Panel>
  )
}
