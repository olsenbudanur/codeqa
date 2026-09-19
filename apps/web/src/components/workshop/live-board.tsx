import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, CircleAlert } from 'lucide-react'
import { navigate } from '@/lib/router'
import { fmtNum, fmtWhen, workshop, type MetricRow, type RunDetail, type RunRow } from '@/lib/workshop'
import { cn } from '@/lib/utils'
import { MetricChart, SERIES } from './charts'
import { Chip, Panel } from './ui'
import { withDerived } from './panels'

const COLORS = [SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[4], SERIES[6], SERIES[7]]

// Every run that is currently writing metrics, on one chart, with a card each.
export function LiveBoard({ runs, selected, onSelect, pollMs }: { runs: RunRow[]; selected: string | null; onSelect: (name: string) => void; pollMs: number }) {
  const live = runs.filter((r) => r.live)
  const [details, setDetails] = useState<Record<string, RunDetail>>({})
  const key = live.map((r) => `${r.name}@${r.updated}`).join(',')
  useEffect(() => {
    let alive = true
    const load = () =>
      Promise.all(live.map((r) => workshop.run(r.name).catch(() => null))).then((rs) => {
        if (!alive) return
        const next: Record<string, RunDetail> = {}
        rs.forEach((d) => { if (d) next[d.name] = d })
        setDetails(next)
      })
    void load()
    const t = window.setInterval(load, pollMs)
    return () => {
      alive = false
      window.clearInterval(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, pollMs])

  const rows = useMemo(() => {
    const per = live.map((r) => (details[r.name] ? withDerived(details[r.name]) : []))
    const steps = new Set(per.flatMap((p) => p.map((m) => m.step)))
    return [...steps].sort((a, b) => a - b).map((step) => {
      const row: Record<string, unknown> = { step }
      per.forEach((p, i) => {
        const m = p.find((x) => x.step === step)
        row[`r${i}`] = m?.reward ?? null
        row[`e${i}`] = m?.eval_reward ?? null
      })
      return row as MetricRow
    })
  }, [live, details])

  if (live.length < 2) return null
  return (
    <Panel title={`${live.length} runs training now`} aside="reward per step, held-out as points; click a card to monitor it below">
      <MetricChart
        data={rows}
        height={320}
        yDomain={[0, 'auto']}
        series={live.flatMap((r, i) => [
          { key: `r${i}`, label: r.name, color: COLORS[i % COLORS.length] },
          { key: `e${i}`, label: `${r.name} held-out`, color: COLORS[i % COLORS.length], kind: 'points' as const },
        ])}
      />
      <div className={cn('mt-4 grid gap-3 md:grid-cols-2', live.length >= 3 && 'xl:grid-cols-4')}>
        {live.map((r, i) => {
          const d = details[r.name]
          const last = d ? withDerived(d).at(-1) : undefined
          const warn = d?.warnings.length ?? 0
          const done = r.planned_steps ? Math.min(1, r.steps / r.planned_steps) : null
          return (
            <button
              key={r.name}
              type="button"
              onClick={() => onSelect(r.name)}
              className={cn('rounded-lg border p-3.5 text-left transition-colors hover:bg-accent/60', selected === r.name ? 'border-verified/60 bg-verified-soft/30' : 'border-border')}
            >
              <span className="flex items-center gap-2">
                <span className="size-2.5 rounded-sm" style={{ background: COLORS[i % COLORS.length] }} aria-hidden />
                <span className="truncate text-[13.5px] font-medium">{r.title}</span>
              </span>
              <span className="mt-0.5 block truncate font-mono text-[11px] text-muted-foreground">{r.name}{r.variant ? ` · ${r.variant}` : ''}</span>
              <dl className="mt-3 grid grid-cols-3 gap-2">
                <div>
                  <dd className="font-mono text-[17px] leading-none tabular-nums">{r.steps}{r.planned_steps ? <span className="text-[11px] text-muted-foreground">/{r.planned_steps}</span> : null}</dd>
                  <dt className="mt-1 text-[11px] text-muted-foreground">steps</dt>
                </div>
                <div>
                  <dd className="font-mono text-[17px] leading-none tabular-nums">{fmtNum(last?.reward as number, 3)}</dd>
                  <dt className="mt-1 text-[11px] text-muted-foreground">reward</dt>
                </div>
                <div>
                  <dd className={cn('font-mono text-[17px] leading-none tabular-nums', warn > 0 ? 'text-unverified' : 'text-verified')}>{warn}</dd>
                  <dt className="mt-1 text-[11px] text-muted-foreground">warnings</dt>
                </div>
              </dl>
              {done !== null && (
                <span className="mt-3 block h-1 overflow-hidden rounded-full bg-muted">
                  <span className="block h-full rounded-full" style={{ width: `${done * 100}%`, background: COLORS[i % COLORS.length] }} />
                </span>
              )}
              <span className="mt-2 flex items-center justify-between font-mono text-[11px] text-muted-foreground">
                <span>updated {fmtWhen(r.updated)}</span>
                {warn > 0 && <span className="flex items-center gap-1 text-unverified"><CircleAlert className="size-3" />{d?.warnings[0]?.slice(0, 40)}…</span>}
              </span>
            </button>
          )
        })}
      </div>
      <p className="mt-3 flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
        <Chip>sync</Chip> logs arrive from the Modal volume every ~2 min; "updated" is the last sync that changed the file.
        <button type="button" className="ml-auto inline-flex items-center gap-1 hover:text-foreground" onClick={() => navigate(`/workshop/runs/${live[0].name}?vs=${encodeURIComponent(live.slice(1).map((r) => r.name).join(','))}`)}>
          full overlay on the run page <ArrowRight className="size-3" />
        </button>
      </p>
    </Panel>
  )
}
