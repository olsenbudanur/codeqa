import { useState } from 'react'
import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { MetricRow } from '@/lib/workshop'

// One axis per chart. Series colors are the viz roles; text stays in text tokens.
export const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)', 'var(--series-5)', 'var(--series-6)', 'var(--series-7)', 'var(--series-8)']

export interface Series {
  key: string // field on the row
  label: string
  color: string
  kind?: 'line' | 'points' | 'band' | 'dashed' | 'area'
  hidden?: boolean // off until the viewer clicks it in the legend (raw per-step lines default to hidden behind their 5-step mean)
  follows?: string // a band drawn around another series: shown and hidden with it, never listed in the legend
}

export function rollingMean(values: (number | null)[], window = 5): (number | null)[] {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - window + 1), i + 1).filter((v): v is number => v !== null && v !== undefined)
    return slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null
  })
}

const tickStyle = { fontSize: 12, fill: 'var(--muted-foreground)', fontFamily: 'var(--font-mono)' }

// Ticks and tooltips: significant digits, so 3e-4 does not read as 0.00.
export function fmtSig(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '–'
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return `${(v / 1000).toFixed(1)}k`
  if (a >= 10) return v.toFixed(1)
  if (a >= 0.01) return v.toFixed(a >= 1 ? 2 : 3)
  return v.toExponential(1)
}

function TooltipBox({ active, payload, label, series, pct, format }: { active?: boolean; payload?: { dataKey?: string; value?: unknown; name?: string }[]; label?: unknown; series: Series[]; pct?: boolean; format?: (v: number) => string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="min-w-[220px] rounded-md border bg-popover px-3 py-2.5 font-mono text-[12.5px] shadow-md">
      <div className="mb-1 text-muted-foreground">step {String(label)}</div>
      {series.map((s) => {
        const p = payload.find((x) => x.dataKey === s.key)
        if (!p || p.value === null || p.value === undefined) return null
        if (pct && typeof p.value === 'number' && Math.round(p.value * 100) === 0) return null   // stacked bars: only reasons that occurred
        const one = (x: number) => (format ? format(x) : fmtSig(x))
        const v = Array.isArray(p.value) ? `${one(p.value[0] as number)}–${one(p.value[1] as number)}` : pct ? `${Math.round((p.value as number) * 100)}%` : one(p.value as number)
        return (
          <div key={s.key} className="flex items-center gap-2">
            <span className="size-2 rounded-sm" style={{ background: s.color }} aria-hidden />
            <span className="whitespace-nowrap text-muted-foreground">{s.label}</span>
            <span className="ml-auto pl-3 text-foreground">{v}</span>
          </div>
        )
      })}
    </div>
  )
}

export function MetricChart({
  data,
  series,
  height = 220,
  yDomain,
  legend = true,
  refLines = [],
  yFormat,
}: {
  data: (MetricRow | Record<string, unknown>)[]
  series: Series[]
  height?: number
  yDomain?: [number | 'auto', number | 'auto']
  legend?: boolean
  refLines?: { y: number; label: string }[]
  yFormat?: (v: number) => string
}) {
  // Legend clicks toggle series. State is keyed by series key, so a re-render with the same series keeps the choice.
  const [hidden, setHidden] = useState<Set<string>>(() => new Set(series.filter((s) => s.hidden).map((s) => s.key)))
  const isHidden = (s: Series) => hidden.has(s.follows ?? s.key)
  const toggle = (key: string) =>
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  const visible = series.filter((s) => !isHidden(s))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
        <XAxis dataKey="step" tick={tickStyle} axisLine={{ stroke: 'var(--viz-grid)' }} tickLine={false} allowDecimals={false} />
        <YAxis tick={tickStyle} axisLine={false} tickLine={false} domain={yDomain ?? ['auto', 'auto']} width={52} tickFormatter={(v: number) => (yFormat ? yFormat(v) : fmtSig(v))} />
        <Tooltip content={<TooltipBox series={visible} format={yFormat} />} cursor={{ stroke: 'var(--muted-foreground)', strokeDasharray: '3 3' }} allowEscapeViewBox={{ x: false, y: true }} wrapperStyle={{ zIndex: 50 }} />
        {legend && series.length > 1 && (
          <Legend
            iconType="plainline"
            wrapperStyle={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--muted-foreground)', cursor: 'pointer' }}
            onClick={(e: { dataKey?: unknown }) => typeof e.dataKey === 'string' && toggle(e.dataKey)}
            formatter={(value: string, entry: { dataKey?: unknown }) => (
              <span style={{ opacity: typeof entry.dataKey === 'string' && hidden.has(entry.dataKey) ? 0.45 : 1, textDecoration: typeof entry.dataKey === 'string' && hidden.has(entry.dataKey) ? 'line-through' : undefined }}>{value}</span>
            )}
          />
        )}
        {refLines.map((r) => (
          <ReferenceLine key={r.label} y={r.y} stroke="var(--unverified)" strokeDasharray="4 3" label={{ value: r.label, position: 'insideTopRight', fontSize: 10, fill: 'var(--muted-foreground)', fontFamily: 'var(--font-mono)' }} />
        ))}
        {series.map((s) =>
          s.kind === 'band' ? (
            <Area key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke="none" fill={s.color} fillOpacity={0.16} isAnimationActive={false} connectNulls legendType="none" hide={isHidden(s)} />
          ) : s.kind === 'area' ? (
            <Area key={s.key} type="monotone" dataKey={s.key} name={s.label} stackId="a" stroke={s.color} strokeWidth={1} fill={s.color} fillOpacity={0.55} isAnimationActive={false} connectNulls hide={isHidden(s)} />
          ) : s.kind === 'points' ? (
            <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke="none" dot={{ r: 5, fill: s.color, stroke: 'var(--background)', strokeWidth: 2 }} activeDot={{ r: 6 }} isAnimationActive={false} connectNulls={false} hide={isHidden(s)} />
          ) : (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={s.kind === 'dashed' ? '4 3' : undefined}
              dot={s.kind === 'dashed' ? false : { r: 4, fill: s.color, strokeWidth: 0 }}
              activeDot={{ r: 5, stroke: 'var(--background)', strokeWidth: 2 }}
              isAnimationActive={false}
              connectNulls
              hide={isHidden(s)}
            />
          ),
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function Histogram({ counts, window, total }: { counts: number[]; window: [number, number]; total: number }) {
  const bins = counts.length
  const max = Math.max(1, ...counts)
  return (
    <div>
      <div className="flex h-[140px] items-end gap-[2px]" role="img" aria-label="Pass-rate histogram">
        {counts.map((c, i) => {
          const lo = i / bins
          const hi = (i + 1) / bins
          const kept = lo >= window[0] - 1e-9 && hi <= window[1] + 1e-9
          return (
            <div key={i} className="group relative flex h-full flex-1 flex-col justify-end" title={`${Math.round(lo * 100)}–${Math.round(hi * 100)}%: ${c} tasks`}>
              <div className="rounded-t-[4px]" style={{ height: `${(c / max) * 100}%`, minHeight: c ? 2 : 0, background: kept ? 'var(--series-1)' : 'var(--muted-foreground)', opacity: kept ? 1 : 0.45 }} />
            </div>
          )
        })}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[11px] text-muted-foreground">
        <span>0% (base never right)</span>
        <span>kept window {Math.round(window[0] * 100)}–{Math.round(window[1] * 100)}%</span>
        <span>100% (always right)</span>
      </div>
      <p className="mt-1 font-mono text-[11px] text-muted-foreground">{total.toLocaleString()} tasks scored</p>
    </div>
  )
}


// Stacked bars per step (a funnel): each series is a share of 1; a 2px surface gap separates segments.
export function StackedBars({ data, series, height = 200 }: { data: (MetricRow | Record<string, unknown>)[]; series: Series[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }} barCategoryGap="20%">
        <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
        <XAxis dataKey="step" tick={tickStyle} axisLine={{ stroke: 'var(--viz-grid)' }} tickLine={false} allowDecimals={false} />
        <YAxis tick={tickStyle} axisLine={false} tickLine={false} domain={[0, 1]} width={56} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
        <Tooltip content={<TooltipBox series={series} pct />} cursor={{ fill: 'var(--accent)' }} allowEscapeViewBox={{ x: true, y: true }} wrapperStyle={{ zIndex: 50 }} />
        <Legend iconType="square" wrapperStyle={{ fontSize: 11.5, fontFamily: 'var(--font-mono)', color: 'var(--muted-foreground)' }} />
        {series.map((s, i) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} stackId="g" fill={s.color} stroke="var(--background)" strokeWidth={2} isAnimationActive={false} maxBarSize={56} radius={i === series.length - 1 ? [4, 4, 0, 0] : 0} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
