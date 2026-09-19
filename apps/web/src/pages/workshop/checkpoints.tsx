import { useEffect, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { navigate } from '@/lib/router'
import { fmtNum, fmtPct, workshop, type Checkpoints, type EvalSummary } from '@/lib/workshop'
import { Button } from '@/components/ui/button'
import { Chip, ErrorNote, Loading, Page, Panel } from '@/components/workshop/ui'

const COLS: { key: keyof EvalSummary & string; label: string; fmt: (v: number) => string }[] = [
  { key: 'reward', label: 'reward', fmt: (v) => fmtNum(v, 3) },
  { key: 'correct_rate', label: 'correct', fmt: fmtPct },
  { key: 'format_ok', label: 'format ok', fmt: fmtPct },
  { key: 'citation_valid', label: 'citations valid', fmt: fmtPct },
  { key: 'tool_calls', label: 'tool calls', fmt: (v) => fmtNum(v, 1) },
  { key: 'tool_calls_per_correct', label: 'calls / correct', fmt: (v) => fmtNum(v, 1) },
  { key: 'sweqa_total', label: 'SWE-QA /100', fmt: (v) => fmtNum(v, 1) },
  { key: 'n', label: 'n', fmt: (v) => String(v) },
]

export function CheckpointsPage() {
  const [data, setData] = useState<Checkpoints | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    workshop.checkpoints().then(setData).catch((e: Error) => setError(e.message))
  }, [])
  return (
    <Page title="Checkpoints" wide>
      {error && <ErrorNote error={error} />}
      {!data && !error && <Loading what="checkpoints" />}
      {data && (
        <div className="space-y-4">
          {data.sets.length === 0 && <p className="text-sm text-muted-foreground">No evals on disk yet.</p>}
          {data.sets.map((set) => (
            <Panel key={set} title={`Eval set: ${set}`} aside="baselines first, then checkpoints">
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead className="text-left text-xs text-muted-foreground">
                    <tr className="border-b">
                      <th className="py-2 pr-3 font-normal">Model</th>
                      {COLS.map((c) => (
                        <th key={c.key} className="px-2 py-2 text-right font-normal">{c.label}</th>
                      ))}
                      <th className="py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {data.baselines.map((b) => (
                      <Row key={b.name} name={b.name} sub={b.model ?? ''} evals={b.evals[set]} />
                    ))}
                    {data.checkpoints.map((c) => (
                      <Row
                        key={c.name}
                        name={c.name}
                        sub={`${c.run}, step ${c.step}${c.is_final ? ', final' : ''}`}
                        evals={c.evals[set]}
                        chips={[c.servable ? <Chip key="s" tone="good">served</Chip> : <Chip key="t">tinker</Chip>]}
                        action={
                          <Button variant="ghost" size="sm" onClick={() => navigate(`/compare?left=qwen4b-base&right=${encodeURIComponent(c.profile)}`)}>
                            Compare with the untrained model
                            <ArrowRight />
                          </Button>
                        }
                      />
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          ))}
          <Panel title="All checkpoints" aside={`${data.checkpoints.length} in data/models/manifest.json`}>
            {data.checkpoints.length === 0 ? (
              <p className="text-sm text-muted-foreground">None yet. Training runs write here through `serving.export`.</p>
            ) : (
              <ul className="divide-y">
                {data.checkpoints.map((c) => (
                  <li key={c.name} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-2 text-[13px]">
                    <span className="font-mono">{c.name}</span>
                    <span className="text-muted-foreground">{c.run}, step {c.step}, {new Date(c.created_at).toLocaleString()}</span>
                    <span className="truncate font-mono text-[11.5px] text-muted-foreground">{c.tinker_path}</span>
                    {c.notes && <span className="basis-full text-xs text-muted-foreground">{c.notes}</span>}
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      )}
    </Page>
  )
}

function Row({ name, sub, evals, chips, action }: { name: string; sub: string; evals?: EvalSummary; chips?: React.ReactNode[]; action?: React.ReactNode }) {
  return (
    <tr className="border-b last:border-0">
      <td className="py-2 pr-3">
        <span className="flex items-center gap-2 font-mono">{name} {chips}</span>
        <span className="block text-xs text-muted-foreground">{sub}</span>
      </td>
      {COLS.map((c) => {
        const v = evals?.[c.key]
        return (
          <td key={c.key} className="px-2 py-2 text-right font-mono tabular-nums">{v === null || v === undefined ? <span className="text-muted-foreground">–</span> : c.fmt(v)}</td>
        )
      })}
      <td className="py-2 text-right">{action}</td>
    </tr>
  )
}
