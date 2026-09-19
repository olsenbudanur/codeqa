// Read-only workshop API (D6). Everything here needs the real API; there is no mock.
import type { SSEEvent, Span } from './contracts'
import { authHeaders, handleUnauthorized } from './auth'

export const WORKSHOP_BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? '').replace(/\/$/, '')
export const HAS_API = WORKSHOP_BASE !== ''

async function get<T>(path: string, params?: Record<string, string | number | boolean | undefined | null>): Promise<T> {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined && v !== null && v !== '') q.set(k, String(v))
  const url = WORKSHOP_BASE + path + (q.size ? `?${q}` : '')
  const res = await fetch(url, { headers: authHeaders(), signal: AbortSignal.timeout(30_000) })
  handleUnauthorized(res)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  return (await res.json()) as T
}

export interface RunConfig {
  learning_rate?: number
  model_name?: string
  group_size?: number
  groups_per_batch?: number
  tasks_path?: string
  profile_name?: string
  epochs?: number
  variant?: string
  judge_model?: string | null
  eval_every?: number
  save_every?: number
  max_tokens?: number
  lora_rank?: number
}
export interface RunRow {
  name: string
  title: string
  hypothesis?: string | null
  variant?: string | null
  planned_steps?: number | null
  reward_setting?: string | null
  steps: number
  started: number | null
  updated: number | null
  live: boolean
  config: RunConfig
  iterations: number[]
  last_reward: number | null
}
export type MetricRow = Record<string, number | null> & { step: number }
export interface RunDetail extends RunRow {
  config_full: Record<string, unknown>
  metrics: MetricRow[]
  checkpoints: { name: string; batch: number; sampler_path: string }[]
  warnings: string[]
}
export interface RolloutRow {
  id: string
  traj_idx: number
  tags: string[]
  reward: number | null
  metrics: Record<string, number | null>
  stop: string | null
  tool_sequence: string[]
  answer_excerpt: string
}
export interface Iteration {
  run: string
  iteration: number
  groups: { group_idx: number; trajectories: RolloutRow[] }[]
}

export type EvalSummary = Record<string, number | null | undefined>
export interface CheckpointRow {
  name: string
  run: string
  step: number
  created_at: string
  tinker_path: string
  merged_path: string | null
  modal_volume: string | null
  profile: string
  is_final: boolean
  notes: string
  profile_kind: string | null
  servable: boolean
  evals: Record<string, EvalSummary>
}
export interface Checkpoints {
  checkpoints: CheckpointRow[]
  baselines: { name: string; profile: string; kind: string | null; model: string | null; evals: Record<string, EvalSummary> }[]
  sets: string[]
}

export interface DataSummary {
  counts: Record<string, Record<string, Record<string, number>>>
  repos: { repo_id: string; url: string; files: number; lines: number; symbols: number | null; summaries: boolean; map: boolean; nodoc: boolean }[]
  readme: string
}
export interface Passrate {
  bins: number
  histogram: number[]
  n: number
  n_scored: number
  window: [number, number]
  buckets: Record<string, number>
  per_source: Record<string, Record<string, number>>
}
export interface TaskCard {
  task_id: string
  repo_id: string
  split: string
  question: string
  task_type: string
  source: string
  source_id?: string | null
  grading: {
    expected_paths: string[]
    expected_symbols: string[]
    expected_literal: string | null
    reference_answer: string | null
    rubric: string[]
    required_citations: Span[]
  }
  budget: { max_tool_calls: number; max_turns: number; max_answer_tokens: number } | null
  passrate: Record<string, number | null> | null
  difficulty: number | null
  bucket: 'too_easy' | 'kept' | 'too_hard' | null
  example_traces: string[]
}

export interface TraceRow {
  id: string
  task_id: string
  profile: string
  run: string
  repo_id: string | null
  source: string | null
  task_type: string | null
  question: string
  stop_reason: string | null
  turns: number | null
  tool_calls: number | null
  prompt_tokens: number | null
  reward: number | null
  has_citations: boolean
  mtime: number | null
}
export interface TraceList {
  total: number
  facets: Record<'profile' | 'run' | 'source' | 'task_type' | 'stop_reason', string[]>
  traces: TraceRow[]
}
export interface Grade {
  reward: number | null
  components: Record<string, number | null>
  gate_failed: string | null
  notes: string
  source: string
}
export interface CitationRow extends Span {
  exists: boolean
  grounded: boolean
  verified: boolean
}
export interface TraceDetail {
  id: string
  kind: 'trace' | 'eval' | 'rollout'
  run: string
  task_id: string
  profile: string
  task: Partial<TaskCard> | null
  repo_id: string | null
  question: string
  events: SSEEvent[]
  stats: Record<string, unknown>
  answer: string
  grade: Grade
  citations: CitationRow[]
  messages?: import('@/components/workshop/raw-conversation').RawMessage[]
  messages_note?: string
}

export interface RepoOverview {
  repo_id: string
  url: string | null
  sha: string | null
  files: { path: string; lines: number; lang: string | null }[]
  n_files: number
  lines: number
  symbols: number | null
  dropped: Record<string, number>
  languages: Record<string, number>
  top_dirs: Record<string, number>
  map: string
  map_lines: number
  n_summaries: number
  nodoc: boolean
  has_nodoc_variant: boolean
}
export interface ToolSpec {
  name: string
  description: string
  parameters: { properties: Record<string, { type?: string; anyOf?: { type: string }[]; description?: string; default?: unknown }>; required?: string[] }
}
export interface ToolRun {
  name: string
  args: Record<string, unknown>
  output: string
  chars: number
  seconds: number
  error: boolean
  files_read: Span[]
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(WORKSHOP_BASE + path, { method: 'POST', headers: { 'content-type': 'application/json', ...authHeaders() }, body: JSON.stringify(body), signal: AbortSignal.timeout(60_000) })
  handleUnauthorized(res)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
  return (await res.json()) as T
}

export const workshop = {
  repoOverview: (id: string) => get<RepoOverview>(`/repos/${encodeURIComponent(id)}/overview`),
  repoTools: (id: string) => get<{ tools: ToolSpec[]; caps: Record<string, number> }>(`/repos/${encodeURIComponent(id)}/tools`),
  runTool: (id: string, name: string, args: Record<string, unknown>) => post<ToolRun>(`/repos/${encodeURIComponent(id)}/tool`, { name, args }),
  runs: () => get<RunRow[]>('/runs'),
  run: (name: string) => get<RunDetail>(`/runs/${encodeURIComponent(name)}`),
  iteration: (name: string, n: number) => get<Iteration>(`/runs/${encodeURIComponent(name)}/iterations/${n}`),
  checkpoints: () => get<Checkpoints>('/checkpoints'),
  dataSummary: () => get<DataSummary>('/data/summary'),
  passrate: (bins = 10) => get<Passrate>('/data/passrate', { bins }),
  tasks: (p: Record<string, string | number | boolean | undefined>) => get<{ total: number; tasks: TaskCard[] }>('/data/tasks', p),
  traces: (p: Record<string, string | number | boolean | undefined>) => get<TraceList>('/traces', p),
  trace: (id: string) => get<TraceDetail>(`/traces/${id}`),
  compare: (a: string, b: string) => get<{ a: TraceDetail; b: TraceDetail }>('/traces/compare', { a, b }),
}

export const fmtNum = (n: number | null | undefined, digits = 2): string =>
  n === null || n === undefined || Number.isNaN(n) ? '–' : Math.abs(n) >= 1000 ? `${(n / 1000).toFixed(1)}k` : n.toFixed(digits)
export const fmtPct = (n: number | null | undefined): string => (n === null || n === undefined ? '–' : `${Math.round(n * 100)}%`)
export const fmtWhen = (ts: number | null): string => (ts ? new Date(ts * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '–')
