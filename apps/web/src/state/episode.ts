import type { CitationItem, SSEEvent, Span, ToolName } from '@/lib/contracts'

export interface ToolRow {
  kind: 'call'
  id: number
  name: ToolName
  args: Record<string, unknown>
  why?: string
  result?: { summary: string; chars: number; text?: string; seconds?: number }
  startedAt?: number // seconds since episode start (from the API's `t`)
  turn?: number // the assistant turn this call came from; several calls per turn = one round (bash_v3)
}

export interface ThinkingRow {
  kind: 'thinking'
  id: number
  text: string
}

// A message the harness injected (today: the forced final-answer turn when the round budget ran out).
export interface NoticeRow {
  kind: 'notice'
  id: number
  notice: string
  text: string
}

export type LogRow = ToolRow | ThinkingRow | NoticeRow

export type EpisodeStatus = 'idle' | 'running' | 'done' | 'error'

export interface Episode {
  status: EpisodeStatus
  question: string
  repoId: string
  profile: string
  rows: LogRow[]
  answer?: string
  citations?: CitationItem[]
  stats?: { tool_calls: number; prompt_tokens: number; completion_tokens: number; seconds: number | null; model_seconds?: number; tool_seconds?: number; forced_answer?: boolean }
  format?: { ok: boolean; reason: string }
  budget?: { context_tokens: number; context_cap: number; messages_left: number; turn: number } // last round's trailer (bash_v3)
  error?: string
  startedAt?: number
}

export const emptyEpisode: Episode = { status: 'idle', question: '', repoId: '', profile: '', rows: [] }

export type Action =
  | { type: 'start'; question: string; repoId: string; profile: string }
  | { type: 'event'; event: SSEEvent }
  | { type: 'abort' }
  | { type: 'reset' }
  | { type: 'load'; episode: Episode }

let nextId = 1

export function episodeReducer(state: Episode, action: Action): Episode {
  switch (action.type) {
    case 'start':
      return {
        ...emptyEpisode,
        status: 'running',
        question: action.question,
        repoId: action.repoId,
        profile: action.profile,
        startedAt: Date.now(),
      }
    case 'reset':
      return emptyEpisode
    case 'load':
      return action.episode
    case 'abort':
      return state.status === 'running' ? { ...state, status: 'error', error: 'Stopped.' } : state
    case 'event':
      return applyEvent(state, action.event)
  }
}

function applyEvent(state: Episode, ev: SSEEvent): Episode {
  switch (ev.type) {
    case 'thinking':
      return { ...state, rows: [...state.rows, { kind: 'thinking', id: nextId++, text: ev.text }] }
    case 'tool_call':
      return {
        ...state,
        rows: [...state.rows, { kind: 'call', id: nextId++, name: ev.name, args: ev.args, why: ev.why, startedAt: ev.t, turn: ev.turn }],
      }
    case 'tool_result': {
      // Attach to the first unanswered call with the same tool name: the driver runs a round's calls in order and
      // emits their results in the same order, so several open `bash` calls resolve first-in first-out.
      const rows = [...state.rows]
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i]
        if (r.kind === 'call' && r.name === ev.name && !r.result) {
          const seconds = ev.t !== undefined && r.startedAt !== undefined ? Math.max(0, ev.t - r.startedAt) : undefined
          rows[i] = { ...r, result: { summary: ev.summary, chars: ev.chars, text: ev.text, seconds } }
          break
        }
      }
      return { ...state, rows }
    }
    case 'budget':
      return { ...state, budget: { context_tokens: ev.context_tokens, context_cap: ev.context_cap, messages_left: ev.messages_left, turn: ev.turn } }
    case 'notice':
      return { ...state, rows: [...state.rows, { kind: 'notice', id: nextId++, notice: ev.kind, text: ev.text }] }
    case 'answer':
      return { ...state, answer: ev.markdown }
    case 'citations':
      return { ...state, citations: ev.items, format: ev.format_ok === undefined ? state.format : { ok: ev.format_ok, reason: ev.format_reason ?? '' } }
    case 'stats': {
      const { type: _t, t: _time, ...stats } = ev
      return { ...state, stats }
    }
    case 'done':
      return { ...state, status: state.status === 'error' ? 'error' : 'done' }
    case 'error':
      return { ...state, status: 'error', error: ev.message }
  }
}

// Files the agent actually read this episode (from read_file calls), for the
// "consulted" list and for the stub grounding check.
export function filesRead(rows: LogRow[]): Span[] {
  const out: Span[] = []
  for (const r of rows) {
    if (r.kind !== 'call' || r.name !== 'read_file' || !r.result) continue
    const path = String(r.args.path ?? '')
    const start = Number(r.args.start ?? 1)
    const end = Number(r.args.end ?? start)
    if (path) out.push({ path, start, end })
  }
  return out
}

// Files the agent touched in any way (reads, symbol hits), for "consulted".
export function filesTouched(rows: LogRow[]): string[] {
  const seen = new Set<string>()
  for (const r of rows) {
    if (r.kind !== 'call') continue
    if (r.name === 'read_file' && r.args.path) seen.add(String(r.args.path))
    if (r.result) for (const m of r.result.summary.matchAll(/\b([\w./-]+\.\w+):L\d+/g)) seen.add(m[1])
  }
  return [...seen]
}


// Rounds (bash_v3): consecutive call rows from the same assistant turn, two or more, fold into one group so a
// message with four commands reads as one step until expanded.
export type RoundGroup = { kind: 'round'; turn: number; rows: ToolRow[] }
export type GroupedRow = LogRow | RoundGroup

export function groupRounds(rows: LogRow[]): GroupedRow[] {
  const out: GroupedRow[] = []
  for (const r of rows) {
    const last = out[out.length - 1]
    if (r.kind === 'call' && r.turn !== undefined) {
      if (last && last.kind === 'round' && last.turn === r.turn) { last.rows.push(r); continue }
      if (last && last.kind === 'call' && last.turn === r.turn) { out[out.length - 1] = { kind: 'round', turn: r.turn, rows: [last, r] }; continue }
    }
    out.push(r)
  }
  return out
}
