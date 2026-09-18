import type { CitationItem, SSEEvent, Span, ToolName } from '@/lib/contracts'

export interface ToolRow {
  kind: 'call'
  id: number
  name: ToolName
  args: Record<string, unknown>
  why?: string
  result?: { summary: string; chars: number }
}

export interface ThinkingRow {
  kind: 'thinking'
  id: number
  text: string
}

export type LogRow = ToolRow | ThinkingRow

export type EpisodeStatus = 'idle' | 'running' | 'done' | 'error'

export interface Episode {
  status: EpisodeStatus
  question: string
  repoId: string
  profile: string
  rows: LogRow[]
  answer?: string
  citations?: CitationItem[]
  stats?: { tool_calls: number; prompt_tokens: number; completion_tokens: number; seconds: number }
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
        rows: [...state.rows, { kind: 'call', id: nextId++, name: ev.name, args: ev.args, why: ev.why }],
      }
    case 'tool_result': {
      // Attach to the last unanswered call with the same tool name.
      const rows = [...state.rows]
      for (let i = rows.length - 1; i >= 0; i--) {
        const r = rows[i]
        if (r.kind === 'call' && r.name === ev.name && !r.result) {
          rows[i] = { ...r, result: { summary: ev.summary, chars: ev.chars } }
          break
        }
      }
      return { ...state, rows }
    }
    case 'answer':
      return { ...state, answer: ev.markdown }
    case 'citations':
      return { ...state, citations: ev.items }
    case 'stats': {
      const { type: _t, ...stats } = ev
      return { ...state, stats }
    }
    case 'done':
      return { ...state, status: 'done' }
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
