import type { CitationItem, SSEEvent } from './contracts'
import { authHeaders, handleUnauthorized } from './auth'
import { readSSE } from './sse'
import { WORKSHOP_BASE } from './workshop'

export interface Candidate {
  label: string
  answer: string
  citations: CitationItem[]
  tool_calls?: number
  profile?: string // lets the API grade the saved trace of this answer with the training grader
}

// The training grader's verdict on one candidate (reference = the referee's answer): reward 0..1, the components
// the held-out evaluator reports, and which gate zeroed it, if any.
export interface Grade {
  index: number
  label: string
  reward?: number | null
  components?: Record<string, number | null>
  gate_failed?: string | null
  notes?: string
  error?: string
}

export interface Verdict {
  index: number
  label: string
  score?: number
  correct?: boolean
  summary?: string
  issues?: string[]
  strengths?: string[]
  error?: string
}

export type JudgeFrame =
  | { type: 'phase'; phase: 'research'; model: string }
  | { type: 'phase'; phase: 'judging'; reference: string; ref_calls: number; ref_seconds: number }
  | { type: 'ref'; event: SSEEvent }
  | ({ type: 'verdict' } & Verdict)
  | ({ type: 'grade' } & Grade)
  | { type: 'error'; message: string }
  | { type: 'done' }

export async function* judge(repoId: string, question: string, candidates: Candidate[], signal?: AbortSignal, variant?: string): AsyncIterable<JudgeFrame> {
  if (!WORKSHOP_BASE) throw new Error('The referee needs the API (set VITE_API_URL).')
  const res = await fetch(`${WORKSHOP_BASE}/judge`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream', ...authHeaders() },
    body: JSON.stringify(variant ? { repo_id: repoId, question, candidates, variant } : { repo_id: repoId, question, candidates }),
    signal,
  })
  handleUnauthorized(res)
  yield* readSSE<JudgeFrame>(res, signal)
}
