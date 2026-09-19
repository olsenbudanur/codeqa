// TypeScript mirror of the parts of codeqa/shared/contracts.py the web app consumes.
// Do not extend these here; request changes in docs/agents/LOG.md.

// C4 · exact copy of CITATION_RE. tests/test_web_contracts.py asserts the two stay equal.
export const CITATION_RE = /\[([^\]\s:]+):L(\d+)(?:-L(\d+))?\]/g

// C1 · repo list (subset of Manifest plus index stage)
export type RepoStage = 'snapshot' | 'index' | 'summaries' | 'ready' | 'error'

export interface RepoSummary {
  repo_id: string
  url: string
  sha: string
  files: number
  lines: number
  symbols?: number
  stage: RepoStage
}

export interface RepoJobStatus {
  repo_id: string
  stage: RepoStage
  progress: number // 0..1 within the current stage
  seconds: number
  message?: string
}

// C8 · endpoint profile (what the picker needs)
export interface Profile {
  name: string
  kind: 'openai' | 'anthropic' | 'tinker'
  model: string
  label?: string
  note?: string
  source?: 'profiles.yaml' | 'checkpoints'
}

// C9 · product stream events
export type ToolName = 'overview' | 'find_symbol' | 'grep' | 'read_file' | 'list_dir'

export interface CitationItem {
  path: string
  start: number
  end: number
  verified: boolean
}

// `t` = seconds since the episode started, stamped by the API on every event (optional; the driver itself does not send it).
export type SSEEvent =
  | { type: 'thinking'; text: string; t?: number }
  | { type: 'tool_call'; name: ToolName; args: Record<string, unknown>; why?: string; t?: number }
  | { type: 'tool_result'; name: ToolName; summary: string; chars: number; text?: string; t?: number }
  | { type: 'answer'; markdown: string; t?: number }
  | { type: 'citations'; items: CitationItem[]; format_ok?: boolean; format_reason?: string; t?: number }
  | { type: 'stats'; tool_calls: number; prompt_tokens: number; completion_tokens: number; seconds: number; model_seconds?: number; tool_seconds?: number; t?: number }
  | { type: 'done' }
  | { type: 'error'; message: string }

export type TaskType = 'locate' | 'value' | 'enumerate' | 'trace' | 'explain'

export interface AskRequest {
  repo_id: string
  question: string
  profile: string
  task_type?: TaskType
}

export interface Suggestion {
  type: TaskType
  question: string
}

export interface Span {
  path: string
  start: number
  end: number
}
