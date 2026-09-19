import type { AskRequest, Profile, RepoJobStatus, RepoSummary, SSEEvent, Suggestion } from './contracts'
import { readSSE } from './sse'
import { authHeaders, handleUnauthorized } from './auth'
import mockEvents from '../../mock/events.json'
import mockRepos from '../../mock/repos.json'
import mockProfiles from '../../mock/profiles.json'
import flaskAppPy from '../../mock/files/pallets__flask__85c5d93/src/flask/app.py?raw'
import flaskSansioAppPy from '../../mock/files/pallets__flask__85c5d93/src/flask/sansio/app.py?raw'

export interface Api {
  listRepos(): Promise<RepoSummary[]>
  addRepo(url: string, sha?: string): Promise<{ job_id: string }>
  repoStatus(jobId: string): Promise<RepoJobStatus>
  listProfiles(): Promise<Profile[]>
  ask(req: AskRequest, signal?: AbortSignal): AsyncIterable<SSEEvent>
  getFile(repoId: string, path: string): Promise<string>
  suggestions(repoId: string): Promise<Suggestion[]>
}

// ---------------------------------------------------------------------------
// HTTP client against apps/api (FastAPI). Base URL from VITE_API_URL.
// ---------------------------------------------------------------------------

export class HttpApi implements Api {
  private base: string
  constructor(base: string) {
    this.base = base
  }

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(this.base + path, {
      ...init,
      headers: { 'content-type': 'application/json', ...authHeaders(), ...(init?.headers ?? {}) },
      signal: init?.signal ?? AbortSignal.timeout(60_000),
    })
    handleUnauthorized(res)
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`)
    return (await res.json()) as T
  }

  listRepos() {
    return this.json<RepoSummary[]>('/repos')
  }
  addRepo(url: string, sha?: string) {
    return this.json<{ job_id: string }>('/repos', { method: 'POST', body: JSON.stringify({ url, sha }) })
  }
  repoStatus(jobId: string) {
    return this.json<RepoJobStatus>(`/repos/${encodeURIComponent(jobId)}/status`)
  }
  listProfiles() {
    return this.json<Profile[]>('/profiles')
  }
  async suggestions(repoId: string) {
    const r = await this.json<{ items: Suggestion[] }>(`/repos/${encodeURIComponent(repoId)}/suggestions`)
    return r.items
  }
  async *ask(req: AskRequest, signal?: AbortSignal) {
    const res = await fetch(this.base + '/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'text/event-stream', ...authHeaders() },
      body: JSON.stringify(req),
      signal,
    })
    handleUnauthorized(res)
    yield* readSSE<SSEEvent>(res, signal)
  }
  async getFile(repoId: string, path: string) {
    const q = new URLSearchParams({ repo_id: repoId, path })
    const res = await fetch(`${this.base}/file?${q}`, { headers: authHeaders(), signal: AbortSignal.timeout(30_000) })
    handleUnauthorized(res)
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return res.text()
  }
}

// ---------------------------------------------------------------------------
// Mock client: replays mock/events.json with realistic pacing, fakes an
// indexing job, serves the two flask files that the mock answer cites.
// ---------------------------------------------------------------------------

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(t)
      reject(new DOMException('aborted', 'AbortError'))
    })
  })

const PACE: Record<SSEEvent['type'], number> = {
  thinking: 420,
  tool_call: 180,
  tool_result: 380,
  answer: 260,
  citations: 120,
  stats: 60,
  done: 0,
  error: 0,
}

const MOCK_FILES: Record<string, string> = {
  'pallets__flask__85c5d93/src/flask/app.py': flaskAppPy,
  'pallets__flask__85c5d93/src/flask/sansio/app.py': flaskSansioAppPy,
}

interface MockJob {
  repo_id: string
  startedAt: number
}

const STAGES: { stage: RepoJobStatus['stage']; seconds: number }[] = [
  { stage: 'snapshot', seconds: 2.5 },
  { stage: 'index', seconds: 2 },
  { stage: 'summaries', seconds: 3 },
]

export class MockApi implements Api {
  private repos: RepoSummary[] = structuredClone(mockRepos) as RepoSummary[]
  private jobs = new Map<string, MockJob>()

  async listRepos() {
    await sleep(120)
    return this.repos
  }

  async addRepo(url: string, sha?: string) {
    await sleep(200)
    const m = url.match(/github\.com\/([^/\s]+)\/([^/\s#?]+)/)
    if (!m) throw new Error('Paste a GitHub URL like https://github.com/owner/repo')
    const owner = m[1]
    const repo = m[2].replace(/\.git$/, '')
    const sha7 = (sha ?? 'a1b2c3d').slice(0, 7)
    const repo_id = `${owner}__${repo}__${sha7}`
    const job_id = `job_${Date.now().toString(36)}`
    this.jobs.set(job_id, { repo_id, startedAt: Date.now() })
    if (!this.repos.some((r) => r.repo_id === repo_id)) {
      this.repos = [
        ...this.repos,
        { repo_id, url: `https://github.com/${owner}/${repo}`, sha: sha7, files: 0, lines: 0, stage: 'snapshot' },
      ]
    }
    return { job_id }
  }

  async repoStatus(jobId: string): Promise<RepoJobStatus> {
    await sleep(80)
    const job = this.jobs.get(jobId)
    if (!job) throw new Error(`Unknown job ${jobId}`)
    const elapsed = (Date.now() - job.startedAt) / 1000
    let t = 0
    for (const s of STAGES) {
      if (elapsed < t + s.seconds) {
        const progress = (elapsed - t) / s.seconds
        this.patch(job.repo_id, { stage: s.stage })
        return { repo_id: job.repo_id, stage: s.stage, progress, seconds: elapsed }
      }
      t += s.seconds
    }
    this.patch(job.repo_id, { stage: 'ready', files: 412, lines: 61_380, symbols: 2_904 })
    return { repo_id: job.repo_id, stage: 'ready', progress: 1, seconds: elapsed }
  }

  private patch(repoId: string, p: Partial<RepoSummary>) {
    this.repos = this.repos.map((r) => (r.repo_id === repoId ? { ...r, ...p } : r))
  }

  async listProfiles() {
    return mockProfiles as Profile[]
  }

  async *ask(_req: AskRequest, signal?: AbortSignal): AsyncIterable<SSEEvent> {
    const t0 = Date.now()
    let model = 0
    let tools = 0
    let mark: number | null = 0
    let openCall: number | null = null
    for (const raw of mockEvents.events as SSEEvent[]) {
      await sleep(PACE[raw.type], signal)
      const t = (Date.now() - t0) / 1000
      if ((raw.type === 'thinking' || raw.type === 'tool_call' || raw.type === 'answer') && mark !== null) {
        model += t - mark
        mark = null
      }
      if (raw.type === 'tool_call') openCall = t
      if (raw.type === 'tool_result') {
        if (openCall !== null) tools += t - openCall
        openCall = null
        mark = t
      }
      let ev: SSEEvent = { ...raw, t: Number(t.toFixed(3)) } as SSEEvent
      if (ev.type === 'stats') ev = { ...ev, model_seconds: Number(model.toFixed(2)), tool_seconds: Number(tools.toFixed(2)) }
      if (ev.type === 'citations') ev = { ...ev, format_ok: true, format_reason: '' }
      yield ev
    }
  }

  async suggestions(repoId: string): Promise<Suggestion[]> {
    await sleep(60)
    if (repoId.startsWith('pallets__flask')) {
      return [
        { type: 'locate', question: 'Where is the Flask application class defined, and what does it inherit from?' },
        { type: 'value', question: 'What is the default value of `use_cookies` in `Flask.test_client`?' },
        { type: 'enumerate', question: 'Which methods does the `Request` class define?' },
        { type: 'trace', question: 'Trace what happens when a request raises an exception.' },
        { type: 'explain', question: 'How does Flask decide which session interface to use?' },
      ]
    }
    const name = repoId.split('__')[1] ?? repoId
    return [
      { type: 'locate', question: `Where is the main entry point of ${name} defined?` },
      { type: 'value', question: `What version string does ${name} declare, and where?` },
      { type: 'enumerate', question: `Which modules make up the public API of ${name}?` },
      { type: 'trace', question: `Trace what happens when ${name} handles an error.` },
      { type: 'explain', question: `How does ${name} decide what to do on startup, and why?` },
    ]
  }
  async getFile(repoId: string, path: string) {
    await sleep(150)
    const text = MOCK_FILES[`${repoId}/${path}`]
    if (text === undefined) throw new Error(`No mock content for ${path}. The mock only serves the files the sample answer cites.`)
    return text
  }
}

export const MOCK_QUESTION = mockEvents.question

const base = import.meta.env.VITE_API_URL as string | undefined
export const api: Api = base ? new HttpApi(base.replace(/\/$/, '')) : new MockApi()
export const IS_MOCK = !base
