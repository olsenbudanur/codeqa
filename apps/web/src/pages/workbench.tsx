import { useCallback, useEffect, useRef, useState } from 'react'
import { MessageSquarePlus, PanelLeft, Square } from 'lucide-react'
import { navigate } from '@/lib/router'
import { Wordmark } from '@/components/wordmark'
import { Toaster, toast } from 'sonner'
import { api, IS_MOCK } from '@/lib/api'
import type { Profile, RepoJobStatus, RepoSummary, Span } from '@/lib/contracts'
import { useEpisode } from '@/hooks/use-episode'
import { useMediaQuery } from '@/hooks/use-media-query'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ThemeToggle } from '@/components/theme-toggle'
import { ModelPicker } from '@/components/model-picker'
import { RepoSwitcher } from '@/components/repo/repo-switcher'
import { Conversations } from '@/components/repo/conversations'
import { IndexStepper } from '@/components/repo/index-stepper'
import { deleteConversation, listConversations, saveConversation, type Conversation } from '@/lib/history'
import { repoName } from '@/lib/repo'
import { QuestionBox } from '@/components/ask/question-box'
import { Ledger } from '@/components/research/ledger'
import { AnswerPanel } from '@/components/answer/answer-panel'
import { FileViewer } from '@/components/file/file-viewer'

const TOOL_BUDGET = 8

export function Workbench() {
  const [repos, setRepos] = useState<RepoSummary[]>([])
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [repoId, setRepoId] = useState<string | null>(null)
  const [profile, setProfile] = useState<string>('')
  const [job, setJob] = useState<RepoJobStatus | null>(null)
  const [railOpen, setRailOpen] = useState(false)
  const [openSpan, setOpenSpan] = useState<Span | null>(null)
  const { episode, ask, stop, reset, load } = useEpisode()
  const [conversations, setConversations] = useState<Conversation[]>(() => listConversations())
  const [activeId, setActiveId] = useState<string | null>(null)
  const [samples, setSamples] = useState<string[]>([])
  const wide = useMediaQuery('(min-width: 1280px)')
  const pollRef = useRef<number | null>(null)

  const refreshRepos = useCallback(async () => {
    const list = await api.listRepos()
    setRepos(list)
    return list
  }, [])

  useEffect(() => {
    void refreshRepos()
      .then((list) => {
        const first = list.find((r) => r.stage === 'ready')
        if (first) setRepoId((cur) => cur ?? first.repo_id)
      })
      .catch((e: Error) => toast.error(`Could not load repositories: ${e.message}`))
    api
      .listProfiles()
      .then((p) => {
        setProfiles(p)
        const base = p.find((x) => x.name === 'qwen4b-base') ?? p.find((x) => x.kind === 'tinker') ?? p[0]
        setProfile((cur) => cur || base?.name || '')
      })
      .catch((e: Error) => toast.error(`Could not load models: ${e.message}`))
  }, [refreshRepos])

  // Poll an indexing job until it settles.
  const watchJob = useCallback(
    (jobId: string) => {
      if (pollRef.current) window.clearInterval(pollRef.current)
      const tick = async () => {
        try {
          const s = await api.repoStatus(jobId)
          setJob(s)
          if (s.stage === 'ready' || s.stage === 'error') {
            if (pollRef.current) window.clearInterval(pollRef.current)
            pollRef.current = null
            await refreshRepos()
            if (s.stage === 'ready') {
              setRepoId(s.repo_id)
              setActiveId(null)
              setOpenSpan(null)
              reset()
              toast.success(`${repoName(s)} is ready`)
            } else {
              toast.error(s.message ?? `Could not index ${repoName(s)}`)
            }
            window.setTimeout(() => setJob(null), 1500)
          }
        } catch (e) {
          if (pollRef.current) window.clearInterval(pollRef.current)
          pollRef.current = null
          setJob(null)
          toast.error((e as Error).message)
        }
      }
      void tick()
      pollRef.current = window.setInterval(tick, 700)
    },
    [refreshRepos, reset],
  )
  useEffect(() => () => {
    if (pollRef.current) window.clearInterval(pollRef.current)
  }, [])

  const addRepo = useCallback(
    async (url: string) => {
      const { job_id } = await api.addRepo(url)
      await refreshRepos()
      watchJob(job_id)
    },
    [refreshRepos, watchJob],
  )

  // Keep finished conversations.
  useEffect(() => {
    if (episode.status !== 'done' && !(episode.status === 'error' && episode.answer)) return
    const saved = saveConversation(episode, activeId ?? undefined)
    setActiveId(saved.id)
    setConversations(listConversations())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episode.status])

  const openConversation = (c: Conversation) => {
    setOpenSpan(null)
    setActiveId(c.id)
    if (c.repoId !== repoId && repos.some((r) => r.repo_id === c.repoId)) setRepoId(c.repoId)
    if (profiles.some((p) => p.name === c.profile)) setProfile(c.profile)
    load(c.episode)
    setRailOpen(false)
  }
  const removeConversation = (id: string) => {
    deleteConversation(id)
    setConversations(listConversations())
    if (id === activeId) {
      setActiveId(null)
      reset()
    }
  }
  const newQuestion = () => {
    setActiveId(null)
    setOpenSpan(null)
    reset()
    setRailOpen(false)
  }

  const selected = repos.find((r) => r.repo_id === repoId) ?? null
  const running = episode.status === 'running'

  // Sample questions come from the repository's index.
  useEffect(() => {
    if (!repoId) return
    let cancelled = false
    setSamples([])
    api.suggestions(repoId).then((q) => {
      if (!cancelled) setSamples(q)
    }).catch(() => {})
    return () => {
      cancelled = true
    }
  }, [repoId])

  const switchRepo = (id: string) => {
    if (id === repoId) return
    setRepoId(id)
    setActiveId(null)
    setOpenSpan(null)
    reset()
  }

  const onAsk = (q: string) => {
    if (!repoId || !profile) return
    setOpenSpan(null)
    setActiveId(null)
    void ask(q, repoId, profile)
  }

  const viewer = selected && openSpan && (
    <FileViewer repoId={selected.repo_id} repoUrl={selected.url} sha={selected.sha} span={openSpan} onClose={() => setOpenSpan(null)} />
  )

  const busy = !!job && job.stage !== 'ready' && job.stage !== 'error'
  const rail = (
    <div className="flex h-full flex-col">
      <div className="p-2">
        <RepoSwitcher
          repos={repos}
          selected={selected}
          onSelect={(id) => {
            switchRepo(id)
            setRailOpen(false)
          }}
          onAdd={addRepo}
          busy={busy}
        />
      </div>
      {job && busy && (
        <div className="border-y">
          <IndexStepper status={job} repoName={repoName(job)} />
        </div>
      )}
      <Conversations
        items={conversations.filter((c) => c.repoId === repoId)}
        activeId={activeId}
        onOpen={openConversation}
        onDelete={removeConversation}
        onNew={newQuestion}
      />
    </div>
  )

  return (
    <TooltipProvider delayDuration={200}>
      <div className="h-dvh overflow-hidden bg-canvas md:grid md:grid-cols-[256px_minmax(0,1fr)]">
        <aside className="hidden h-dvh border-r bg-sidebar md:block">{rail}</aside>

        <div className="flex h-dvh min-w-0 flex-col">
          <header className="flex h-12 shrink-0 items-center gap-2 border-b bg-background px-3">
            <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setRailOpen(true)} aria-label="Repositories">
              <PanelLeft />
            </Button>
            <a href="/" onClick={(e) => { e.preventDefault(); navigate('/') }} className="mr-1 shrink-0 rounded-sm focus-visible:outline-2 focus-visible:outline-ring max-sm:hidden" aria-label="Home">
              <Wordmark />
            </a>
            <span className="h-4 w-px bg-border max-sm:hidden" aria-hidden />
            <div className="min-w-0 flex-1">
              {selected ? (
                <p className="truncate text-sm">
                  {repoName(selected)} <span className="font-mono text-xs text-muted-foreground">{selected.sha.slice(0, 7)}</span>
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">No repository selected</p>
              )}
            </div>
            {IS_MOCK && (
              <span className="rounded-full border border-dashed px-2 py-0.5 text-xs text-muted-foreground max-sm:hidden" title="No backend configured; replaying a recorded episode">
                Sample data
              </span>
            )}
            <Button variant="ghost" size="sm" onClick={() => navigate('/compare')} className="max-sm:hidden">
              Compare
            </Button>
            <ModelPicker profiles={profiles} value={profile} onChange={setProfile} disabled={running} />
            <ThemeToggle />
          </header>

          <div className="flex min-h-0 flex-1">
            <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
              {episode.status === 'idle' ? (
                <div className="mx-auto flex w-full max-w-[720px] flex-1 flex-col justify-center px-4 py-10 sm:px-6">
                  <h1 className="mb-1 text-center text-[22px] leading-tight font-medium tracking-tight text-balance">
                    {selected ? `Ask about ${repoName(selected)}` : 'Pick a repository to begin'}
                  </h1>
                  <p className="mx-auto mb-6 max-w-[48ch] text-center text-sm text-muted-foreground">
                    {selected
                      ? 'The agent reads the code, then answers with a citation for every claim, verified against the lines it read.'
                      : 'Use the switcher at the top left, or add one from a GitHub URL.'}
                  </p>
                  <QuestionBox onAsk={onAsk} onStop={stop} running={running} disabled={!selected} disabledReason="Pick a repository to ask about" samples={samples} />
                </div>
              ) : (
                <div className="mx-auto w-full max-w-[720px] px-4 pt-8 pb-16 sm:px-6">
                  <div>
                    <div className="mb-6 flex items-start justify-between gap-4">
                      <p className="text-[17px] leading-snug font-medium">{episode.question}</p>
                      {running ? (
                        <Button variant="outline" size="sm" onClick={stop} className="shrink-0">
                          <Square className="size-3 fill-current" />
                          Stop
                        </Button>
                      ) : (
                        <Button variant="outline" size="sm" onClick={newQuestion} className="shrink-0">
                          <MessageSquarePlus />
                          Ask another
                        </Button>
                      )}
                    </div>
                    <Ledger rows={episode.rows} running={running} budget={TOOL_BUDGET} onOpen={setOpenSpan} />
                    <AnswerPanel episode={episode} onOpen={setOpenSpan} />
                    {episode.status === 'error' && (
                      <div className="mt-6 border-t pt-3 text-sm">
                        <p>The agent stopped before answering.</p>
                        <p className="mt-0.5 text-xs text-muted-foreground">{episode.error}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </main>

            {wide && viewer && <aside className="w-[44%] max-w-[760px] shrink-0 border-l bg-background">{viewer}</aside>}
          </div>
        </div>

        <Sheet open={railOpen} onOpenChange={setRailOpen}>
          <SheetContent side="left" className="w-[300px] p-0" showCloseButton={false}>
            <SheetTitle className="sr-only">Repositories</SheetTitle>
            {rail}
          </SheetContent>
        </Sheet>

        <Sheet open={!wide && !!openSpan} onOpenChange={(o) => !o && setOpenSpan(null)}>
          <SheetContent side="bottom" className="h-[85dvh] p-0 sm:h-[80dvh]" showCloseButton={false}>
            <SheetTitle className="sr-only">File</SheetTitle>
            {viewer}
          </SheetContent>
        </Sheet>
      </div>
      <Toaster position="bottom-right" />
    </TooltipProvider>
  )
}
