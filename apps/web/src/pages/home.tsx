import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ArrowRight, ArrowUp, Check, CircleDashed, Plus, RotateCcw } from 'lucide-react'
import { navigate } from '@/lib/router'
import { useReplay } from '@/hooks/use-replay'
import { CITATION_RE, type CitationItem, type RepoJobStatus, type SSEEvent, type Span } from '@/lib/contracts'
import { formatRange } from '@/lib/citations'
import { cn } from '@/lib/utils'
import { emptyEpisode, episodeReducer, type Episode } from '@/state/episode'
import { Transcript } from '@/components/chat/transcript'
import mockEvents from '../../mock/events.json'
import { Button } from '@/components/ui/button'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Wordmark } from '@/components/wordmark'
import { Ledger } from '@/components/research/ledger'
import { CitedMarkdown } from '@/components/answer/answer-panel'
import { CitationChip, type Verdict } from '@/components/answer/citation-chip'
import { IndexStepper } from '@/components/repo/index-stepper'
import './home.css'

const HEADLINE = 'Ask a codebase anything. Every answer cites the lines it read'
const HERO_SPAN: Span = { path: 'src/flask/app.py', start: 81, end: 85 }

const EVENTS = mockEvents.events as SSEEvent[]
const ANSWER = (EVENTS.find((e) => e.type === 'answer') as { markdown: string }).markdown
const CITATIONS = (EVENTS.find((e) => e.type === 'citations') as { items: CitationItem[] }).items
const ANSWER_BODY = ANSWER.split(/^Sources:\s*$/m)[0].trim()

const SPEC: [string, string][] = [
  ['Model', 'Qwen3.5-4B, thinking on'],
  ['Training', 'Reinforcement learning on Tinker, one trajectory per question'],
  ['Reward', 'Correctness × efficiency. Bad format or an unread citation scores zero'],
  ['Teacher', 'Claude Sonnet 5, same environment, same tools'],
  ['Tools', 'overview, find_symbol, grep, read_file, list_dir. All read-only'],
  ['Execution', 'None. The agent reads a snapshot; nothing runs'],
  ['Budget', 'A fixed number of tool calls per question, line ranges instead of files'],
  ['Environment', 'One class, three drivers: trainer, teacher, product'],
]

const FAQ = [
  { q: 'Does it run my code?', a: 'No. It works over a snapshot with five read-only tools: overview, find symbol, grep, read lines and list directory. Nothing executes, and there is no sandbox to escape.' },
  { q: 'What does “verified” mean on a citation?', a: 'The cited range falls inside a range the agent read during this session, using the same check the training reward uses. It does not mean the claim is true; it means you can open the lines and check in one click.' },
  { q: 'Which model answers?', a: 'Qwen3.5-4B, trained with reinforcement learning on Tinker. You can switch to the untrained base model, or to Claude, to compare answers on the same question.' },
  { q: 'How was it trained?', a: 'Reward is correctness times efficiency. Malformed answers and citations to unread lines score zero. Fewer tool calls and smaller prompts score higher, so it learns to look in the right place first.' },
  { q: 'Which repositories work?', a: 'Any public GitHub repository. Paste a URL and a mid-size repository is askable in about a minute; directory summaries fill in afterwards.' },
  { q: 'Is this the same agent as in training?', a: 'Yes. One environment class runs the trainer, the teacher and this product. Only the model behind it changes.' },
]

const SOURCES: [string, string][] = [
  ['mock/events.json', 'the sample episode: 3 tool calls, 9.8 s, 1,654 prompt tokens, 3 citations of which 2 verified'],
  ['data/smoke_episode3.log', 'the recorded run the sample is transcribed from'],
  ['docs/agent_design.md', 'tools, budget, answer format, and the one-environment rule'],
  ['docs/gap_specs.md §3', 'the reward gates: format, citations exist, citations grounded'],
]

export function Home() {
  const { episode, play, played } = useReplay()
  const open = () => navigate('/app')
  useReveal()
  const wordsDone = 80 + HEADLINE.split(' ').length * 45

  return (
    <TooltipProvider delayDuration={200}>
      <div className="landing">
        <div className="atmosphere" aria-hidden>
          <div className="grid" />
          <div className="fold a" />
          <div className="fold b" />
          <div className="fold c" />
          <div className="bloom" />
        </div>

        <div className="flex min-h-dvh flex-col">
        <header className="mx-auto flex h-[72px] short:h-16 w-full max-w-[1200px] shrink-0 items-center gap-8 px-5 sm:px-8">
          <Wordmark />
          <nav className="hidden items-center gap-7 text-[14.5px] text-muted-foreground md:flex" aria-label="Sections">
            <a href="#how" className="transition-colors hover:text-foreground">How it works</a>
            <a href="#spec" className="transition-colors hover:text-foreground">Specification</a>
                      </nav>
          <div className="ml-auto flex items-center gap-2">
            <Button asChild variant="ghost" className="h-10 px-3.5 text-[14.5px] text-muted-foreground hover:text-foreground max-md:hidden">
              <a href="#faq">Questions</a>
            </Button>
            <Button className="h-10 px-4 text-[14.5px]" onClick={open}>
              Open the workbench
              <ArrowRight />
            </Button>
          </div>
        </header>

        {/* Hero */}
        <section className="mx-auto flex w-full max-w-[1200px] flex-1 flex-col items-center justify-center px-5 pt-6 pb-8 text-center sm:px-8">
          <p className="rise mb-5 inline-flex items-center gap-2 rounded-full short:mb-3 border border-border bg-background/60 px-3 py-1 font-mono text-[12px] text-muted-foreground backdrop-blur" style={{ '--d': '0ms' } as React.CSSProperties}>
            <span className="size-1.5 rounded-full bg-verified" aria-hidden />
            <span className="max-sm:hidden">Qwen3.5-4B, trained with reinforcement learning on Tinker</span>
            <span className="sm:hidden">Qwen3.5-4B, trained with RL on Tinker</span>
          </p>
          <h1 className="display max-w-[22ch] text-[2.5rem] leading-[1.02] text-balance sm:text-[3.3rem] lg:text-[3.6rem] short:lg:text-[3rem]" aria-label={`${HEADLINE}.`}>
            <Words text={HEADLINE} />
            <HeroCite delay={wordsDone + 350} onOpen={open} />
          </h1>
          <p className="rise mt-4 max-w-[62ch] text-[16px] leading-7 text-muted-foreground sm:text-[17px] short:mt-3 short:text-[15px] short:leading-6" style={{ '--d': `${wordsDone - 100}ms` } as React.CSSProperties}>
            A research agent that indexes a repository, reads a few line ranges, and answers with a citation on every claim.
            Each citation is verified against what it actually read.
          </p>
          <div className="rise mt-6 flex flex-wrap items-center justify-center gap-3 short:mt-4" style={{ '--d': `${wordsDone + 50}ms` } as React.CSSProperties}>
            <Button size="lg" className="h-11 px-5 text-[15px]" onClick={open}>
              Ask about a repository
              <ArrowRight />
            </Button>
            <Button asChild size="lg" variant="outline" className="h-11 border-border bg-transparent px-4 text-[15px] hover:bg-accent">
              <a href="#how">See how it works</a>
            </Button>
          </div>

          <div className="rise mt-8 w-full max-w-[880px] short:mt-5" style={{ '--d': `${wordsDone + 250}ms` } as React.CSSProperties}>
            <div className="frame glow bracket bracket-in">
              <Brackets />
              <div className="frame-inner">
                <div className="flex items-center gap-3 border-b px-4 py-2.5 font-mono text-[11.5px] text-muted-foreground">
                  <span className="flex items-center gap-1.5">
                    <span className={cn('size-1.5 rounded-full bg-verified', episode.status === 'running' && 'animate-pulse')} aria-hidden />
                    Sample episode
                  </span>
                  <span className="max-sm:hidden">pallets/flask 85c5d93</span>
                  <span className="ml-auto">qwen4b-run1</span>
                  {played && (
                    <button type="button" onClick={() => void play()} className="rounded p-0.5 text-foreground hover:bg-accent" aria-label="Replay">
                      <RotateCcw className="size-3.5" />
                    </button>
                  )}
                </div>
                <Transcript episode={episode} onOpen={open} className="h-[236px] text-left sm:h-[252px] short:sm:h-[196px]" />
                <button
                  type="button"
                  onClick={open}
                  className="flex w-full items-center gap-3 border-t border-border px-4 py-3 text-left text-[14px] text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground sm:px-5"
                >
                  <span className="flex-1">Ask about pallets/flask, or paste a repository URL</span>
                  <span className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground" aria-hidden>
                    <ArrowUp className="size-3.5" />
                  </span>
                </button>
              </div>
            </div>
          </div>

        </section>
        </div>

        {/* Before / after */}
        <section className="rule mt-20 lg:mt-28" data-reveal>
          <div className="rule-body mx-auto max-w-[1200px] px-5 py-24 sm:px-8 lg:py-32">
            <div className="mx-auto max-w-[44rem] text-center">
              <p className="font-mono text-[12px] text-verified">Verified, not trusted</p>
              <h2 className="display mt-4 text-[2.1rem] leading-[1.05] text-balance sm:text-[2.9rem]">A confident answer with no source is a guess.</h2>
              <p className="mx-auto mt-5 max-w-[56ch] text-[16px] leading-7 text-muted-foreground sm:text-[17px]">
                Same question, same model. The difference is whether you can check it. One citation below points at lines the
                agent never read, so it is flagged instead of trusted.
              </p>
            </div>
            <BeforeAfter onOpen={open} />
          </div>
        </section>

        {/* How it works */}
        <section id="how" className="rule scroll-mt-4" data-reveal>
          <div className="rule-body mx-auto max-w-[1200px] px-5 py-24 sm:px-8 lg:py-32">
            <HowItWorks onOpen={open} />
          </div>
        </section>

        {/* Spec */}
        <section id="spec" className="rule scroll-mt-4" data-reveal>
          <div className="rule-body mx-auto grid max-w-[1200px] gap-12 px-5 py-24 sm:px-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:py-32">
            <div>
              <p className="font-mono text-[12px] text-verified">Specification</p>
              <h2 className="display mt-4 max-w-[18ch] text-[2.1rem] leading-[1.05] text-balance sm:text-[2.9rem]">Built so the receipts are the point, not a footnote.</h2>
              <p className="mt-5 max-w-[42ch] text-[16px] leading-7 text-muted-foreground">
                A small model can be good at this if the reward says exactly what good means. Here it means: right, grounded, and
                found in as few reads as possible. The product runs the same environment the model trained in.
              </p>
            </div>
            <div className="frame">
              <dl className="frame-inner divide-y divide-border px-5">
                {SPEC.map(([k, v]) => (
                  <div key={k} className="grid grid-cols-[7.5rem_minmax(0,1fr)] gap-4 py-4 sm:grid-cols-[9rem_minmax(0,1fr)]">
                    <dt className="font-mono text-[12px] text-muted-foreground">{k}</dt>
                    <dd className="text-[15px] leading-6">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </section>

        {/* FAQ */}
        <section id="faq" className="rule scroll-mt-4" data-reveal>
          <div className="rule-body mx-auto grid max-w-[1200px] gap-12 px-5 py-24 sm:px-8 lg:grid-cols-[minmax(0,4fr)_minmax(0,8fr)] lg:py-32">
            <div>
              <p className="font-mono text-[12px] text-verified">Questions</p>
              <h2 className="display mt-4 text-[2.1rem] leading-[1.05] sm:text-[2.9rem]">Before you paste a URL</h2>
            </div>
            <dl className="divide-y divide-border border-y border-border">
              {FAQ.map((f) => (
                <details key={f.q} className="group">
                  <summary className="flex items-center justify-between gap-6 py-5 text-[17px] font-medium tracking-tight">
                    <dt>{f.q}</dt>
                    <Plus className="faq-plus size-4 shrink-0 text-muted-foreground" aria-hidden />
                  </summary>
                  <dd className="max-w-[62ch] pb-6 text-[15px] leading-7 text-muted-foreground">{f.a}</dd>
                </details>
              ))}
            </dl>
          </div>
        </section>

        {/* Final CTA */}
        <section className="rule" data-reveal>
          <div className="rule-body relative mx-auto flex max-w-[1200px] flex-col items-center px-5 py-28 text-center sm:px-8 lg:py-36">
            <div className="absolute inset-x-0 bottom-0 -z-10 h-[420px] bg-[radial-gradient(ellipse_at_50%_100%,rgba(var(--glow),0.18),transparent_65%)]" aria-hidden />
            <h2 className="display max-w-[22ch] text-[2.1rem] leading-[1.05] text-balance sm:text-[2.9rem]">
              Point it at a repository and ask what you would ask a maintainer.
            </h2>
            <Button size="lg" className="mt-9 h-11 px-5 text-[15px]" onClick={open}>
              Open the workbench
              <ArrowRight />
            </Button>
          </div>
        </section>

        {/* The page cites its own numbers, in the agent's answer format. */}
        <footer className="border-t border-border">
          <div className="mx-auto max-w-[1200px] px-5 py-10 font-mono text-[12px] text-muted-foreground sm:px-8">
            <p className="text-foreground">Sources:</p>
            <ul className="mt-2 space-y-1">
              {SOURCES.map(([path, what]) => (
                <li key={path} className="grid gap-x-4 sm:grid-cols-[15rem_minmax(0,1fr)]">
                  <span className="text-foreground">- {path}</span>
                  <span>{what}</span>
                </li>
              ))}
            </ul>
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border pt-6">
              <Wordmark />
              <span>Qwen3.5-4B on Tinker</span>
              <span>Read-only tools, no code execution</span>
              <a href="/app" onClick={(e) => { e.preventDefault(); open() }} className="ml-auto text-foreground hover:underline">
                Workbench
              </a>
            </div>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  )
}

// --- hero -----------------------------------------------------------------

function Words({ text }: { text: string }) {
  const words = text.split(' ')
  return (
    <span aria-hidden>
      {words.map((w, i) => (
        <span key={i}>
          <span className="reveal-word">
            <span style={{ '--d': `${80 + i * 45}ms` } as React.CSSProperties}>{w}</span>
          </span>{' '}
        </span>
      ))}
    </span>
  )
}

// The headline's own citation: shows up after the words, checks itself, and is a real link.
function HeroCite({ delay, onOpen }: { delay: number; onOpen: () => void }) {
  const reduced = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches, [])
  const [stage, setStage] = useState<'hidden' | 'pending' | 'verified'>(reduced ? 'verified' : 'hidden')
  useEffect(() => {
    if (reduced) return
    const a = window.setTimeout(() => setStage('pending'), delay)
    const b = window.setTimeout(() => setStage('verified'), delay + 1100)
    return () => {
      window.clearTimeout(a)
      window.clearTimeout(b)
    }
  }, [delay, reduced])
  return (
    <span className={cn('hero-cite text-[0.42em] font-normal tracking-normal', stage !== 'hidden' && 'show')} aria-hidden>
      <CitationChip span={HERO_SPAN} verdict={stage === 'verified' ? 'verified' : 'pending'} onOpen={onOpen} />
    </span>
  )
}

function Brackets() {
  return (
    <>
      <span className="bracket-corner tl" aria-hidden />
      <span className="bracket-corner tr" aria-hidden />
      <span className="bracket-corner bl" aria-hidden />
      <span className="bracket-corner br" aria-hidden />
    </>
  )
}

// --- before / after -------------------------------------------------------

function verdictFromMock(s: Span): Verdict {
  const c = CITATIONS.find((i) => i.path === s.path && i.start === s.start && i.end === s.end)
  return c ? (c.verified ? 'verified' : 'unverified') : 'pending'
}

function BeforeAfter({ onOpen }: { onOpen: (s: Span) => void }) {
  const bare = useMemo(() => ANSWER_BODY.replace(CITATION_RE, '').replace(/\s+([.,])/g, '$1'), [])
  const [inView, ref] = useInView<HTMLDivElement>()
  return (
    <div ref={ref} className="mt-12 grid gap-5 lg:grid-cols-2">
      <Panel label="A typical answer" tone="muted">
        <CitedMarkdown markdown={bare} verdictFor={verdictFromMock} onOpen={onOpen} className="text-muted-foreground" />
        <p className="mt-5 font-mono text-[11.5px] text-muted-foreground">No sources. Nothing to open.</p>
      </Panel>
      <Panel label="The same answer here" tone="live">
        {/* Mount the cited version only when in view so the stamps play on arrival. */}
        {inView ? (
          <CitedMarkdown markdown={ANSWER_BODY} verdictFor={verdictFromMock} onOpen={onOpen} />
        ) : (
          <CitedMarkdown markdown={bare} verdictFor={verdictFromMock} onOpen={onOpen} />
        )}
        <p className="mt-5 font-mono text-[11.5px]">
          <span className="text-verified">2 verified</span>
          <span className="text-muted-foreground">, </span>
          <span className="text-unverified">1 flagged</span>
          <span className="text-muted-foreground">. Click any citation to open the lines.</span>
        </p>
      </Panel>
    </div>
  )
}

function Panel({ label, tone, children }: { label: string; tone: 'muted' | 'live'; children: ReactNode }) {
  return (
    <div className={tone === 'live' ? 'frame glow bracket' : 'rounded-[14px] border border-dashed border-border'}>
      {tone === 'live' && <Brackets />}
      <div className={tone === 'live' ? 'frame-inner' : ''}>
        <div className="border-b border-border px-5 py-2.5 font-mono text-[11.5px] text-muted-foreground">{label}</div>
        <div className="px-5 py-5">{children}</div>
      </div>
    </div>
  )
}

// --- how it works: the product's own components, stepping through a short sequence --

const STEPS = [
  { title: 'Index', body: 'Paste a GitHub URL. The repository is snapshotted, parsed for symbols and summarised into a map the agent reads first.' },
  { title: 'Research', body: 'Five read-only tools: overview, find symbol, grep, read lines, list directory. Line ranges, not whole files. Nothing executes.' },
  { title: 'Answer', body: 'It answers as soon as the evidence is sufficient. Every factual claim carries a file and line range.' },
  { title: 'Verify', body: 'Each citation is checked against the lines the agent actually read this session. Verified means read, not trusted.' },
]

function HowItWorks({ onOpen }: { onOpen: () => void }) {
  const [inView, ref] = useInView<HTMLDivElement>()
  const [phase, setPhase] = useState(0)
  const [runId, setRunId] = useState(0)
  const reduced = useMemo(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches, [])
  const LAST = 6

  useEffect(() => {
    if (!inView) return
    if (reduced) {
      setPhase(LAST)
      return
    }
    setPhase(0)
    let p = 0
    const t = window.setInterval(() => {
      p += 1
      setPhase(p)
      if (p >= LAST) window.clearInterval(t)
    }, 750)
    return () => window.clearInterval(t)
  }, [inView, runId, reduced])

  return (
    <div ref={ref}>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="font-mono text-[12px] text-verified">How it works</p>
          <h2 className="display mt-4 text-[2.1rem] leading-[1.05] sm:text-[2.9rem]">Four states, all of them visible.</h2>
          <p className="mt-4 max-w-[44ch] text-[16px] leading-7 text-muted-foreground">
            Four states. These are the product's own components, stepping through the sample episode.
          </p>
        </div>
        {phase >= LAST && !reduced && (
          <button type="button" onClick={() => setRunId((n) => n + 1)} className="inline-flex items-center gap-1.5 font-mono text-[12px] text-muted-foreground hover:text-foreground">
            <RotateCcw className="size-3.5" /> Replay
          </button>
        )}
      </div>
      <ol className="mt-12 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {STEPS.map((s, i) => (
          <li key={s.title} className="frame flex flex-col">
            <div className="frame-inner flex h-full flex-col"><div className="h-[168px] overflow-hidden border-b border-border bg-[var(--canvas)]/60 p-4">
              {i === 0 && <IndexIllustration phase={phase} />}
              {i === 1 && <ResearchIllustration phase={phase} onOpen={onOpen} />}
              {i === 2 && <AnswerIllustration phase={phase} onOpen={onOpen} />}
              {i === 3 && <VerifyIllustration phase={phase} />}
            </div>
            <div className="p-5">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-sm text-verified tabular-nums">{i + 1}</span>
                <h3 className="text-[18px] font-medium tracking-tight">{s.title}</h3>
              </div>
              <p className="mt-2 text-[14.5px] leading-6 text-muted-foreground">{s.body}</p>
            </div>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}

function IndexIllustration({ phase }: { phase: number }) {
  const status: RepoJobStatus =
    phase <= 0
      ? { repo_id: 'pallets__flask__85c5d93', stage: 'snapshot', progress: 0.4, seconds: 1 }
      : phase === 1
        ? { repo_id: 'pallets__flask__85c5d93', stage: 'index', progress: 0.6, seconds: 2 }
        : phase === 2
          ? { repo_id: 'pallets__flask__85c5d93', stage: 'summaries', progress: 0.3, seconds: 3 }
          : { repo_id: 'pallets__flask__85c5d93', stage: 'ready', progress: 1, seconds: 3 }
  return (
    <div className="-mx-3 -mt-3">
      <IndexStepper status={status} repoName="pallets/flask" />
    </div>
  )
}

function ResearchIllustration({ phase, onOpen }: { phase: number; onOpen: () => void }) {
  const rows = useMemo(() => {
    // Phases: 1 → first call pending, 2 → result, 3 → second call, 4 → result.
    const upto = phase <= 0 ? 0 : phase === 1 ? 2 : phase === 2 ? 3 : phase === 3 ? 5 : 6
    let s: Episode = episodeReducer(emptyEpisode, { type: 'start', question: '', repoId: '', profile: '' })
    for (const ev of EVENTS.slice(0, upto)) s = episodeReducer(s, { type: 'event', event: ev })
    return s.rows.filter((r) => r.kind === 'call')
  }, [phase])
  return (
    <div className="text-[13px]">
      {rows.length === 0 ? (
        <p className="font-mono text-[11.5px] text-muted-foreground">Reading the repository map</p>
      ) : (
        <Ledger rows={rows} running={phase < 4} onOpen={onOpen} compact />
      )}
    </div>
  )
}

function AnswerIllustration({ phase, onOpen }: { phase: number; onOpen: () => void }) {
  const first = ANSWER_BODY.split('\n\n')[0]
  if (phase < 4) return <p className="font-mono text-[11.5px] text-muted-foreground">Waiting for enough evidence</p>
  return <CitedMarkdown markdown={first} verdictFor={() => (phase >= 5 ? 'verified' : 'pending')} onOpen={onOpen} className="text-[14px] leading-6" />
}

function VerifyIllustration({ phase }: { phase: number }) {
  return (
    <ul className="space-y-2">
      {CITATIONS.map((c, i) => {
        const shown = phase >= 5 + Math.min(i, 1)
        const v: Verdict = !shown ? 'pending' : c.verified ? 'verified' : 'unverified'
        return (
          <li key={`${c.path}:${c.start}`} className="flex items-center gap-2 font-mono text-[12px]">
            <span
              className={cn(
                'flex size-4 shrink-0 items-center justify-center rounded-full border',
                v === 'verified' && 'dot-done border-verified bg-verified text-white',
                v === 'unverified' && 'dot-done border-unverified text-unverified',
                v === 'pending' && 'border-border text-muted-foreground',
              )}
              aria-hidden
            >
              {v === 'verified' ? <Check className="size-2.5" strokeWidth={3} /> : v === 'unverified' ? <CircleDashed className="size-2.5" /> : null}
            </span>
            <span className={cn('truncate', v === 'pending' && 'text-muted-foreground')}>
              {c.path.split('/').slice(-2).join('/')} <span className="text-muted-foreground">{formatRange(c)}</span>
            </span>
            <span className={cn('ml-auto shrink-0', v === 'verified' && 'text-verified', v === 'unverified' && 'text-unverified', v === 'pending' && 'text-muted-foreground')}>
              {v === 'verified' ? 'read' : v === 'unverified' ? 'not read' : 'checking'}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

// --- hooks ------------------------------------------------------------------

function useInView<T extends HTMLElement>(): [boolean, React.RefObject<T | null>] {
  const ref = useRef<T | null>(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || !('IntersectionObserver' in window)) {
      setInView(true)
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setInView(true)
          io.disconnect()
        }
      },
      { threshold: 0.35 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return [inView, ref]
}

function useReveal() {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>('[data-reveal]'))
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) {
      els.forEach((el) => el.classList.add('is-in'))
      return
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('is-in')
            io.unobserve(e.target)
          }
        }
      },
      { threshold: 0.05, rootMargin: '0px 0px -60px 0px' },
    )
    els.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [])
}
