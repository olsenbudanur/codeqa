import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, ArrowUp, RotateCcw } from 'lucide-react'
import { navigate } from '@/lib/router'
import { useReplay } from '@/hooks/use-replay'
import type { Span } from '@/lib/contracts'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Wordmark } from '@/components/wordmark'
import { Transcript } from '@/components/chat/transcript'
import { CitationChip } from '@/components/answer/citation-chip'
import './home.css'

const HEADLINE = 'Ask a codebase anything. Every answer cites the lines it read'
const HERO_SPAN: Span = { path: 'src/flask/app.py', start: 81, end: 85 }

const STEPS: [string, string][] = [
  ['Index', 'a repository is snapshotted and mapped in seconds'],
  ['Research', 'five read-only tools, a few line ranges, nothing executes'],
  ['Answer', 'every claim carries a file and line range'],
  ['Verify', 'each citation is checked against the lines actually read'],
]

// A simple page for a simple product: what it is, the product doing it, how to start.
export function Home() {
  const { episode, play, played } = useReplay()
  const open = () => navigate('/app')
  const wordsDone = 80 + HEADLINE.split(' ').length * 45

  return (
    <TooltipProvider delayDuration={200}>
      <div className="landing">
        <div className="atmosphere" aria-hidden>
          <div className="grid" />
          <div className="fold a" />
          <div className="fold b" />
          <div className="bloom" />
        </div>

        <div className="flex min-h-dvh flex-col">
          <header className="mx-auto flex h-[72px] w-full max-w-[1200px] shrink-0 items-center gap-8 px-5 short:h-16 sm:px-8">
            <Wordmark />
            <div className="ml-auto flex items-center gap-2">
              <Button className="h-10 px-4 text-[14.5px]" onClick={open}>
                Open the workbench
                <ArrowRight />
              </Button>
            </div>
          </header>

          <section className="mx-auto flex w-full max-w-[1200px] flex-1 flex-col -translate-y-4 items-center justify-center px-5 pt-0 pb-20 text-center sm:px-8 short:translate-y-0 short:pb-8">
            <p className="rise mb-7 inline-flex items-center gap-2 rounded-full border border-border bg-background/60 px-3 py-1 font-mono text-[12px] text-muted-foreground backdrop-blur short:mb-4" style={{ '--d': '0ms' } as React.CSSProperties}>
              <span className="size-1.5 rounded-full bg-verified" aria-hidden />
              <span className="max-sm:hidden">Qwen3.5-4B, trained with reinforcement learning on Tinker</span>
              <span className="sm:hidden">Qwen3.5-4B, trained with RL on Tinker</span>
            </p>
            <h1 className="display max-w-[34ch] text-[2.5rem] leading-[1.02] text-balance sm:text-[3.3rem] lg:text-[3.6rem] short:lg:text-[3rem]" aria-label={`${HEADLINE}.`}>
              <Words text={HEADLINE} />
              <HeroCite delay={wordsDone + 350} onOpen={open} />
            </h1>
            <p className="rise mt-6 max-w-[80ch] text-[16px] leading-7 text-muted-foreground sm:text-[17px] short:mt-3 short:text-[15px] short:leading-6" style={{ '--d': `${wordsDone - 100}ms` } as React.CSSProperties}>
              A research agent that indexes a repository, reads a few line ranges, and answers with a citation on every claim.
              Each citation is verified against what it actually read.
            </p>

            <div className="rise mt-10 w-full max-w-[980px] short:mt-5" style={{ '--d': `${wordsDone + 250}ms` } as React.CSSProperties}>
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

            <ol className="rise mt-9 grid w-full max-w-[980px] gap-x-8 gap-y-2 text-left text-[13px] sm:grid-cols-4 short:mt-4" style={{ '--d': `${wordsDone + 500}ms` } as React.CSSProperties}>
              {STEPS.map(([t, body], i) => (
                <li key={t} className="flex gap-2">
                  <span className="font-mono text-verified tabular-nums">{i + 1}</span>
                  <span>
                    <span className="font-medium">{t}</span>
                    <span className="text-muted-foreground">: {body}</span>
                  </span>
                </li>
              ))}
            </ol>
          </section>

          <footer className="mx-auto flex w-full max-w-[1200px] flex-wrap items-center justify-center gap-x-6 gap-y-2 px-5 py-4 font-mono text-[11.5px] text-muted-foreground sm:px-8">
            <span>Qwen3.5-4B on Tinker</span>
            <span>Read-only tools, no code execution</span>
          </footer>
        </div>
      </div>
    </TooltipProvider>
  )
}

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
