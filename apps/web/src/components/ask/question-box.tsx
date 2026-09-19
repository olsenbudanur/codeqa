import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { ArrowUp, BookOpen, Crosshair, Hash, List, Mic, MicOff, Route, Square } from 'lucide-react'
import type { Suggestion, TaskType } from '@/lib/contracts'

const KIND_ICON: Record<TaskType, typeof Crosshair> = { locate: Crosshair, value: Hash, enumerate: List, trace: Route, explain: BookOpen }
import { useDictation } from '@/hooks/use-dictation'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

export function QuestionBox({
  onAsk,
  onStop,
  running,
  disabled,
  disabledReason,
  initial,
  showSamples,
  samples = [],
  bare,
  samplesAs = 'chips',
  hideBox,
}: {
  onAsk: (q: string, type?: TaskType) => void
  onStop: () => void
  running: boolean
  disabled: boolean
  disabledReason?: string
  initial?: string
  showSamples?: boolean
  samples?: Suggestion[]
  bare?: boolean
  samplesAs?: 'chips' | 'cards'
  hideBox?: boolean
}) {
  const [text, setText] = useState(initial ?? '')
  const [interim, setInterim] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  const dictation = useDictation((final, partial) => {
    if (final) setText((t) => (t ? t.replace(/\s*$/, ' ') : '') + final.trim())
    setInterim(partial)
  })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [text, interim])

  function submit() {
    const q = text.trim()
    if (!q || disabled || running) return
    if (dictation.listening) dictation.stop()
    setText('')
    setInterim('')
    onAsk(q)
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div>
      {!hideBox && (
      <div
        className={cn(
          'relative bg-background transition-[box-shadow,border-color]',
          bare ? 'rounded-none border-0' : 'rounded-lg border focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/25',
          dictation.listening && (bare ? 'bg-verified-soft/40' : 'border-verified ring-2 ring-verified/25'),
        )}
      >
        <label htmlFor="question" className="sr-only">
          Question
        </label>
        <textarea
          id="question"
          ref={ref}
          value={interim ? `${text}${text ? ' ' : ''}${interim}` : text}
          onChange={(e) => {
            setInterim('')
            setText(e.target.value)
          }}
          onKeyDown={onKey}
          rows={bare ? 2 : 1}
          placeholder={disabled ? (disabledReason ?? 'Pick a repository first') : 'Ask about this repository'}
          disabled={disabled}
          className={cn('w-full resize-none bg-transparent px-4 pt-3.5 pb-14 leading-6 outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed', bare ? 'min-h-[112px] text-[16px]' : 'text-[15px]')}
        />
        <div className="absolute inset-x-2.5 bottom-2.5 flex items-center gap-1.5">
          {dictation.supported && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  className={cn('size-9 rounded-full', dictation.listening && 'border-verified text-verified')}
                  aria-pressed={dictation.listening}
                  aria-label={dictation.listening ? 'Stop dictating' : 'Dictate the question'}
                  disabled={disabled || running}
                  onClick={() => (dictation.listening ? dictation.stop() : dictation.start())}
                >
                  {dictation.listening ? <MicOff /> : <Mic />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{dictation.listening ? 'Stop dictating' : 'Dictate'}</TooltipContent>
            </Tooltip>
          )}
          {dictation.listening && <span className="text-xs text-verified">Listening</span>}
          {dictation.error && <span className="text-xs text-destructive">{dictation.error}</span>}
          <kbd className="ml-auto rounded-md border px-1.5 py-0.5 font-mono text-[10.5px] text-muted-foreground max-sm:hidden">Enter</kbd>
          {running ? (
            <Button type="button" variant="outline" className="h-9 rounded-full px-3.5" onClick={onStop}>
              <Square className="size-3 fill-current" />
              Stop
            </Button>
          ) : (
            <Button type="button" className="h-9 rounded-full pr-3 pl-4 text-[14px]" onClick={submit} disabled={disabled || !text.trim()}>
              Ask
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>
      </div>
      )}
      {showSamples !== false && samples.length > 0 && !text && !running && !disabled && samplesAs === 'cards' && (
        <ul className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5" aria-label="Sample questions">
          {samples.map((s) => {
            const Icon = KIND_ICON[s.type]
            const kind = s.type
            return (
              <li key={s.question}>
                <button
                  type="button"
                  onClick={() => {
                    if (dictation.listening) dictation.stop()
                    setText('')
                    setInterim('')
                    onAsk(s.question, s.type)
                  }}
                  className="group flex h-full w-full flex-col gap-2 rounded-xl border bg-background p-3.5 text-left transition-[transform,border-color,box-shadow] hover:-translate-y-0.5 hover:border-verified/50 hover:shadow-[0_8px_24px_-16px_rgba(0,0,0,0.35)]"
                >
                  <span className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
                    <Icon className="size-3.5 text-verified" aria-hidden />
                    {kind}
                  </span>
                  <span className="text-[13px] leading-5 [overflow-wrap:anywhere]">{s.question}</span>
                  <span className="mt-auto flex items-center gap-1 pt-1 text-[12px] text-muted-foreground group-hover:text-foreground">
                    Ask this <ArrowUp className="size-3 rotate-45" aria-hidden />
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
      {showSamples !== false && samples.length > 0 && !text && !running && !disabled && samplesAs === 'chips' && (
        <ul className={cn('flex flex-wrap gap-1.5', bare ? 'border-t px-3.5 py-3' : 'mt-2.5')} aria-label="Sample questions">
          {bare && <li className="mr-1 self-center text-xs text-muted-foreground">Try</li>}
          {samples.map((s) => {
            const Icon = KIND_ICON[s.type]
            return (
              <li key={s.question}>
                <button
                  type="button"
                  onClick={() => {
                    // A suggestion is a question, not a draft: ask it straight away.
                    if (dictation.listening) dictation.stop()
                    setText('')
                    setInterim('')
                    onAsk(s.question, s.type)
                  }}
                  className="group inline-flex max-w-full items-center gap-1.5 rounded-full border bg-background px-3 py-1.5 text-left text-[12.5px] text-muted-foreground transition-[color,border-color,transform] hover:-translate-y-px hover:border-verified/50 hover:text-foreground"
                >
                  <Icon className="size-3.5 shrink-0 text-verified" aria-hidden />
                  <span className="truncate">{s.question}</span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
