import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { ArrowUp, Mic, MicOff, Square } from 'lucide-react'
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
}: {
  onAsk: (q: string) => void
  onStop: () => void
  running: boolean
  disabled: boolean
  disabledReason?: string
  initial?: string
  showSamples?: boolean
  samples?: string[]
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
      <div
        className={cn(
          'relative rounded-lg border bg-background transition-[box-shadow,border-color]',
          'focus-within:border-ring focus-within:ring-2 focus-within:ring-ring/25',
          dictation.listening && 'border-verified ring-2 ring-verified/25',
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
          rows={1}
          placeholder={disabled ? (disabledReason ?? 'Pick a repository first') : 'Ask about this repository'}
          disabled={disabled}
          className="w-full resize-none bg-transparent px-3.5 pt-3 pb-12 text-[15px] leading-6 outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
        />
        <div className="absolute inset-x-2 bottom-2 flex items-center gap-1">
          {dictation.supported && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className={cn('size-8', dictation.listening && 'text-verified')}
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
          <span className="ml-auto text-xs text-muted-foreground max-sm:hidden">Enter to ask</span>
          {running ? (
            <Button type="button" size="icon" variant="outline" className="size-8" onClick={onStop} aria-label="Stop">
              <Square className="size-3.5 fill-current" />
            </Button>
          ) : (
            <Button type="button" size="icon" className="size-8" onClick={submit} disabled={disabled || !text.trim()} aria-label="Ask">
              <ArrowUp />
            </Button>
          )}
        </div>
      </div>
      {showSamples !== false && samples.length > 0 && !text && !running && !disabled && (
        <ul className="mt-2.5 flex flex-wrap gap-1.5" aria-label="Sample questions">
          {samples.map((s) => (
            <li key={s}>
              <button
                type="button"
                onClick={() => {
                  setText(s)
                  ref.current?.focus()
                }}
                className="rounded-full border px-2.5 py-1 text-left text-xs text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground"
              >
                {s}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
