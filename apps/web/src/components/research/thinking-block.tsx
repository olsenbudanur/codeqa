import { Brain, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

// One step's thinking, folded to a preview line; a button unravels it.
export function ThinkingBlock({ text, open, onToggle, compact }: { text: string; open: boolean; onToggle: () => void; compact?: boolean }) {
  const words = text.trim().split(/\s+/).filter(Boolean).length
  const preview = text.trim().replace(/\s+/g, ' ')
  return (
    <div className={cn('rounded-md border border-dashed', open ? 'bg-muted/40' : '', compact ? 'text-[12px]' : 'text-[12.5px]')}>
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-muted-foreground hover:text-foreground"
      >
        <Brain className="size-3.5 shrink-0 text-verified" aria-hidden />
        <span className="shrink-0 font-mono text-[11px]">Thinking</span>
        {!open && <span className="min-w-0 flex-1 truncate">{preview}</span>}
        {open && <span className="min-w-0 flex-1 font-mono text-[11px]">{words} words</span>}
        <ChevronRight className={cn('size-3.5 shrink-0 transition-transform', open && 'rotate-90')} aria-hidden />
      </button>
      {open && <p className="border-t border-dashed px-3 py-2.5 leading-5 whitespace-pre-wrap text-foreground/90">{text.trim()}</p>}
    </div>
  )
}

export function ThinkingToggleAll({ count, allOpen, onToggle }: { count: number; allOpen: boolean; onToggle: () => void }) {
  if (count === 0) return null
  return (
    <button type="button" onClick={onToggle} className="inline-flex items-center gap-1 font-mono text-[11px] text-muted-foreground hover:text-foreground" aria-pressed={allOpen}>
      <Brain className="size-3" aria-hidden />
      {allOpen ? 'hide thinking' : `show all thinking (${count})`}
    </button>
  )
}
