import { MessageSquarePlus, X } from 'lucide-react'
import type { Conversation } from '@/lib/history'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

function when(ts: number): string {
  const d = Date.now() - ts
  const m = Math.round(d / 60_000)
  if (m < 1) return 'now'
  if (m < 60) return `${m} min`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h`
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function Conversations({
  items,
  activeId,
  onOpen,
  onDelete,
  onNew,
}: {
  items: Conversation[]
  activeId: string | null
  onOpen: (c: Conversation) => void
  onDelete: (id: string) => void
  onNew: () => void
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="px-2 pt-1 pb-2">
        <Button variant="outline" size="sm" className="w-full justify-start" onClick={onNew}>
          <MessageSquarePlus />
          New question
        </Button>
      </div>
      <p className="px-3 pt-2 pb-1 text-xs text-muted-foreground">Conversations</p>
      {items.length === 0 ? (
        <p className="px-3 py-2 text-xs text-muted-foreground">Answers about this repository are kept here.</p>
      ) : (
        <ul className="min-h-0 flex-1 space-y-1.5 overflow-y-auto px-2 pb-2" aria-label="Past conversations">
          {items.map((c) => {
            const isActive = c.id === activeId
            return (
              <li key={c.id} className="group relative">
                <button
                  type="button"
                  onClick={() => onOpen(c)}
                  className={cn(
                    'flex w-full flex-col gap-1 rounded-lg border bg-background py-2 pr-7 pl-2.5 text-left transition-[background-color,border-color] hover:border-foreground/25 hover:bg-accent/60 focus-visible:outline-2 focus-visible:outline-ring',
                    isActive ? 'border-verified/50 bg-verified-soft/40' : 'border-border',
                  )}
                >
                  <span className="line-clamp-2 text-[13px] leading-[1.35]">{c.question}</span>
                  <span className="flex items-center gap-1.5 truncate font-mono text-[11px] text-muted-foreground">
                    <span className={cn('size-1.5 shrink-0 rounded-full', c.episode.answer ? 'bg-verified' : 'bg-unverified')} aria-hidden />
                    {c.profile}, {when(c.createdAt)}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(c.id)}
                  aria-label="Delete conversation"
                  className="absolute top-2 right-1.5 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-background hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <X className="size-3.5" />
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
