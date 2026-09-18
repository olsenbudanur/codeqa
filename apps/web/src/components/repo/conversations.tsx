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
        <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-2" aria-label="Past conversations">
          {items.map((c) => {
            const isActive = c.id === activeId
            return (
              <li key={c.id} className="group relative">
                <button
                  type="button"
                  onClick={() => onOpen(c)}
                  className={cn(
                    'flex w-full flex-col gap-0.5 rounded-md py-1.5 pr-7 pl-2 text-left transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring',
                    isActive && 'bg-accent',
                  )}
                >
                  <span className="line-clamp-2 text-[13px] leading-5">{c.question}</span>
                  <span className="truncate font-mono text-[11px] text-muted-foreground">
                    {c.profile}, {when(c.createdAt)}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(c.id)}
                  aria-label="Delete conversation"
                  className="absolute top-1.5 right-1 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-background hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100"
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
