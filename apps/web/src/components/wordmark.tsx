import { cn } from '@/lib/utils'
import { navigate } from '@/lib/router'

// The mark is a citation bracket: the product's whole promise in one glyph.
export function Wordmark({ className, large, home }: { className?: string; large?: boolean; home?: boolean }) {
  const mark = (
    <span className={cn('inline-flex items-center gap-1.5', large ? 'text-base' : 'text-sm', className)}>
      <span className={cn('font-mono text-verified', large ? 'text-lg' : 'text-[15px]')} aria-hidden>
        [L]
      </span>
      <span className="font-medium tracking-tight">Code Q&amp;A</span>
    </span>
  )
  if (!home) return mark
  return (
    <a
      href="/"
      onClick={(e) => {
        e.preventDefault()
        navigate('/')
      }}
      className="rounded-sm focus-visible:outline-2 focus-visible:outline-ring"
      aria-label="Home"
    >
      {mark}
    </a>
  )
}
