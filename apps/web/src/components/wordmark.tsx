import { cn } from '@/lib/utils'

// The mark is a citation bracket: the product's whole promise in one glyph.
export function Wordmark({ className, large }: { className?: string; large?: boolean }) {
  return (
    <span className={cn('inline-flex items-center gap-1.5', large ? 'text-base' : 'text-sm', className)}>
      <span className={cn('font-mono text-verified', large ? 'text-lg' : 'text-[15px]')} aria-hidden>
        [L]
      </span>
      <span className="font-medium tracking-tight">Code Q&amp;A</span>
    </span>
  )
}
