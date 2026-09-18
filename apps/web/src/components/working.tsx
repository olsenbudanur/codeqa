import { cn } from '@/lib/utils'

// Loading treatments that belong to this product: brackets and a scan, not a spinner.
export function Dots({ className }: { className?: string }) {
  return (
    <span className={cn('dots', className)} aria-hidden>
      <span />
      <span />
      <span />
    </span>
  )
}

export function BracketSpinner({ className }: { className?: string }) {
  return (
    <span className={cn('bracket-spin', className)} aria-hidden>
      <span className="l">[</span>
      <span className="mx-px text-foreground">L</span>
      <span className="r">]</span>
    </span>
  )
}

export function ScanLine({ className }: { className?: string }) {
  return <div className={cn('scan-line', className)} role="progressbar" aria-label="Researching" />
}
