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

import { useEffect, useState } from 'react'

// Seconds since `since`; ticks once a second.
export function useElapsed(since: number | undefined, active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [active])
  return since ? Math.max(0, Math.round((now - since) / 1000)) : 0
}

// What to say while the model has produced nothing yet.
export function waitingText(seconds: number, calls: number): string {
  if (calls > 0) return `researching, ${calls} ${calls === 1 ? 'call' : 'calls'} so far`
  if (seconds < 8) return 'reading the repository map'
  if (seconds < 30) return `waiting for the model, ${seconds} s`
  return `waiting for the model, ${seconds} s. Tinker sampling is queued (training jobs share the key); Claude answers in seconds if you need it now`
}

export function ScanLine({ className }: { className?: string }) {
  return <div className={cn('scan-line', className)} role="progressbar" aria-label="Researching" />
}
