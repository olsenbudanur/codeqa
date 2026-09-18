import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export function Page({ title, children, actions, wide }: { title: ReactNode; children: ReactNode; actions?: ReactNode; wide?: boolean }) {
  return (
    <div className={cn('mx-auto w-full px-4 py-6 sm:px-6', wide ? 'max-w-[1400px]' : 'max-w-[1100px]')}>
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[20px] font-medium tracking-tight">{title}</h1>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  )
}

export function Panel({ title, children, className, aside }: { title?: ReactNode; children: ReactNode; className?: string; aside?: ReactNode }) {
  return (
    <section className={cn('rounded-lg border bg-background', className)}>
      {(title || aside) && (
        <header className="flex items-baseline justify-between gap-3 border-b px-4 py-2.5">
          <h2 className="text-sm font-medium">{title}</h2>
          {aside && <div className="font-mono text-xs text-muted-foreground">{aside}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Loading({ what }: { what: string }) {
  return <p className="px-1 py-6 text-sm text-muted-foreground">Loading {what}</p>
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <div className="rounded-md border border-destructive/30 px-3 py-2 text-sm">
      <p>Could not load this.</p>
      <p className="mt-0.5 font-mono text-xs text-muted-foreground">{error}</p>
    </div>
  )
}

export function Chip({ children, tone = 'default' }: { children: ReactNode; tone?: 'default' | 'good' | 'warn' | 'bad' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md border px-1.5 py-0.5 font-mono text-[11.5px] leading-4',
        tone === 'good' && 'border-verified/40 text-verified',
        tone === 'warn' && 'border-unverified/50 text-unverified',
        tone === 'bad' && 'border-destructive/40 text-destructive',
        tone === 'default' && 'text-muted-foreground',
      )}
    >
      {children}
    </span>
  )
}

export function Stat({ label, value, tone }: { label: string; value: ReactNode; tone?: 'good' | 'warn' | 'bad' }) {
  return (
    <div className="min-w-[7rem]">
      <div className={cn('font-mono text-[20px] leading-none tabular-nums', tone === 'good' && 'text-verified', tone === 'warn' && 'text-unverified', tone === 'bad' && 'text-destructive')}>{value}</div>
      <div className="mt-1.5 text-xs text-muted-foreground">{label}</div>
    </div>
  )
}
