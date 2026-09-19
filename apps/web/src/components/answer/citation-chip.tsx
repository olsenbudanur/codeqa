import { Check, CircleDashed } from 'lucide-react'
import type { Span } from '@/lib/contracts'
import { formatRange } from '@/lib/citations'
import { cn } from '@/lib/utils'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'

export type Verdict = 'verified' | 'unverified' | 'pending'

export function CitationChip({ span, verdict, onOpen, detail }: { span: Span; verdict: Verdict; onOpen: (s: Span) => void; detail?: string }) {
  const file = span.path.split('/').slice(-2).join('/')
  const tip =
    verdict === 'verified'
      ? 'Verified: the agent read these lines'
      : verdict === 'unverified'
        ? 'Not verified: the agent did not read these lines'
        : 'Checking'
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={() => onOpen(span)}
          data-citation={`${span.path}:${span.start}-${span.end}`}
          className={cn(
            'mx-0.5 inline-flex max-w-full items-center gap-1 rounded-md border px-1.5 py-0.5 align-baseline font-mono text-[12px] leading-4',
            'transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring',
            verdict === 'verified' && 'chip-verified border-verified/40 text-verified',
            verdict === 'unverified' && 'border-unverified/50 text-unverified',
            verdict === 'pending' && 'border-border text-muted-foreground',
          )}
        >
          {verdict === 'verified' ? <Check className="size-3" strokeWidth={3} aria-hidden /> : <CircleDashed className="size-3" aria-hidden />}
          <span className="truncate">{file}</span>
          <span className="opacity-70">{formatRange(span)}</span>
          <span className="sr-only">, {tip}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-[360px]">
        <span className="font-mono">{span.path}</span>, {tip}
        {detail && <span className="mt-1 block text-[11.5px] opacity-80">{detail}</span>}
      </TooltipContent>
    </Tooltip>
  )
}
