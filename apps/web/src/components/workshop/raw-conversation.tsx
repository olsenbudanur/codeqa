import { useState } from 'react'
import { ChevronRight, Copy } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Chip } from './ui'

export interface RawMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  thinking?: string | null
  tool_calls?: { name: string; args: Record<string, unknown>; call_id?: string | null }[]
  name?: string | null
  call_id?: string | null
  parse_error?: string | null
  usage?: Record<string, number | null | undefined>
  reward?: number | null
}

const ROLE: Record<RawMessage['role'], { label: string; cls: string }> = {
  system: { label: 'system', cls: 'border-border text-muted-foreground' },
  user: { label: 'user', cls: 'border-series-1 text-foreground' },
  assistant: { label: 'assistant', cls: 'border-verified/50 text-verified' },
  tool: { label: 'tool', cls: 'border-unverified/50 text-unverified' },
}

const fmt = new Intl.NumberFormat('en-US')

// Every message the model saw, in order, untruncated. Long prompts start folded.
export function RawConversation({ messages, note }: { messages: RawMessage[]; note?: string }) {
  const [open, setOpen] = useState<Set<number>>(() => new Set(messages.map((m, i) => (m.content.length > 1500 && (m.role === 'system' || m.role === 'user' || m.role === 'tool') ? -1 : i)).filter((i) => i >= 0)))
  const [allOpen, setAllOpen] = useState(false)
  const toggle = (i: number) => setOpen((s) => { const n = new Set(s); if (n.has(i)) n.delete(i); else n.add(i); return n })
  const total = messages.reduce((a, m) => a + m.content.length + (m.thinking?.length ?? 0), 0)
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-3 font-mono text-[11.5px] text-muted-foreground">
        <span>{messages.length} messages, {fmt.format(total)} characters</span>
        <button type="button" className="hover:text-foreground" onClick={() => setAllOpen((v) => !v)}>{allOpen ? 'fold long ones' : 'unfold everything'}</button>
        <button type="button" className="inline-flex items-center gap-1 hover:text-foreground" onClick={() => navigator.clipboard?.writeText(JSON.stringify(messages, null, 1)).catch(() => {})}>
          <Copy className="size-3" /> copy as JSON
        </button>
      </div>
      {note && <p className="mb-3 rounded-md border border-dashed px-3 py-2 text-[12px] text-muted-foreground">{note}</p>}
      <ol className="space-y-2">
        {messages.map((m, i) => {
          const isOpen = allOpen || open.has(i)
          const role = ROLE[m.role]
          const usage = m.usage ?? {}
          return (
            <li key={i} className="rounded-lg border bg-background">
              <button type="button" onClick={() => toggle(i)} aria-expanded={isOpen} className="flex w-full items-center gap-2 px-3 py-2 text-left">
                <ChevronRight className={cn('size-3.5 shrink-0 text-muted-foreground transition-transform', isOpen && 'rotate-90')} aria-hidden />
                <span className={cn('rounded-md border px-1.5 py-0.5 font-mono text-[11px]', role.cls)}>{role.label}{m.role === 'tool' && m.name ? ` · ${m.name}` : ''}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-muted-foreground">
                  {m.role === 'assistant' && m.tool_calls?.length ? `${m.tool_calls.length} tool call${m.tool_calls.length > 1 ? 's' : ''}: ${m.tool_calls.map((c) => c.name).join(', ')}` : m.content.replace(/\s+/g, ' ').slice(0, 120)}
                </span>
                <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                  {fmt.format(m.content.length + (m.thinking?.length ?? 0))} ch
                  {usage.prompt_tokens != null && ` · ${fmt.format(usage.prompt_tokens)} in`}
                  {usage.completion_tokens != null && ` · ${fmt.format(usage.completion_tokens)} out`}
                </span>
              </button>
              {isOpen && (
                <div className="border-t px-3 py-3">
                  {m.parse_error && <p className="mb-2 text-[12px] text-destructive">parse error: {m.parse_error}</p>}
                  {m.thinking && (
                    <details className="mb-3 rounded-md border border-dashed px-2.5 py-1.5" open>
                      <summary className="cursor-pointer font-mono text-[11px] text-muted-foreground">thinking · {fmt.format(m.thinking.length)} ch</summary>
                      <pre className="mt-2 font-mono text-[12px] leading-5 whitespace-pre-wrap text-foreground/90">{m.thinking}</pre>
                    </details>
                  )}
                  {m.content && <pre className="font-mono text-[12px] leading-5 whitespace-pre-wrap">{m.content}</pre>}
                  {m.tool_calls && m.tool_calls.length > 0 && (
                    <div className={cn('space-y-1', m.content && 'mt-3')}>
                      {m.tool_calls.map((c, k) => (
                        <pre key={k} className="rounded-md bg-muted px-2.5 py-1.5 font-mono text-[12px] leading-5 whitespace-pre-wrap">
                          {c.name}({JSON.stringify(c.args)})
                        </pre>
                      ))}
                    </div>
                  )}
                  {m.reward !== undefined && m.reward !== null && <p className="mt-2"><Chip tone={m.reward > 0 ? 'good' : 'default'}>reward {m.reward}</Chip></p>}
                </div>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
