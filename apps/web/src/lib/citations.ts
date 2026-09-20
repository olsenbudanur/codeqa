import { CITATION_RE, type Span } from './contracts'

// One parser for the whole app, built on the grader's regex.
export function parseCitations(text: string): Span[] {
  const out: Span[] = []
  for (const m of text.matchAll(CITATION_RE)) {
    const start = Number(m[2])
    const end = m[3] ? Number(m[3]) : start
    out.push({ path: m[1], start, end })
  }
  return out
}

export const spanKey = (s: Span) => `${s.path}:${s.start}-${s.end}`

export function formatRange(s: Span): string {
  return s.start === s.end ? `L${s.start}` : `L${s.start}–${s.end}`
}

// Rewrite `[path:L10-L20]` into a markdown link with a cite: href so the
// markdown renderer can hand it to the citation chip component.
export function linkifyCitations(markdown: string): string {
  // Code spans and fences are left alone: `[path:L10-L20]` quoted as the format is not a citation.
  return markdown
    .split(/(```[\s\S]*?```|`[^`\n]*`)/)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(CITATION_RE, (whole, path: string, a: string, b?: string) => {
            const end = b ?? a
            return `[${whole.slice(1, -1)}](cite:${encodeURIComponent(path)}:${a}:${end})`
          }),
    )
    .join('')
}

export function parseCiteHref(href: string): Span | null {
  if (!href.startsWith('cite:')) return null
  const [path, start, end] = href.slice(5).split(':')
  return { path: decodeURIComponent(path), start: Number(start), end: Number(end) }
}


// Models trained on the citation contract (Scholia) put every citation in a trailing `Sources:` list and keep the
// prose clean. Attach each such citation to the first body line that names its file (path or basename, outside code
// fences) so it renders inline like a Claude answer; citations no line mentions are left for the Sources row.
export function attachSources(body: string, sources: Span[]): { body: string; attached: Set<string> } {
  const lines = body.split('\n')
  const inFence: boolean[] = []
  let fence = false
  for (const l of lines) {
    if (/^\s*```/.test(l)) fence = !fence
    inFence.push(fence || /^\s*```/.test(l))
  }
  const attached = new Set<string>()
  const escape = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  for (const c of sources) {
    const base = c.path.split('/').pop() ?? c.path
    const needles = [c.path, base].filter((n) => n.length >= 4)
    const idx = lines.findIndex((l, i) => !inFence[i] && !/^\s*Sources:/.test(l) && needles.some((n) => new RegExp(`(^|[^\\w/])${escape(n)}([^\\w/]|$)`).test(l)))
    if (idx === -1) continue
    const tag = `[${c.path}:L${c.start}${c.end !== c.start ? `-L${c.end}` : ''}]`
    if (lines[idx].includes(tag)) { attached.add(spanKey(c)); continue }
    // before a trailing colon (the line introduces a code block) or at the end of the line
    lines[idx] = /:\s*$/.test(lines[idx]) ? lines[idx].replace(/:\s*$/, ` ${tag}:`) : `${lines[idx]} ${tag}`
    attached.add(spanKey(c))
  }
  return { body: lines.join('\n'), attached }
}
