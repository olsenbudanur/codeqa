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
  return markdown.replace(CITATION_RE, (whole, path: string, a: string, b?: string) => {
    const end = b ?? a
    return `[${whole.slice(1, -1)}](cite:${encodeURIComponent(path)}:${a}:${end})`
  })
}

export function parseCiteHref(href: string): Span | null {
  if (!href.startsWith('cite:')) return null
  const [path, start, end] = href.slice(5).split(':')
  return { path: decodeURIComponent(path), start: Number(start), end: Number(end) }
}
