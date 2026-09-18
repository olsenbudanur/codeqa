import { useEffect, useRef, useState } from 'react'
import { ExternalLink, X } from 'lucide-react'
import { EditorState, RangeSetBuilder, StateEffect, StateField, type Extension } from '@codemirror/state'
import { Decoration, EditorView, GutterMarker, gutterLineClass, lineNumbers, type DecorationSet } from '@codemirror/view'
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { tags as t } from '@lezer/highlight'
import { python } from '@codemirror/lang-python'
import { javascript } from '@codemirror/lang-javascript'
import { markdown } from '@codemirror/lang-markdown'
import type { Span } from '@/lib/contracts'
import { api } from '@/lib/api'
import { formatRange } from '@/lib/citations'
import { Button } from '@/components/ui/button'

// --- range highlight -------------------------------------------------------

const setCited = StateEffect.define<Span | null>()

const citedLine = Decoration.line({ class: 'cm-cited' })
const citedGutter = new (class extends GutterMarker {
  elementClass = 'cm-cited-gutter'
})()

const citedField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    for (const e of tr.effects) {
      if (e.is(setCited)) {
        if (!e.value) return Decoration.none
        const b = new RangeSetBuilder<Decoration>()
        const last = Math.min(e.value.end, tr.state.doc.lines)
        for (let n = e.value.start; n <= last; n++) b.add(tr.state.doc.line(n).from, tr.state.doc.line(n).from, citedLine)
        return b.finish()
      }
    }
    return deco.map(tr.changes)
  },
  provide: (f) => EditorView.decorations.from(f),
})

const citedGutterExt = gutterLineClass.compute([citedField], (state) => {
  const b = new RangeSetBuilder<GutterMarker>()
  state.field(citedField).between(0, state.doc.length, (from) => {
    b.add(from, from, citedGutter)
  })
  return b.finish()
})

// Quiet highlighting: structure in ink, literals and comments in the muted tone.
const quiet = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.operatorKeyword], color: 'var(--foreground)', fontWeight: '500' },
  { tag: [t.definition(t.name), t.className, t.function(t.definition(t.name))], color: 'var(--foreground)', fontWeight: '500' },
  { tag: [t.string, t.special(t.string), t.docString], color: 'var(--muted-foreground)' },
  { tag: [t.comment, t.lineComment, t.blockComment], color: 'var(--muted-foreground)', fontStyle: 'italic' },
  { tag: [t.number, t.bool, t.null, t.atom], color: 'var(--foreground)' },
  { tag: t.invalid, color: 'var(--destructive)' },
])

function langFor(path: string): Extension {
  const ext = path.split('.').pop()?.toLowerCase()
  if (ext === 'py' || ext === 'pyi') return python()
  if (ext === 'js' || ext === 'jsx' || ext === 'mjs' || ext === 'cjs') return javascript()
  if (ext === 'ts' || ext === 'tsx') return javascript({ typescript: true, jsx: ext === 'tsx' })
  if (ext === 'md' || ext === 'rst') return markdown()
  return []
}

// --- component ---------------------------------------------------------------

export function FileViewer({
  repoId,
  repoUrl,
  sha,
  span,
  onClose,
}: {
  repoId: string
  repoUrl?: string
  sha?: string
  span: Span
  onClose: () => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadedPath, setLoadedPath] = useState<string | null>(null)

  // Load the file when path changes.
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    api
      .getFile(repoId, span.path)
      .then((text) => {
        if (cancelled || !host.current) return
        view.current?.destroy()
        view.current = new EditorView({
          parent: host.current,
          state: EditorState.create({
            doc: text,
            extensions: [
              lineNumbers(),
              citedField,
              citedGutterExt,
              langFor(span.path),
              syntaxHighlighting(quiet),
              EditorState.readOnly.of(true),
              EditorView.editable.of(false),
              EditorView.lineWrapping,
            ],
          }),
        })
        setLoadedPath(span.path)
        setLoading(false)
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
        setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoId, span.path])

  // Highlight and scroll whenever the span (or the loaded doc) changes.
  useEffect(() => {
    const v = view.current
    if (!v || loadedPath !== span.path) return
    const lines = v.state.doc.lines
    const start = Math.min(Math.max(1, span.start), lines)
    const end = Math.min(Math.max(start, span.end), lines)
    v.dispatch({
      effects: [setCited.of({ ...span, start, end }), EditorView.scrollIntoView(v.state.doc.line(start).from, { y: 'start', yMargin: 72 })],
    })
  }, [span, loadedPath])

  useEffect(() => () => view.current?.destroy(), [])

  const github = repoUrl && sha ? `${repoUrl}/blob/${sha}/${span.path}#L${span.start}-L${span.end}` : null

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <div className="min-w-0 flex-1">
          <p className="truncate font-mono text-[13px]">{span.path}</p>
          <p className="font-mono text-xs text-verified">{formatRange(span)}</p>
        </div>
        {github && (
          <Button asChild variant="ghost" size="icon" aria-label="Open on GitHub">
            <a href={github} target="_blank" rel="noreferrer">
              <ExternalLink />
            </a>
          </Button>
        )}
        <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close file">
          <X />
        </Button>
      </header>
      <div className="relative min-h-0 flex-1 overflow-hidden">
        {loading && <p className="p-3 text-sm text-muted-foreground">Loading {span.path.split('/').pop()}</p>}
        {error && (
          <div className="p-3 text-sm">
            <p>Could not open this file.</p>
            <p className="mt-1 text-xs text-muted-foreground">{error}</p>
          </div>
        )}
        <div ref={host} className="h-full overflow-auto" hidden={loading || !!error} />
      </div>
    </div>
  )
}
