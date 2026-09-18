import { describe, expect, it } from 'vitest'
import { linkifyCitations, parseCitations, parseCiteHref } from '../src/lib/citations'

describe('citations', () => {
  it('parses single-line and range citations with the contract regex', () => {
    const spans = parseCitations('A [src/a.py:L41-L67] and B [src/b.py:L5].')
    expect(spans).toEqual([
      { path: 'src/a.py', start: 41, end: 67 },
      { path: 'src/b.py', start: 5, end: 5 },
    ])
  })

  it('ignores backticked or malformed citations, like the grader', () => {
    expect(parseCitations('`src/a.py:L81` and [src/a.py L81-L85] and [a.py:81]')).toEqual([])
  })

  it('round-trips a citation through a cite: link', () => {
    const md = linkifyCitations('See [src/a.py:L41-L67].')
    expect(md).toBe('See [src/a.py:L41-L67](cite:src%2Fa.py:41:67).')
    expect(parseCiteHref('cite:src%2Fa.py:41:67')).toEqual({ path: 'src/a.py', start: 41, end: 67 })
    expect(parseCiteHref('https://example.com')).toBeNull()
  })
})
