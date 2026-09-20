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

describe('attachSources', () => {
  it('moves Sources-list citations onto the prose line that names the file, outside code fences', async () => {
    const { attachSources } = await import('../src/lib/citations')
    const body = ['The entrypoint is `isympy`, defined in `setup.py` at line 44:', '```', "'isympy = isympy:main'", '```', 'Install with pip.'].join('\n')
    const r = attachSources(body, [{ path: 'setup.py', start: 44, end: 44 }, { path: 'doc/src/install.md', start: 13, end: 13 }])
    expect(r.body.split('\n')[0]).toBe('The entrypoint is `isympy`, defined in `setup.py` at line 44 [setup.py:L44]:')
    expect(r.body).not.toContain("'isympy = isympy:main' [")
    expect([...r.attached]).toEqual(['setup.py:44-44'])
  })
})

describe('linkifyCitations and code', () => {
  it('does not turn a citation quoted inside a code span or fence into a link', () => {
    const md = 'Cite as `[path:L10-L20]`. Real: [a.py:L1-L2].\n```\n[b.py:L3]\n```'
    const out = linkifyCitations(md)
    expect(out).toContain('`[path:L10-L20]`')
    expect(out).toContain('[a.py:L1-L2](cite:a.py:1:2)')
    expect(out).toContain('\n[b.py:L3]\n')
  })
})
