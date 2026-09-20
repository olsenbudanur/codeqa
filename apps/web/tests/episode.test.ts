import { describe, expect, it } from 'vitest'
import events from '../mock/events.json'
import type { SSEEvent } from '../src/lib/contracts'
import { emptyEpisode, episodeReducer, filesRead, type Episode } from '../src/state/episode'

function replay(): Episode {
  let s = episodeReducer(emptyEpisode, { type: 'start', question: 'q', repoId: 'r', profile: 'p' })
  for (const event of events.events as SSEEvent[]) s = episodeReducer(s, { type: 'event', event })
  return s
}

describe('episode reducer over the recorded C9 stream', () => {
  it('pairs every tool_result with its tool_call and finishes', () => {
    const s = replay()
    const calls = s.rows.filter((r) => r.kind === 'call')
    expect(calls).toHaveLength(3)
    expect(calls.every((c) => c.kind === 'call' && c.result)).toBe(true)
    expect(s.status).toBe('done')
    expect(s.stats?.tool_calls).toBe(3)
    expect(s.citations).toHaveLength(3)
  })

  it('records the line ranges the agent read', () => {
    expect(filesRead(replay().rows)).toEqual([
      { path: 'src/flask/app.py', start: 1, end: 50 },
      { path: 'src/flask/app.py', start: 81, end: 85 },
    ])
  })

  it('marks the episode failed on an error event', () => {
    let s = episodeReducer(emptyEpisode, { type: 'start', question: 'q', repoId: 'r', profile: 'p' })
    s = episodeReducer(s, { type: 'event', event: { type: 'error', message: 'boom' } })
    expect(s.status).toBe('error')
    expect(s.error).toBe('boom')
  })
})

describe('rounds (bash_v3)', () => {
  it('attaches results first-in first-out when a round has several calls of the same tool', () => {
    let ep = episodeReducer(emptyEpisode, { type: 'start', question: 'q', repoId: 'r', profile: 'p' })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'tool_call', name: 'bash', args: { command: 'ls' }, turn: 1 } })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'tool_call', name: 'bash', args: { command: 'grep -rn x .' }, turn: 1 } })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'tool_result', name: 'bash', summary: 'a.py', chars: 4 } })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'tool_result', name: 'bash', summary: 'a.py:3: x', chars: 9 } })
    const calls = ep.rows.filter((r) => r.kind === 'call')
    expect(calls.map((c) => c.kind === 'call' && c.result?.summary)).toEqual(['a.py', 'a.py:3: x'])
    expect(calls.map((c) => c.kind === 'call' && c.turn)).toEqual([1, 1])
  })

  it('keeps the last budget trailer and records the forced-answer notice as a row', () => {
    let ep = episodeReducer(emptyEpisode, { type: 'start', question: 'q', repoId: 'r', profile: 'p' })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'budget', context_tokens: 12000, context_cap: 32000, messages_left: 20, turn: 4 } })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'notice', kind: 'forced_answer', text: '[Budget exhausted: ...]' } })
    ep = episodeReducer(ep, { type: 'event', event: { type: 'stats', tool_calls: 9, prompt_tokens: 1, completion_tokens: 1, seconds: 3, forced_answer: true } })
    expect(ep.budget).toEqual({ context_tokens: 12000, context_cap: 32000, messages_left: 20, turn: 4 })
    expect(ep.rows.at(-1)).toMatchObject({ kind: 'notice', notice: 'forced_answer' })
    expect(ep.stats?.forced_answer).toBe(true)
  })
})

describe('groupRounds', () => {
  it('folds consecutive same-turn calls into one round and leaves single calls alone', async () => {
    const { groupRounds } = await import('../src/state/episode')
    let ep = episodeReducer(emptyEpisode, { type: 'start', question: 'q', repoId: 'r', profile: 'p' })
    for (const [turn, cmd] of [[1, 'ls'], [1, 'grep a'], [1, 'grep b'], [2, 'cat x']] as [number, string][])
      ep = episodeReducer(ep, { type: 'event', event: { type: 'tool_call', name: 'bash', args: { command: cmd }, turn } })
    const g = groupRounds(ep.rows)
    expect(g.map((r) => r.kind)).toEqual(['round', 'call'])
    expect(g[0].kind === 'round' && g[0].rows.length).toBe(3)
  })
})
