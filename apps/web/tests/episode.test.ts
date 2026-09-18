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
