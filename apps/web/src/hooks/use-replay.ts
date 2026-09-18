import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { MockApi, MOCK_QUESTION } from '@/lib/api'
import { emptyEpisode, episodeReducer } from '@/state/episode'

// Replays the recorded episode from mock/events.json. Always the mock, never
// the backend: the home page must not fire a real request on load.
export function useReplay(autoplay = true) {
  const [episode, dispatch] = useReducer(episodeReducer, emptyEpisode)
  const [played, setPlayed] = useState(false)
  const ctrl = useRef<AbortController | null>(null)

  const play = useCallback(async () => {
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    dispatch({ type: 'start', question: MOCK_QUESTION, repoId: 'pallets__flask__85c5d93', profile: 'qwen4b-run1' })
    try {
      for await (const event of new MockApi().ask({ repo_id: '', question: '', profile: '' }, c.signal)) {
        if (c.signal.aborted) return
        dispatch({ type: 'event', event })
      }
      setPlayed(true)
    } catch {
      /* aborted */
    }
  }, [])

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (!autoplay) return
    if (reduced) {
      // Show the finished state without the animation.
      dispatch({ type: 'start', question: MOCK_QUESTION, repoId: 'pallets__flask__85c5d93', profile: 'qwen4b-run1' })
      ;(async () => {
        const api = new MockApi()
        for await (const event of api.ask({ repo_id: '', question: '', profile: '' })) dispatch({ type: 'event', event })
        setPlayed(true)
      })()
      return
    }
    const t = window.setTimeout(() => void play(), 250)
    return () => {
      window.clearTimeout(t)
      ctrl.current?.abort()
    }
  }, [autoplay, play])

  return { episode, play, played }
}
