import { useCallback, useReducer, useRef } from 'react'
import { api } from '@/lib/api'
import { emptyEpisode, episodeReducer, type Episode } from '@/state/episode'

export function useEpisode() {
  const [episode, dispatch] = useReducer(episodeReducer, emptyEpisode)
  const abortRef = useRef<AbortController | null>(null)

  const stop = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    dispatch({ type: 'abort' })
  }, [])

  const ask = useCallback(async (question: string, repoId: string, profile: string) => {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    dispatch({ type: 'start', question, repoId, profile })
    try {
      for await (const event of api.ask({ repo_id: repoId, question, profile }, ctrl.signal)) {
        if (ctrl.signal.aborted) return
        dispatch({ type: 'event', event })
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return
      dispatch({ type: 'event', event: { type: 'error', message: (err as Error).message } })
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null
    }
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    dispatch({ type: 'reset' })
  }, [])

  const load = useCallback((ep: Episode) => {
    abortRef.current?.abort()
    dispatch({ type: 'load', episode: ep })
  }, [])

  return { episode, ask, stop, reset, load }
}
