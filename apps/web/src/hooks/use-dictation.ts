import { useCallback, useEffect, useRef, useState } from 'react'

// Web Speech API. Chrome and Safari; Firefox is behind a flag. The caller keeps a text box either way.
type Recognition = {
  lang: string
  interimResults: boolean
  continuous: boolean
  onresult: ((e: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null
  onend: (() => void) | null
  onerror: ((e: { error: string }) => void) | null
  start(): void
  stop(): void
  abort(): void
}

function getCtor(): (new () => Recognition) | null {
  const w = window as unknown as { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

export function useDictation(onText: (final: string, interim: string) => void) {
  const [supported] = useState(() => getCtor() !== null)
  const [listening, setListening] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const recRef = useRef<Recognition | null>(null)
  const cbRef = useRef(onText)
  useEffect(() => {
    cbRef.current = onText
  })

  const stop = useCallback(() => {
    recRef.current?.stop()
  }, [])

  const start = useCallback(() => {
    const Ctor = getCtor()
    if (!Ctor) return
    const rec = new Ctor()
    rec.lang = navigator.language || 'en-US'
    rec.interimResults = true
    rec.continuous = true
    rec.onresult = (e) => {
      let final = ''
      let interim = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i]
        if (r.isFinal) final += r[0].transcript
        else interim += r[0].transcript
      }
      cbRef.current(final, interim)
    }
    rec.onend = () => setListening(false)
    rec.onerror = (e) => {
      setError(e.error === 'not-allowed' ? 'Microphone access was blocked. Type your question instead.' : e.error)
      setListening(false)
    }
    recRef.current = rec
    setError(null)
    setListening(true)
    rec.start()
  }, [])

  useEffect(() => () => recRef.current?.abort(), [])

  return { supported, listening, error, start, stop }
}
