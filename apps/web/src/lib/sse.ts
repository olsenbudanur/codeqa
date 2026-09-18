// Plain fetch + ReadableStream SSE reader. Yields parsed JSON `data:` payloads.
export async function* readSSE<T>(res: Response, signal?: AbortSignal): AsyncGenerator<T> {
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  try {
    while (true) {
      if (signal?.aborted) return
      const { value, done } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const data = frame
          .split('\n')
          .filter((l) => l.startsWith('data:'))
          .map((l) => l.slice(5).trimStart())
          .join('\n')
        if (data) yield JSON.parse(data) as T
      }
    }
  } finally {
    reader.releaseLock()
  }
}
