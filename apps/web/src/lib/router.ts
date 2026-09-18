import { useEffect, useState } from 'react'

// Two routes, no library: "/" is the home page, "/app" the workbench.
export function navigate(path: string) {
  if (window.location.pathname + window.location.search === path) return
  window.history.pushState(null, '', path)
  window.dispatchEvent(new PopStateEvent('popstate'))
}

export function usePath(): string {
  const [path, setPath] = useState(window.location.pathname + window.location.search)
  useEffect(() => {
    const on = () => setPath(window.location.pathname + window.location.search)
    window.addEventListener('popstate', on)
    return () => window.removeEventListener('popstate', on)
  }, [])
  return path
}
