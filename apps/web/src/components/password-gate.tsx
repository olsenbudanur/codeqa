import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { ArrowRight } from 'lucide-react'
import { getPassword, setPassword, verifyPassword } from '@/lib/auth'
import { WORKSHOP_BASE } from '@/lib/workshop'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Wordmark } from '@/components/wordmark'

// Everything but the home page sits behind this.
export function PasswordGate({ children }: { children: ReactNode }) {
  const [ok, setOk] = useState(() => !!getPassword())
  useEffect(() => {
    const on = () => setOk(!!getPassword())
    window.addEventListener('codeqa:auth', on)
    return () => window.removeEventListener('codeqa:auth', on)
  }, [])
  return ok ? <>{children}</> : <GateScreen />
}

function GateScreen() {
  const [pw, setPw] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!pw) return
    setBusy(true)
    setError(null)
    try {
      if (await verifyPassword(pw, WORKSHOP_BASE)) setPassword(pw)
      else setError('That is not it.')
    } catch (err) {
      setError(`Could not reach the API: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-canvas px-4">
      <form onSubmit={submit} className="w-full max-w-[360px] rounded-lg border bg-background p-6">
        <Wordmark home />
        <p className="mt-4 text-sm text-muted-foreground">This part needs the password.</p>
        <label htmlFor="pw" className="sr-only">Password</label>
        <Input id="pw" type="password" value={pw} onChange={(e) => setPw(e.target.value)} autoFocus autoComplete="current-password" className="mt-3" disabled={busy} />
        {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
        <Button type="submit" className="mt-3 w-full" disabled={busy || !pw}>
          Continue
          <ArrowRight />
        </Button>
      </form>
    </div>
  )
}
