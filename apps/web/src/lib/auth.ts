// Shared-secret gate for the demo. The API checks the same password on every request.
// In mock mode (no API) the check is local so the gate still behaves the same.
const KEY = 'codeqa.password'
const MOCK_PASSWORD = 'Action!'

export function getPassword(): string | null {
  try {
    return localStorage.getItem(KEY)
  } catch {
    return null
  }
}

export function setPassword(pw: string): void {
  try {
    localStorage.setItem(KEY, pw)
  } catch {
    /* private mode */
  }
  window.dispatchEvent(new Event('codeqa:auth'))
}

export function clearPassword(): void {
  try {
    localStorage.removeItem(KEY)
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new Event('codeqa:auth'))
}

export function authHeaders(): Record<string, string> {
  const pw = getPassword()
  return pw ? { authorization: `Bearer ${pw}` } : {}
}

// A 401 from any endpoint means the stored password is wrong or was rotated: drop it and show the gate.
export function handleUnauthorized(res: Response): void {
  if (res.status === 401) clearPassword()
}

export async function verifyPassword(pw: string, apiBase: string): Promise<boolean> {
  if (!apiBase) return pw === MOCK_PASSWORD
  const res = await fetch(`${apiBase}/auth/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password: pw }),
    signal: AbortSignal.timeout(15_000),
  })
  if (res.status === 401) return false
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return true
}
