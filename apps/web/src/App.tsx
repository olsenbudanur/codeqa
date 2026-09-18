import { usePath } from '@/lib/router'
import { Home } from '@/pages/home'
import { Workbench } from '@/pages/workbench'
import { Compare } from '@/pages/compare'
import { Workshop } from '@/pages/workshop'
import { PasswordGate } from '@/components/password-gate'

export default function App() {
  const full = usePath()
  const path = full.split('?')[0]
  if (path === '/') return <Home />
  let page = <Workbench />
  if (path === '/compare') page = <Compare />
  else if (path === '/workshop' || path.startsWith('/workshop/')) page = <Workshop path={path} search={full.includes('?') ? full.slice(full.indexOf('?')) : ''} />
  return <PasswordGate>{page}</PasswordGate>
}
