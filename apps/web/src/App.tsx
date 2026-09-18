import { usePath } from '@/lib/router'
import { Home } from '@/pages/home'
import { Workbench } from '@/pages/workbench'
import { Compare } from '@/pages/compare'

export default function App() {
  const path = usePath()
  if (path === '/app') return <Workbench />
  if (path === '/compare') return <Compare />
  return <Home />
}
