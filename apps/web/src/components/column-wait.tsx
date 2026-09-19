import type { Episode } from '@/state/episode'
import { useElapsed, waitingText } from '@/components/working'

export function ColumnWait({ episode }: { episode: Episode }) {
  const elapsed = useElapsed(episode.startedAt, episode.status === 'running')
  return <>{waitingText(elapsed, episode.rows.filter((r) => r.kind === 'call').length)}</>
}
