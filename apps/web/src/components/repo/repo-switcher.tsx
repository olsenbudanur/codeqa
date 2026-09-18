import { useState, type FormEvent } from 'react'
import { Check, ChevronsUpDown, FolderGit2, Plus } from 'lucide-react'
import type { RepoSummary } from '@/lib/contracts'
import { repoName } from '@/lib/repo'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'

const fmt = new Intl.NumberFormat('en-US')

// Top-of-rail repository switcher, in the shape of shadcn-admin's team switcher.
export function RepoSwitcher({
  repos,
  selected,
  onSelect,
  onAdd,
  busy,
}: {
  repos: RepoSummary[]
  selected: RepoSummary | null
  onSelect: (repoId: string) => void
  onAdd: (url: string) => Promise<void>
  busy: boolean
}) {
  const [adding, setAdding] = useState(false)
  const [url, setUrl] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!url.trim()) return
    setSubmitting(true)
    setError(null)
    try {
      await onAdd(url.trim())
      setUrl('')
      setAdding(false)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            className="flex w-full items-center gap-2.5 rounded-md px-2 py-2 text-left transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring"
            aria-label="Switch repository"
          >
            <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground" aria-hidden>
              <FolderGit2 className="size-4" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium">{selected ? repoName(selected) : 'Pick a repository'}</span>
              <span className="block truncate font-mono text-[11px] text-muted-foreground">
                {selected ? `${selected.sha.slice(0, 7)}, ${fmt.format(selected.files)} files` : `${repos.length} indexed`}
              </span>
            </span>
            <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground" aria-hidden />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-[268px]">
          <DropdownMenuLabel className="text-xs text-muted-foreground">Repositories</DropdownMenuLabel>
          <div className="max-h-[50dvh] overflow-y-auto">
            {repos.map((r) => {
              const ready = r.stage === 'ready'
              return (
                <DropdownMenuItem key={r.repo_id} disabled={!ready} onSelect={() => onSelect(r.repo_id)} className="gap-2">
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{repoName(r)}</span>
                    <span className="block font-mono text-[11px] text-muted-foreground">
                      {ready ? `${r.sha.slice(0, 7)}, ${fmt.format(r.files)} files` : 'indexing'}
                    </span>
                  </span>
                  <Check className={cn('size-4', selected?.repo_id === r.repo_id ? 'opacity-100' : 'opacity-0')} aria-hidden />
                </DropdownMenuItem>
              )
            })}
          </div>
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={() => setAdding(true)} disabled={busy} className="gap-2">
            <Plus className="size-4" aria-hidden />
            Add a repository
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={adding} onOpenChange={setAdding}>
        <DialogContent className="sm:max-w-[440px]">
          <form onSubmit={submit}>
            <DialogHeader>
              <DialogTitle>Add a repository</DialogTitle>
              <DialogDescription>Paste a GitHub URL. It is askable in about a minute; summaries fill in afterwards.</DialogDescription>
            </DialogHeader>
            <div className="py-4">
              <label htmlFor="repo-url" className="sr-only">
                Repository URL
              </label>
              <Input
                id="repo-url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                className="font-mono text-sm"
                autoFocus
                autoComplete="off"
                spellCheck={false}
                disabled={submitting}
              />
              {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
            </div>
            <DialogFooter>
              <Button type="button" variant="ghost" onClick={() => setAdding(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={submitting || !url.trim()}>
                Index repository
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  )
}
