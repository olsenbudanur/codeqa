import type { Episode } from '@/state/episode'

// Saved conversations, per viewer, in localStorage. Every read/write is guarded:
// storage can be empty or throw in private windows. Swap this module for an
// API-backed store when conversations need to be shared or read by the API.
export interface Conversation {
  id: string
  createdAt: number
  repoId: string
  profile: string
  question: string
  episode: Episode
}

const KEY = 'codeqa.conversations.v1'
const MAX = 100

function read(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return []
    const list = JSON.parse(raw) as Conversation[]
    return Array.isArray(list) ? list : []
  } catch {
    return []
  }
}

function write(list: Conversation[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX)))
  } catch {
    /* quota or private mode: history is a convenience */
  }
}

export function listConversations(): Conversation[] {
  return read().sort((a, b) => b.createdAt - a.createdAt)
}

export function getConversation(id: string): Conversation | undefined {
  return read().find((c) => c.id === id)
}

export function saveConversation(episode: Episode, id?: string): Conversation {
  const list = read()
  const existing = id ? list.find((c) => c.id === id) : undefined
  const conv: Conversation = {
    id: existing?.id ?? `c_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
    createdAt: existing?.createdAt ?? Date.now(),
    repoId: episode.repoId,
    profile: episode.profile,
    question: episode.question,
    episode,
  }
  write([conv, ...list.filter((c) => c.id !== conv.id)])
  return conv
}

export function deleteConversation(id: string): void {
  write(read().filter((c) => c.id !== id))
}
