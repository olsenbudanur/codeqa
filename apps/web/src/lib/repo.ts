export function repoName(r: { repo_id: string }): string {
  const [owner, name] = r.repo_id.split('__')
  return `${owner}/${name}`
}
