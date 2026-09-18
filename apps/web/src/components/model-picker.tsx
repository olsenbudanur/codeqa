import type { Profile } from '@/lib/contracts'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'

export function ModelPicker({
  profiles,
  value,
  onChange,
  disabled,
}: {
  profiles: Profile[]
  value: string
  onChange: (name: string) => void
  disabled?: boolean
}) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="h-8 w-[220px] max-sm:w-[150px]" aria-label="Model">
        <SelectValue placeholder="Model">{(() => { const p = profiles.find((x) => x.name === value); return p?.source === 'checkpoints' ? p.name : (p?.label ?? value) })()}</SelectValue>
      </SelectTrigger>
      <SelectContent align="end" className="max-h-[70dvh]">
        {(['profiles.yaml', 'checkpoints'] as const).map((src) => {
          const items = profiles.filter((p) => (p.source ?? 'profiles.yaml') === src)
          if (items.length === 0) return null
          return (
            <SelectGroup key={src}>
              <SelectLabel className="font-mono text-[11px] text-muted-foreground">{src === 'checkpoints' ? 'Training checkpoints' : 'Models'}</SelectLabel>
              {items.map((p) => (
                <SelectItem key={p.name} value={p.name}>
                  <span className="flex flex-col">
                    <span>{src === 'checkpoints' ? p.name : (p.label ?? p.name)}</span>
                    {p.note && <span className="text-xs text-muted-foreground">{p.note.replace(' (checkpoint)', '')}</span>}
                  </span>
                </SelectItem>
              ))}
            </SelectGroup>
          )
        })}
      </SelectContent>
    </Select>
  )
}
