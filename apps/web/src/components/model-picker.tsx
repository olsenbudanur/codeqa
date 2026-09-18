import type { Profile } from '@/lib/contracts'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

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
        <SelectValue placeholder="Model">{profiles.find((p) => p.name === value)?.label ?? value}</SelectValue>
      </SelectTrigger>
      <SelectContent align="end">
        {profiles.map((p) => (
          <SelectItem key={p.name} value={p.name}>
            <span className="flex flex-col">
              <span>{p.label ?? p.name}</span>
              {p.note && <span className="text-xs text-muted-foreground">{p.note}</span>}
            </span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
