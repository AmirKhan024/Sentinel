import { isValidElement, type ReactNode } from 'react'

/** Label/value pair that renders "—" for null/undefined/empty -- never silently omits a field.
 * Accepts a React element (e.g. a StatusBadge) as well as a primitive value. */
export function FieldRow({ label, value }: { label: string; value: unknown | ReactNode }) {
  let display: ReactNode
  if (isValidElement(value)) {
    display = value
  } else if (value === null || value === undefined || value === '') {
    display = '—'
  } else if (typeof value === 'boolean') {
    display = value ? 'true' : 'false'
  } else {
    display = String(value)
  }
  return (
    <div className="field-row">
      <span className="field-label">{label}</span>
      <span className="field-value">{display}</span>
    </div>
  )
}
