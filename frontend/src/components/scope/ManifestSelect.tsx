/** A plain `<select>` bound to a list of option strings, shared by every manifest-sourced
 * scope field (policy_id, model_name, k_name, schedule_config_id). */
export function ManifestSelect({
  label,
  options,
  value,
  onChange,
  emptyHint = 'not available',
}: {
  label: string
  options: string[]
  value: string | undefined
  onChange: (value: string | undefined) => void
  emptyHint?: string
}) {
  return (
    <label className="scope-field">
      {label}
      <select
        value={value ?? ''}
        disabled={options.length === 0}
        onChange={(e) => onChange(e.target.value || undefined)}
      >
        <option value="">{options.length === 0 ? emptyHint : '— select —'}</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  )
}
