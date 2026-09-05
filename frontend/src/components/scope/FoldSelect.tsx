import { FOLD_SETS, foldIdsForSet } from '../../api/folds'

export function FoldSetSelect({
  value,
  onChange,
}: {
  value: string | undefined
  onChange: (value: string | undefined) => void
}) {
  return (
    <label className="scope-field">
      fold_set
      <select value={value ?? ''} onChange={(e) => onChange(e.target.value || undefined)}>
        <option value="">— select —</option>
        {FOLD_SETS.map((fs) => (
          <option key={fs} value={fs}>
            {fs}
          </option>
        ))}
      </select>
    </label>
  )
}

export function FoldIdSelect({
  foldSet,
  value,
  onChange,
}: {
  foldSet: string | undefined
  value: string | undefined
  onChange: (value: string | undefined) => void
}) {
  const options = foldIdsForSet(foldSet)
  return (
    <label className="scope-field">
      fold_id
      <select
        value={value ?? ''}
        disabled={!foldSet}
        onChange={(e) => onChange(e.target.value || undefined)}
      >
        <option value="">{foldSet ? '— select —' : 'select fold_set first'}</option>
        {options.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </label>
  )
}
