import type { ManifestCheck } from '../../api/types'
import { checkLabel } from '../../lib/copy'

/** Operational explanation first, raw implementation detail last: a short plain-language
 * sentence, then the raw check name and detail, kept rather than removed so nothing auditable
 * is hidden. This panel only ever renders inside a "Technical details" disclosure. */
export function ManifestChecksPanel({ checks }: { checks: ManifestCheck[] | undefined }) {
  if (!checks || checks.length === 0) {
    return <p className="state state-empty">No checks recorded in this manifest.</p>
  }
  return (
    <>
      <p className="hint">
        These are automated integrity checks Sentinel runs against its own data every time it
        builds a plan — for example, that nothing was double-booked or silently altered. A
        checkmark means the check passed; the raw name and detail beside it are the audit record.
      </p>
      <ul className="checks-list">
        {checks.map((check) => (
          <li key={check.name} className={check.passed ? 'check-pass' : `check-fail check-${check.severity}`}>
            <strong>{check.passed ? '✓' : check.severity === 'error' ? '✗' : '⚠'}</strong>{' '}
            {checkLabel(check.name)}
            <span className="check-detail">
              {' '}
              — <code>{check.name}</code>: {check.detail}
            </span>
          </li>
        ))}
      </ul>
    </>
  )
}
