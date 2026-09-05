import type { ClassifiedError } from '../../api/errors'
import { apiErrorCodeLabel } from '../../lib/copy'

export function ErrorState({
  error,
  onCandidateSelect,
}: {
  error: ClassifiedError
  /** Called when the caller clicks a `candidate_values` chip from an ambiguous_scope 422. */
  onCandidateSelect?: (value: unknown) => void
}) {
  if (error.kind === 'network') {
    return (
      <div className="state state-error" role="alert">
        <p>{error.message}</p>
      </div>
    )
  }

  if (error.kind === 'client' && error.error === 'ambiguous_scope') {
    return (
      <div className="state state-error state-ambiguous" role="alert">
        <p>{error.detail}</p>
        {error.missingScopeFields && error.missingScopeFields.length > 0 && (
          <p>
            Add: <strong>{error.missingScopeFields.join(', ')}</strong>
          </p>
        )}
        {error.candidateValues && error.candidateValues.length > 0 && (
          <div className="candidate-chips">
            {error.candidateValues.map((value, i) => (
              <button
                key={i}
                type="button"
                className="chip"
                onClick={() => onCandidateSelect?.(value)}
              >
                {String(value)}
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  if (error.kind === 'client') {
    return (
      <div className="state state-error" role="alert">
        <p>{apiErrorCodeLabel(error.error)}</p>
        <p className="hint state-error-code">
          {error.error}: {error.detail}
        </p>
      </div>
    )
  }

  if (error.kind === 'server') {
    return (
      <div className="state state-error" role="alert">
        <p>{error.detail}</p>
      </div>
    )
  }

  return (
    <div className="state state-error" role="alert">
      <p>{error.kind === 'unknown' ? error.message : 'Request was cancelled.'}</p>
    </div>
  )
}
