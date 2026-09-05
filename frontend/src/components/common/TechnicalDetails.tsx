import type { ReactNode } from 'react'

/** Progressive disclosure, in one place. The plain-language view is always what renders first;
 * anything technical (raw codes, IDs, manifests, scope internals) lives inside this collapsed
 * `<details>` so an inspector never has to see it, while an interviewer or engineer can still
 * open it in one click. */
export function TechnicalDetails({
  summary = 'Technical details',
  children,
}: {
  summary?: string
  children: ReactNode
}) {
  return (
    <details className="technical-details">
      <summary>{summary}</summary>
      <div className="technical-details-body">{children}</div>
    </details>
  )
}
