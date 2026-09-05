import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'

/** One operational count on the Overview page (e.g. "28 recommended for inspection"). Clicking
 * it navigates to the page that explains that number in full -- the count itself is never the
 * end of the story. */
export function SummaryCard({
  label,
  value,
  hint,
  to,
  variant = 'primary',
}: {
  label: string
  value: ReactNode
  hint?: string
  to?: string
  /** `primary` = the operational numbers driving today's plan; `attention` = follow-up counts
   * that need a look but are not evidence of a wrong recommendation. Purely visual -- it does not
   * change what the card links to or how the count itself is computed. */
  variant?: 'primary' | 'attention'
}) {
  const navigate = useNavigate()
  const clickable = Boolean(to)
  return (
    <div
      className={`summary-card${clickable ? ' summary-card-clickable' : ''}${variant === 'attention' ? ' summary-card-attention' : ''}`}
      onClick={clickable ? () => navigate(to!) : undefined}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') navigate(to!)
            }
          : undefined
      }
    >
      <span className="summary-card-value">{value}</span>
      <span className="summary-card-label">{label}</span>
      {hint && <span className="summary-card-hint">{hint}</span>}
    </div>
  )
}
