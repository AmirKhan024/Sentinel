import type { PageMeta } from '../../api/types'

/** Only offset/limit navigation and a descending toggle -- the API fixes the sort column
 * per endpoint server-side, so no column picker is offered here (see DataTable). */
export function PaginationControls({
  page,
  onOffsetChange,
  descending,
  onDescendingChange,
}: {
  page: PageMeta
  onOffsetChange: (offset: number) => void
  descending: boolean
  onDescendingChange: (descending: boolean) => void
}) {
  const hasPrev = page.offset > 0
  const hasNext = page.offset + page.limit < page.total

  return (
    <div className="pagination-controls">
      <span>
        {page.total === 0
          ? '0 rows'
          : `${page.offset + 1}-${Math.min(page.offset + page.limit, page.total)} of ${page.total}`}
      </span>
      <button type="button" disabled={!hasPrev} onClick={() => onOffsetChange(Math.max(0, page.offset - page.limit))}>
        Previous
      </button>
      <button type="button" disabled={!hasNext} onClick={() => onOffsetChange(page.offset + page.limit)}>
        Next
      </button>
      <label>
        <input
          type="checkbox"
          checked={descending}
          onChange={(e) => onDescendingChange(e.target.checked)}
        />
        Descending
      </label>
    </div>
  )
}
