import type { StagedRequestReceipt } from '../../api/types'
import { formatDateTime } from '../../lib/copy'

/** The confirmation every staged write shows, worded honestly against ADR 0049: this is a
 * receipt that the decision was recorded, never a claim that the visible plan changed. */
export function StagedReceiptNotice({ receipt, what }: { receipt: StagedRequestReceipt; what: string }) {
  return (
    <div className="staged-receipt" role="status">
      <p className="staged-receipt-headline">
        <strong>Sentinel recorded your decision.</strong>
      </p>
      <p>
        {what} is <strong>staged</strong>, submitted {formatDateTime(receipt.staged_at)}. It is not
        applied to the visible plan yet -- an operator applies staged decisions the next time this
        planning run is rebuilt. Until then, it's on record and will not be lost.
      </p>
      <p className="hint">Reference id: {receipt.request_id}</p>
    </div>
  )
}
