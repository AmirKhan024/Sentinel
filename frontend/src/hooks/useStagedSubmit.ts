import { useState } from 'react'
import { classifyError, type ClassifiedError } from '../api/errors'
import type { StagedRequestReceipt } from '../api/types'

type Status = 'idle' | 'submitting' | 'success' | 'error'

/** Shared submit lifecycle for the four staged-write forms (override, adjustment, execution
 * event, review resolution). Every one of them behaves identically at this level: call the
 * contract, get back a `StagedRequestReceipt` or a classified error, never pretend the write was
 * applied. What differs between forms is only which fields they collect and which endpoint they
 * call -- that stays in each form component. */
export function useStagedSubmit(submitFn: (signal?: AbortSignal) => Promise<StagedRequestReceipt>) {
  const [status, setStatus] = useState<Status>('idle')
  const [receipt, setReceipt] = useState<StagedRequestReceipt | null>(null)
  const [error, setError] = useState<ClassifiedError | null>(null)

  async function submit() {
    setStatus('submitting')
    setError(null)
    try {
      const result = await submitFn()
      setReceipt(result)
      setStatus('success')
    } catch (err) {
      setError(classifyError(err))
      setStatus('error')
    }
  }

  function reset() {
    setStatus('idle')
    setReceipt(null)
    setError(null)
  }

  return { status, receipt, error, submit, reset }
}
