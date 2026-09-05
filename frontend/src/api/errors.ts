/**
 * Every error the API client can produce, and the one function that turns any thrown value
 * into a shape a component can render. No component should write its own `if (status === 422)`
 * -- that logic lives here, once.
 */

/** The API's own error body shape: `{"error": "...", "detail": "...", ...extra}`. */
export interface ApiErrorBody {
  error: string
  detail: string
  missing_scope_fields?: string[]
  candidate_values?: unknown[]
  [key: string]: unknown
}

export class NetworkError extends Error {
  constructor(message = 'Could not reach the Sentinel API. Is `uv run sentinel serve` running?') {
    super(message)
    this.name = 'NetworkError'
  }
}

export class ApiRequestError extends Error {
  readonly status: number
  readonly body: ApiErrorBody | null

  constructor(status: number, body: ApiErrorBody | null) {
    super(body?.detail ?? `Request failed with status ${status}`)
    this.name = 'ApiRequestError'
    this.status = status
    this.body = body
  }
}

export type ClassifiedError =
  | { kind: 'network'; message: string }
  | {
      kind: 'client'
      status: number
      error: string
      detail: string
      missingScopeFields?: string[]
      candidateValues?: unknown[]
    }
  | { kind: 'server'; status: number; detail: string }
  | { kind: 'aborted' }
  | { kind: 'unknown'; message: string }

export function isAbortError(err: unknown): boolean {
  return (
    (err instanceof Error || err instanceof DOMException) &&
    (err as { name?: string }).name === 'AbortError'
  )
}

export function classifyError(err: unknown): ClassifiedError {
  if (isAbortError(err)) {
    return { kind: 'aborted' }
  }
  if (err instanceof NetworkError) {
    return { kind: 'network', message: err.message }
  }
  if (err instanceof ApiRequestError) {
    if (err.status >= 400 && err.status < 500) {
      return {
        kind: 'client',
        status: err.status,
        error: err.body?.error ?? 'client_error',
        detail: err.body?.detail ?? err.message,
        missingScopeFields: err.body?.missing_scope_fields,
        candidateValues: err.body?.candidate_values,
      }
    }
    return {
      kind: 'server',
      status: err.status,
      detail: err.body?.detail ?? 'An internal error occurred.',
    }
  }
  const message = err instanceof Error ? err.message : String(err)
  return { kind: 'unknown', message }
}
