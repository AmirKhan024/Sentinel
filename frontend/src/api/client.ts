import { ApiRequestError, type ApiErrorBody, NetworkError } from './errors'

const BASE_URL: string =
  (import.meta.env.VITE_SENTINEL_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000'

export type QueryParams = Record<string, string | number | boolean | undefined>

/** The one low-level function every resource module calls. Builds the URL, distinguishes an
 * aborted request from a network failure from a parsed error body, and returns typed JSON. */
export async function apiFetch<T>(
  path: string,
  params: QueryParams = {},
  signal?: AbortSignal,
): Promise<T> {
  const url = new URL(path, BASE_URL)
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      url.searchParams.set(key, String(value))
    }
  }

  let response: Response
  try {
    response = await fetch(url, { signal })
  } catch (err) {
    if ((err instanceof Error || err instanceof DOMException) && (err as { name?: string }).name === 'AbortError') {
      throw err
    }
    throw new NetworkError()
  }

  const body: unknown = await response.json().catch(() => null)

  if (!response.ok) {
    throw new ApiRequestError(response.status, body as ApiErrorBody | null)
  }

  return body as T
}

/** For the four staged-write endpoints only. Same error handling as `apiFetch`; the only
 * difference is a JSON body instead of query params -- every write in Sentinel is a POST that
 * stages a request (ADR 0049), never a PATCH/PUT, so this is the one write primitive needed. */
export async function apiPost<T>(path: string, payload: object, signal?: AbortSignal): Promise<T> {
  const url = new URL(path, BASE_URL)

  let response: Response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal,
    })
  } catch (err) {
    if ((err instanceof Error || err instanceof DOMException) && (err as { name?: string }).name === 'AbortError') {
      throw err
    }
    throw new NetworkError()
  }

  const body: unknown = await response.json().catch(() => null)

  if (!response.ok) {
    throw new ApiRequestError(response.status, body as ApiErrorBody | null)
  }

  return body as T
}

export { BASE_URL }
