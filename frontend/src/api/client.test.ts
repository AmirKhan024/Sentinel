import { http, HttpResponse } from 'msw'
import { describe, expect, it } from 'vitest'
import { server } from '../test/mocks/server'
import { apiFetch } from './client'
import { ApiRequestError, NetworkError } from './errors'

const BASE = 'http://127.0.0.1:8000'

describe('apiFetch', () => {
  it('parses a successful JSON response', async () => {
    server.use(http.get(`${BASE}/v1/thing`, () => HttpResponse.json({ ok: true })))
    const result = await apiFetch<{ ok: boolean }>('/v1/thing')
    expect(result).toEqual({ ok: true })
  })

  it('sends only defined query params', async () => {
    server.use(
      http.get(`${BASE}/v1/thing`, ({ request }) => {
        const url = new URL(request.url)
        expect(url.searchParams.get('a')).toBe('1')
        expect(url.searchParams.has('b')).toBe(false)
        return HttpResponse.json({ ok: true })
      }),
    )
    await apiFetch('/v1/thing', { a: 1, b: undefined })
  })

  it('throws NetworkError when the request fails to connect', async () => {
    server.use(http.get(`${BASE}/v1/thing`, () => HttpResponse.error()))
    await expect(apiFetch('/v1/thing')).rejects.toBeInstanceOf(NetworkError)
  })

  it('throws ApiRequestError carrying the parsed body on a 4xx', async () => {
    server.use(
      http.get(`${BASE}/v1/thing`, () =>
        HttpResponse.json({ error: 'ambiguous_scope', detail: 'missing stuff' }, { status: 422 }),
      ),
    )
    await expect(apiFetch('/v1/thing')).rejects.toMatchObject({
      status: 422,
      body: { error: 'ambiguous_scope', detail: 'missing stuff' },
    })
  })

  it('throws ApiRequestError on a 5xx', async () => {
    server.use(
      http.get(`${BASE}/v1/thing`, () =>
        HttpResponse.json({ error: 'internal_error', detail: 'boom' }, { status: 500 }),
      ),
    )
    const err = await apiFetch('/v1/thing').catch((e) => e)
    expect(err).toBeInstanceOf(ApiRequestError)
    expect((err as ApiRequestError).status).toBe(500)
  })

  it('rejects with AbortError when the signal is aborted', async () => {
    server.use(
      http.get(`${BASE}/v1/thing`, async () => {
        await new Promise((resolve) => setTimeout(resolve, 50))
        return HttpResponse.json({ ok: true })
      }),
    )
    const controller = new AbortController()
    const promise = apiFetch('/v1/thing', {}, controller.signal)
    controller.abort()
    await expect(promise).rejects.toMatchObject({ name: 'AbortError' })
  })
})
