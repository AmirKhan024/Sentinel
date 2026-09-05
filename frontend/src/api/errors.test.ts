import { describe, expect, it } from 'vitest'
import { ApiRequestError, NetworkError, classifyError } from './errors'

describe('classifyError', () => {
  it.each([
    ['artifact_not_found', 404],
    ['row_not_found', 404],
    ['unknown_component', 404],
    ['ambiguous_scope', 422],
  ])('classifies a %s (%d) as a client error', (errorCode, status) => {
    const err = new ApiRequestError(status, { error: errorCode, detail: 'detail text' })
    const classified = classifyError(err)
    expect(classified).toMatchObject({ kind: 'client', status, error: errorCode, detail: 'detail text' })
  })

  it('carries missing_scope_fields and candidate_values through for ambiguous_scope', () => {
    const err = new ApiRequestError(422, {
      error: 'ambiguous_scope',
      detail: 'missing stuff',
      missing_scope_fields: ['fold_id'],
      candidate_values: ['TI-1', 'TI-2'],
    })
    const classified = classifyError(err)
    expect(classified).toMatchObject({
      kind: 'client',
      missingScopeFields: ['fold_id'],
      candidateValues: ['TI-1', 'TI-2'],
    })
  })

  it('classifies a 500 internal_error as a server error', () => {
    const err = new ApiRequestError(500, { error: 'internal_error', detail: 'An internal error occurred.' })
    expect(classifyError(err)).toEqual({ kind: 'server', status: 500, detail: 'An internal error occurred.' })
  })

  it('classifies a NetworkError as network', () => {
    const err = new NetworkError('unreachable')
    expect(classifyError(err)).toEqual({ kind: 'network', message: 'unreachable' })
  })

  it('classifies an AbortError as aborted', () => {
    const err = new DOMException('aborted', 'AbortError')
    expect(classifyError(err)).toEqual({ kind: 'aborted' })
  })

  it('classifies an unrecognized error as unknown', () => {
    expect(classifyError(new Error('mystery'))).toEqual({ kind: 'unknown', message: 'mystery' })
  })
})
