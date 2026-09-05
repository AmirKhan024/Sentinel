import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ErrorState } from './ErrorState'

describe('ErrorState', () => {
  it('shows a plain-language message for a known error code', () => {
    render(
      <ErrorState
        error={{ kind: 'client', status: 404, error: 'artifact_not_found', detail: 'no artifact' }}
      />,
    )
    expect(screen.getByText(/We couldn't find that information yet/)).toBeInTheDocument()
    // The raw code is kept, de-emphasized, for support purposes -- never dropped entirely.
    expect(screen.getByText(/artifact_not_found: no artifact/)).toBeInTheDocument()
  })

  it('never renders blank for an unmapped error code', () => {
    render(
      <ErrorState
        error={{ kind: 'client', status: 500, error: 'something_new', detail: 'unexpected' }}
      />,
    )
    expect(screen.getByText(/Something went wrong/)).toBeInTheDocument()
  })

  it('still handles ambiguous_scope with its own dedicated rendering', () => {
    render(
      <ErrorState
        error={{
          kind: 'client',
          status: 422,
          error: 'ambiguous_scope',
          detail: 'missing fields',
          missingScopeFields: ['policy_id'],
        }}
      />,
    )
    expect(screen.getByText('missing fields')).toBeInTheDocument()
    expect(screen.getByText('policy_id')).toBeInTheDocument()
  })
})
