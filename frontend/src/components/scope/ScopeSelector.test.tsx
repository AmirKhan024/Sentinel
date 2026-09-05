import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { policyManifestFixture } from '../../test/mocks/fixtures'
import { ScopeSelector } from './ScopeSelector'

describe('ScopeSelector', () => {
  it('populates policy_id options from the policy manifest, never hardcoded', () => {
    render(
      <ScopeSelector
        scope={{}}
        setScopeField={vi.fn()}
        requiredFields={['policy_id']}
        manifests={{ policy: policyManifestFixture }}
      />,
    )
    expect(screen.getByRole('option', { name: 'pure_risk' })).toBeInTheDocument()
  })

  it('disables fold_id until fold_set is chosen', () => {
    render(
      <ScopeSelector
        scope={{}}
        setScopeField={vi.fn()}
        requiredFields={['fold_set', 'fold_id']}
        manifests={{}}
      />,
    )
    const foldIdSelect = screen.getByLabelText('fold_id') as HTMLSelectElement
    expect(foldIdSelect).toBeDisabled()
  })

  it('offers the real 17-quarter + covid_shift fold table once fold_set is quarterly', async () => {
    const user = userEvent.setup()
    render(
      <ScopeSelector
        scope={{ fold_set: 'quarterly' }}
        setScopeField={vi.fn()}
        requiredFields={['fold_set', 'fold_id']}
        manifests={{}}
      />,
    )
    const foldIdSelect = screen.getByLabelText('fold_id') as HTMLSelectElement
    expect(foldIdSelect).not.toBeDisabled()
    await user.selectOptions(foldIdSelect, 'quarterly-2024Q2')
    expect(screen.getByRole('option', { name: 'quarterly-2024Q2' })).toBeInTheDocument()
  })
})
