import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DataTable } from './DataTable'

interface Row {
  id: string
  name: string
}

describe('DataTable', () => {
  const rows: Row[] = [
    { id: '1', name: 'a' },
    { id: '2', name: 'b' },
  ]
  const columns = [{ key: 'name', label: 'Name', render: (r: Row) => r.name }]

  it('renders one row per item', () => {
    render(<DataTable columns={columns} rows={rows} rowKey={(r) => r.id} />)
    expect(screen.getAllByRole('row')).toHaveLength(rows.length + 1) // + header row
  })

  it('offers no sort-column control -- the API fixes sort order server-side', () => {
    render(<DataTable columns={columns} rows={rows} rowKey={(r) => r.id} />)
    expect(screen.queryByRole('columnheader', { name: /sort/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /sort/i })).not.toBeInTheDocument()
  })
})
