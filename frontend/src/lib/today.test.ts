import { afterEach, describe, expect, it, vi } from 'vitest'
import { currentOperationalDate, isPlanningDateToday } from './today'

describe('currentOperationalDate', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('returns the real current date as YYYY-MM-DD', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.setSystemTime(new Date(2026, 8, 4, 12, 0, 0)) // September 4, 2026, local time
    expect(currentOperationalDate()).toBe('2026-09-04')
  })

  it('uses local time, not UTC -- a time near a UTC day boundary does not flip the date', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    // 11pm local time -- still the same local calendar day even though UTC may already be the
    // next day for a timezone west of UTC (this app is Chicago-operations-facing).
    vi.setSystemTime(new Date(2026, 8, 4, 23, 0, 0))
    expect(currentOperationalDate()).toBe('2026-09-04')
  })

  it('pads single-digit months and days', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.setSystemTime(new Date(2026, 0, 5, 12, 0, 0)) // January 5, 2026
    expect(currentOperationalDate()).toBe('2026-01-05')
  })
})

describe('isPlanningDateToday', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('is true when the planning date matches the real current date', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.setSystemTime(new Date(2026, 8, 4, 12, 0, 0))
    expect(isPlanningDateToday('2026-09-04')).toBe(true)
  })

  it('is false when the planning date differs', () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.setSystemTime(new Date(2026, 8, 4, 12, 0, 0))
    expect(isPlanningDateToday('2026-08-28')).toBe(false)
  })

  it('is false for null/undefined, never a false positive', () => {
    expect(isPlanningDateToday(undefined)).toBe(false)
    expect(isPlanningDateToday(null)).toBe(false)
  })
})
