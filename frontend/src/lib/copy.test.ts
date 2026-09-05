import { describe, expect, it } from 'vitest'
import {
  apiErrorCodeLabel,
  operationalCoverageNote,
  planLabelForToday,
  planStalenessNote,
  workAreaLabel,
  workBlockDisplayLabel,
} from './copy'

describe('apiErrorCodeLabel', () => {
  it('translates a known error code', () => {
    expect(apiErrorCodeLabel('artifact_not_found')).toMatch(/couldn't find that information/)
  })

  it('falls back to a generic, non-blank message for an unmapped code', () => {
    const label = apiErrorCodeLabel('some_new_code_nobody_mapped_yet')
    expect(label.length).toBeGreaterThan(0)
    expect(label).not.toBe('some_new_code_nobody_mapped_yet')
  })
})

describe('workBlockDisplayLabel', () => {
  it('uses the real label when present', () => {
    expect(workBlockDisplayLabel('Area 1', 0)).toBe('Area 1')
  })

  it('falls back to a 1-based position, never the raw id, when no label is present', () => {
    expect(workBlockDisplayLabel(undefined, 0)).toBe('Work block 1')
    expect(workBlockDisplayLabel('', 2)).toBe('Work block 3')
  })
})

describe('planLabelForToday', () => {
  it('says "Today\'s inspection plan" when the date matches the real current date', () => {
    expect(planLabelForToday('2026-09-04', true)).toMatch(/^Today's inspection plan/)
  })

  it('honestly says a plan is not today\'s when it differs, never silently relabeled', () => {
    const label = planLabelForToday('2026-08-28', false)
    expect(label).toMatch(/not today's plan yet/)
    expect(label).not.toMatch(/^Today/)
  })

  it('says plainly when there is no plan at all', () => {
    expect(planLabelForToday(undefined, false)).toBe('No operational plan is available yet.')
  })
})

describe('planStalenessNote', () => {
  it('states the plan was built for today when the dates match', () => {
    expect(planStalenessNote('2026-09-04', '2026-09-04')).toMatch(/built for today/)
  })

  it('states plainly that no plan exists yet for today when they differ, and that nothing builds automatically', () => {
    const note = planStalenessNote('2026-08-28', '2026-09-04')
    expect(note).toMatch(/No plan has been built yet for/)
    expect(note).toMatch(/never builds a plan automatically/)
  })
})

describe('workAreaLabel', () => {
  it('turns "Area N" into "Work Area N"', () => {
    expect(workAreaLabel('Area 7')).toBe('Work Area 7')
  })

  it('passes any other label through unchanged, never mangling an unexpected format', () => {
    expect(workAreaLabel('unmapped')).toBe('unmapped')
    expect(workAreaLabel('Downtown District')).toBe('Downtown District')
  })
})

describe('operationalCoverageNote', () => {
  it('states all three counts plainly, without claiming anyone unselected is safe', () => {
    const note = operationalCoverageNote(35859, 35859, 30)
    expect(note).toContain('35,859')
    expect(note).toContain('30')
    expect(note).not.toMatch(/safe/i)
    expect(note).toMatch(/not flagged as lower risk/)
  })
})
