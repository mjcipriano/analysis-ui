import type { Cell, ColType, Column, OptionDef, Sheet, WireColumn, WireSheet } from './types'
import { uid } from './format'

export const isBlank = (v: Cell | undefined) => v === null || v === undefined || v === ''

/** Parse what the user typed / pasted into a cell value. */
export function parseCell(text: string): Cell {
  const t = text.trim()
  if (t === '') return null
  if (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(t)) return Number(t)
  // "1,234.5" style thousands separators
  if (/^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$/.test(t)) return Number(t.replace(/,/g, ''))
  return t
}

export function inferType(values: Cell[]): ColType {
  let seen = false
  for (const v of values) {
    if (isBlank(v)) continue
    seen = true
    if (typeof v !== 'number') return 'nominal'
  }
  return seen ? 'continuous' : 'continuous'
}

export function levelsOf(col: Column): string[] {
  const found: string[] = []
  const set = new Set<string>()
  for (const v of col.values) {
    if (isBlank(v)) continue
    const s = String(v)
    if (!set.has(s)) { set.add(s); found.push(s) }
  }
  if (col.levels?.length) {
    const explicit = col.levels.filter((l) => set.has(l))
    return [...explicit, ...found.filter((l) => !explicit.includes(l))]
  }
  if (found.every((l) => l.trim() !== '' && Number.isFinite(Number(l)))) return found.sort((a, b) => Number(a) - Number(b))
  return found
}

export function uniqueName(want: string, taken: string[]): string {
  const base = want.trim() || 'Column'
  if (!taken.includes(base)) return base
  for (let i = 2; ; i++) {
    const n = `${base} ${i}`
    if (!taken.includes(n)) return n
  }
}

export function columnLetter(i: number): string {
  let s = ''
  for (let n = i; n >= 0; n = Math.floor(n / 26) - 1) s = String.fromCharCode(65 + (n % 26)) + s
  return s
}

export function newColumn(name: string, nrows: number, type: ColType = 'continuous'): Column {
  return { id: uid('c'), name, type, values: new Array(nrows).fill(null) }
}

export function blankSheet(name: string, ncols = 4, nrows = 20): Sheet {
  return {
    id: uid('s'), name, nrows, version: 1,
    columns: Array.from({ length: ncols }, (_, i) => newColumn(columnLetter(i), nrows)),
  }
}

export function sheetFromWire(w: WireSheet, takenNames: string[] = []): Sheet {
  // a few spare rows so that new values can be typed straight away
  const nrows = Math.max(0, ...w.columns.map((c) => c.values.length)) + 10
  const names: string[] = []
  const columns = w.columns.map((c) => {
    const name = uniqueName(String(c.name), names)
    names.push(name)
    const values = c.values.slice()
    while (values.length < nrows) values.push(null)
    return { id: uid('c'), name, type: c.type, values, levels: c.levels } as Column
  })
  return { id: uid('s'), name: uniqueName(w.name || 'Data', takenNames), columns, nrows, version: 1 }
}

/** Columns to send to the backend; trailing fully-empty rows are trimmed. */
export function toWire(sheet: Sheet, names?: string[] | null): WireColumn[] {
  const cols = names ? sheet.columns.filter((c) => names.includes(c.name)) : sheet.columns
  let last = -1
  for (const c of cols) {
    for (let i = c.values.length - 1; i > last; i--) {
      if (!isBlank(c.values[i])) { last = i; break }
    }
  }
  return cols.map((c) => {
    const out: WireColumn = { name: c.name, type: c.type, values: c.values.slice(0, last + 1) }
    if (c.type !== 'continuous') out.levels = levelsOf(c)
    return out
  })
}

export function isVisible(o: OptionDef, values: Record<string, any>): boolean {
  if (!o.visible) return true
  return Object.entries(o.visible).every(([k, want]) => (Array.isArray(want) ? want.includes(values[k]) : values[k] === want))
}

/** Names of all columns referenced by the visible variable options. */
export function referencedColumns(schema: OptionDef[], values: Record<string, any>): string[] | null {
  // a free-text formula may reference anything
  if (schema.some((o) => o.type === 'text' && /formula/i.test(o.name) && isVisible(o, values) && values[o.name])) return null
  const out = new Set<string>()
  for (const o of schema) {
    if (!isVisible(o, values)) continue
    const v = values[o.name]
    if (o.type === 'variable' && v) out.add(v)
    if (o.type === 'variables' && Array.isArray(v)) v.forEach((n) => n && out.add(n))
  }
  return [...out]
}

/** Labels of required inputs that are still empty (so we can prompt instead of erroring). */
export function missingInputs(schema: OptionDef[], values: Record<string, any>, sheet: Sheet | undefined): string[] {
  const names = new Set(sheet?.columns.map((c) => c.name) ?? [])
  const out: string[] = []
  for (const o of schema) {
    if (!isVisible(o, values)) continue
    const v = values[o.name]
    if (o.type === 'variable' && o.required && (!v || !names.has(v))) out.push(o.label)
    if (o.type === 'variables') {
      const n = Array.isArray(v) ? v.filter((x) => names.has(x)).length : 0
      if (n < (o.min ?? 0)) out.push(o.min && o.min > 1 ? `${o.label} (at least ${o.min})` : o.label)
    }
  }
  return out
}

export function defaultsOf(schema: OptionDef[]): Record<string, any> {
  const out: Record<string, any> = {}
  for (const o of schema) out[o.name] = Array.isArray(o.default) ? [...o.default] : o.default && typeof o.default === 'object' ? { ...o.default } : o.default
  return out
}

const NAME_HINTS: Record<string, RegExp> = {
  time: /time|month|day|week|year|duration|surviv|follow|\bos\b|\bpfs\b/i,
  event: /status|event|death|dead|censor|outcome/i,
  subject: /subject|patient|animal|mouse|\bid\b|participant|donor/i,
  label: /name|label|gene|id\b|subgroup|patient|subject|sample/i,
  lower: /lower|low|lcl|\bci.?l/i,
  upper: /upper|high|ucl|\bci.?u/i,
  p: /^p$|p.?val|pvalue|padj|fdr|q.?val/i,
  effect: /log2|fold|fc|effect|diff/i,
}

/**
 * Sensible starting values for a new analysis / graph so that results appear
 * immediately: pick the data layout from the shape of the sheet and assign
 * columns to roles.
 */
export function autofill(schema: OptionDef[], sheet: Sheet, opts: { twoGroups?: boolean } = {}): Record<string, any> {
  const values = defaultsOf(schema)
  const cols = sheet.columns.filter((c) => c.values.some((v) => !isBlank(v)))
  const numeric = cols.filter((c) => c.type === 'continuous')
  const cats = cols.filter((c) => c.type !== 'continuous')
  // an ID-like categorical column (all values unique) is not a grouping variable
  const isId = (c: Column) => { const l = levelsOf(c).length; return l > 1 && l === c.values.filter((v) => !isBlank(v)).length }
  const grouping = cats.filter((c) => !isId(c))
  // a numeric column can stand in for a grouping variable only when it has a handful of distinct values (0/1, dose levels …)
  const fewValues = (c: Column) => new Set(c.values.filter((v) => !isBlank(v))).size <= 12
  const byName = new Map(schema.map((o) => [o.name, o]))
  const used = new Set<string>()

  const layout = byName.get('layout')
  if (layout?.choices) {
    const hasWide = layout.choices.some((c) => c.value === 'wide')
    const hasLong = layout.choices.some((c) => c.value === 'long')
    const wantLong = hasLong && grouping.length > 0 && numeric.length >= 1
    values.layout = wantLong || !hasWide ? 'long' : 'wide'
  }

  for (const o of schema) {
    if (!isVisible(o, values)) continue
    const permitted = o.permitted ?? ['continuous']
    const allowsCat = permitted.includes('nominal') || permitted.includes('ordinal')
    const allowsNum = permitted.includes('continuous')
    const hint = NAME_HINTS[o.name]
    if (o.type === 'variable') {
      if (!o.required && !['group', 'subject', 'label'].includes(o.name)) continue
      let pool: Column[]
      if (o.name === 'subject' || o.name === 'label') pool = cats.filter(isId)
      else if (allowsCat && allowsNum) pool = [...grouping, ...(o.required ? numeric.filter(fewValues) : [])]
      else if (allowsCat) pool = grouping
      else pool = numeric
      pool = pool.filter((c) => !used.has(c.name))
      let pick = hint ? pool.find((c) => hint.test(c.name)) : undefined
      if (!pick && hint && (o.name === 'time' || o.name === 'event')) pick = cols.filter((c) => !used.has(c.name)).find((c) => hint.test(c.name))
      if (!pick && (o.required || (o.name === 'group' && values.layout !== 'wide'))) pick = pool[0]
      if (o.name === 'label' && !pick) continue
      if (pick) { values[o.name] = pick.name; used.add(pick.name) }
    } else if (o.type === 'variables') {
      if ((o.min ?? 0) < 1) continue
      // roles that accept categories get grouping columns (plus numeric columns with only a few values)
      const pool = (allowsCat ? [...grouping, ...(allowsNum ? numeric.filter(fewValues) : [])] : numeric).filter((c) => !used.has(c.name))
      let take = pool
      if (o.max) take = pool.slice(0, o.max)
      else if (o.name === 'columns' && opts.twoGroups) take = pool.slice(0, 2)
      else if (o.name === 'y' && (grouping.length > 0 || pool.length > 6)) take = pool.slice(0, 1)
      else if (o.name === 'factors' || o.name === 'within' || o.name === 'between') take = pool.slice(0, 2)
      else if (o.name === 'covariates' || o.name === 'predictors') take = pool.slice(0, o.name === 'predictors' ? 4 : 1)
      else if (pool.length > 12) take = pool.slice(0, 12)
      values[o.name] = take.map((c) => c.name)
      take.forEach((c) => used.add(c.name))
    }
  }
  return values
}

/** Carry options over when switching to a related graph type. */
export function migrateOptions(oldValues: Record<string, any>, newSchema: OptionDef[], oldSchema: OptionDef[]): Record<string, any> {
  const out = defaultsOf(newSchema)
  const oldByName = new Map(oldSchema.map((o) => [o.name, o]))
  for (const o of newSchema) {
    const prev = oldByName.get(o.name)
    if (!prev || prev.type !== o.type) continue
    const v = oldValues[o.name]
    if (v === undefined) continue
    // keep data roles and look-and-feel; geometry defaults differ per type
    const isStyle = ['Titles & labels', 'Colors & theme', 'Axes', 'Legend', 'Fonts', 'Size', 'Data', 'Significance'].includes(o.section ?? '')
    if (!isStyle && JSON.stringify(v) === JSON.stringify(prev.default)) continue
    if (o.type === 'choice' && !o.choices?.some((c) => c.value === v)) continue
    out[o.name] = v
  }
  return out
}
