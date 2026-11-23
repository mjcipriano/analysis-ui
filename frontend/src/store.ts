import { create } from 'zustand'
import { api } from './api'
import { autofill, blankSheet, defaultsOf, inferType, isBlank, migrateOptions, newColumn, sheetFromWire, uniqueName, columnLetter } from './data'
import { uid } from './format'
import type { AnalysisItem, Catalog, Cell, ColType, Column, GraphItem, LayoutItem, Selection, Sheet, WireColumn, WireSheet } from './types'

export type Item = AnalysisItem | GraphItem

interface Toast { id: string; text: string; kind: 'info' | 'error' }

interface Project {
  name: string
  sheets: Sheet[]
  items: Item[]
  layouts?: LayoutItem[]
}

interface State extends Project {
  layouts: LayoutItem[]
  addLayout: (graphIds?: string[]) => void
  patchLayout: (id: string, patch: Partial<LayoutItem>) => void
  deleteLayout: (id: string) => void
  catalog: Catalog | null
  catalogError: string | null
  selection: Selection
  toasts: Toast[]
  dark: boolean
  defaultEngine: 'matplotlib' | 'ggplot2'
  styleClipboard: { options: Record<string, any>; engine: string } | null
  past: Sheet[][]
  future: Sheet[][]

  loadCatalog: () => Promise<void>
  select: (s: Selection) => void
  toast: (text: string, kind?: 'info' | 'error') => void
  dismissToast: (id: string) => void
  setDark: (dark: boolean) => void
  setProjectName: (name: string) => void
  newProject: () => void
  loadProject: (p: Project) => void

  addBlankSheet: () => void
  addWireSheets: (sheets: WireSheet[]) => void
  renameSheet: (id: string, name: string) => void
  deleteSheet: (id: string) => void
  duplicateSheet: (id: string) => void

  editCells: (sheetId: string, edits: { col: number; row: number; value: Cell }[]) => void
  pasteBlock: (sheetId: string, col: number, row: number, block: Cell[][]) => void
  addColumn: (sheetId: string, at?: number) => void
  addWireColumns: (sheetId: string, cols: WireColumn[]) => void
  deleteColumns: (sheetId: string, indexes: number[]) => void
  renameColumn: (sheetId: string, index: number, name: string) => void
  setColumnType: (sheetId: string, index: number, type: ColType) => void
  setColumnLevels: (sheetId: string, index: number, levels: string[]) => void
  moveColumn: (sheetId: string, from: number, to: number) => void
  addRows: (sheetId: string, count: number, at?: number) => void
  deleteRows: (sheetId: string, rows: number[]) => void
  sortRows: (sheetId: string, col: number, descending: boolean) => void
  undo: () => void
  redo: () => void

  addAnalysis: (sheetId: string, analysisId: string) => void
  addGraph: (sheetId: string, graphType: string, options?: Record<string, any>) => void
  patchItem: (id: string, patch: Partial<AnalysisItem> | Partial<GraphItem>) => void
  setOptions: (id: string, patch: Record<string, any>) => void
  resetOptions: (id: string) => void
  renameItem: (id: string, name: string) => void
  deleteItem: (id: string) => void
  duplicateItem: (id: string) => void
  changeGraphType: (id: string, graphType: string) => void
  copyStyle: (id: string) => void
  pasteStyle: (id: string) => void
}

const MAX_UNDO = 40

function withSheet(state: State, sheetId: string, fn: (s: Sheet) => Sheet | null, remember = true): Partial<State> {
  const idx = state.sheets.findIndex((s) => s.id === sheetId)
  if (idx < 0) return {}
  const next = fn(state.sheets[idx])
  if (!next) return {}
  const sheets = state.sheets.slice()
  sheets[idx] = { ...next, version: state.sheets[idx].version + 1 }
  return remember ? { sheets, past: [...state.past.slice(-MAX_UNDO + 1), state.sheets], future: [] } : { sheets }
}

/** Re-detect the type of columns that were edited (unless the user fixed it). */
function retype(col: Column): Column {
  if (col.typeLocked) return col
  const t = inferType(col.values)
  if (t === col.type) return col
  // only switch automatically between "number" and "text"; keep ordinal as chosen
  if (col.type === 'ordinal' && t === 'nominal') return col
  return { ...col, type: t }
}

/** Replace a column name inside every option that refers to columns. */
function renameInOptions(items: Item[], catalog: Catalog | null, sheetId: string, from: string, to: string | null): Item[] {
  if (!catalog) return items
  return items.map((it) => {
    if (it.sheetId !== sheetId) return it
    const schema = it.kind === 'analysis'
      ? catalog.analyses.find((a) => a.id === it.analysisId)?.options
      : catalog.graphs.find((g) => g.id === it.graphType)?.options
    if (!schema) return it
    let changed = false
    const options = { ...it.options }
    for (const o of schema) {
      const v = options[o.name]
      if (o.type === 'variable' && v === from) { options[o.name] = to; changed = true }
      if (o.type === 'variables' && Array.isArray(v) && v.includes(from)) {
        options[o.name] = to ? v.map((n: string) => (n === from ? to : n)) : v.filter((n: string) => n !== from)
        changed = true
      }
    }
    return changed ? ({ ...it, options } as Item) : it
  })
}

export const useStore = create<State>((set, get) => ({
  name: 'Untitled project',
  sheets: [],
  items: [],
  layouts: [],
  catalog: null,
  catalogError: null,
  selection: { kind: 'welcome' },
  toasts: [],
  dark: false,
  defaultEngine: 'matplotlib',
  styleClipboard: null,
  past: [],
  future: [],

  loadCatalog: async () => {
    try {
      const catalog = await api.catalog()
      set({ catalog, catalogError: null })
    } catch (e) {
      set({ catalogError: 'Could not reach the Analysis UI server. Is it running?' })
    }
  },

  select: (selection) => set({ selection }),

  toast: (text, kind = 'info') => {
    const id = uid('t')
    set((s) => ({ toasts: [...s.toasts, { id, text, kind }] }))
    setTimeout(() => get().dismissToast(id), kind === 'error' ? 7000 : 3500)
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  setDark: (dark) => set({ dark }),
  setProjectName: (name) => set({ name }),
  newProject: () => set({ name: 'Untitled project', sheets: [], items: [], layouts: [], selection: { kind: 'welcome' }, past: [], future: [] }),
  loadProject: (p) => set({
    name: p.name || 'Untitled project',
    sheets: p.sheets ?? [],
    items: (p.items ?? []).map((it) => ({ ...it, status: it.status === 'running' ? 'idle' : it.status })),
    layouts: (p.layouts ?? []).map((l) => ({ ...l, status: l.status === 'running' ? 'idle' : l.status })),
    selection: p.sheets?.length ? { kind: 'sheet', id: p.sheets[0].id } : { kind: 'welcome' },
    past: [], future: [],
  }),

  addBlankSheet: () => set((s) => {
    const sheet = blankSheet(uniqueName('Data 1', s.sheets.map((x) => x.name)))
    return { sheets: [...s.sheets, sheet], selection: { kind: 'sheet', id: sheet.id } }
  }),

  addWireSheets: (wire) => set((s) => {
    const taken = s.sheets.map((x) => x.name)
    const added = wire.map((w) => {
      const sheet = sheetFromWire(w, taken)
      taken.push(sheet.name)
      return sheet
    })
    if (!added.length) return {}
    return { sheets: [...s.sheets, ...added], selection: { kind: 'sheet', id: added[0].id } }
  }),

  renameSheet: (id, name) => set((s) => ({
    sheets: s.sheets.map((x) => (x.id === id ? { ...x, name: uniqueName(name, s.sheets.filter((y) => y.id !== id).map((y) => y.name)) } : x)),
  })),

  deleteSheet: (id) => set((s) => {
    const sheets = s.sheets.filter((x) => x.id !== id)
    const items = s.items.filter((it) => it.sheetId !== id)
    const gone = new Set(s.items.filter((it) => it.sheetId === id).map((it) => it.id))
    return { sheets, items, layouts: s.layouts.map((l) => ({ ...l, graphIds: l.graphIds.filter((g) => !gone.has(g)) })), selection: sheets.length ? { kind: 'sheet', id: sheets[0].id } : { kind: 'welcome' } }
  }),

  duplicateSheet: (id) => set((s) => {
    const src = s.sheets.find((x) => x.id === id)
    if (!src) return {}
    const copy: Sheet = {
      ...src, id: uid('s'), name: uniqueName(`${src.name} copy`, s.sheets.map((x) => x.name)),
      columns: src.columns.map((c) => ({ ...c, id: uid('c'), values: c.values.slice() })),
    }
    return { sheets: [...s.sheets, copy], selection: { kind: 'sheet', id: copy.id } }
  }),

  editCells: (sheetId, edits) => set((s) => withSheet(s, sheetId, (sheet) => {
    if (!edits.length) return null
    const maxRow = Math.max(...edits.map((e) => e.row))
    const nrows = Math.max(sheet.nrows, maxRow + 1)
    const touched = new Set(edits.map((e) => e.col))
    const columns = sheet.columns.map((c, ci) => {
      if (!touched.has(ci) && nrows === sheet.nrows) return c
      const values = c.values.slice()
      while (values.length < nrows) values.push(null)
      for (const e of edits) if (e.col === ci) values[e.row] = e.value
      return touched.has(ci) ? retype({ ...c, values }) : { ...c, values }
    })
    return { ...sheet, columns, nrows }
  })),

  pasteBlock: (sheetId, col, row, block) => set((s) => withSheet(s, sheetId, (sheet) => {
    if (!block.length) return null
    const width = Math.max(...block.map((r) => r.length))
    const nrows = Math.max(sheet.nrows, row + block.length)
    let columns = sheet.columns.slice()
    while (columns.length < col + width) {
      columns.push(newColumn(uniqueName(columnLetter(columns.length), columns.map((c) => c.name)), nrows))
    }
    columns = columns.map((c, ci) => {
      const values = c.values.slice()
      while (values.length < nrows) values.push(null)
      if (ci < col || ci >= col + width) return { ...c, values }
      block.forEach((r, ri) => { if (ci - col < r.length) values[row + ri] = r[ci - col] })
      return retype({ ...c, values })
    })
    return { ...sheet, columns, nrows }
  })),

  addColumn: (sheetId, at) => set((s) => withSheet(s, sheetId, (sheet) => {
    const col = newColumn(uniqueName(columnLetter(sheet.columns.length), sheet.columns.map((c) => c.name)), sheet.nrows)
    const columns = sheet.columns.slice()
    columns.splice(at ?? columns.length, 0, col)
    return { ...sheet, columns }
  })),

  addWireColumns: (sheetId, cols) => set((s) => withSheet(s, sheetId, (sheet) => {
    const nrows = Math.max(sheet.nrows, ...cols.map((c) => c.values.length))
    const names = sheet.columns.map((c) => c.name)
    const pad = (v: Cell[]) => { const out = v.slice(); while (out.length < nrows) out.push(null); return out }
    const added: Column[] = cols.map((c) => {
      const name = uniqueName(c.name, names)
      names.push(name)
      return { id: uid('c'), name, type: c.type, values: pad(c.values), levels: c.levels }
    })
    return { ...sheet, nrows, columns: [...sheet.columns.map((c) => ({ ...c, values: pad(c.values) })), ...added] }
  })),

  deleteColumns: (sheetId, indexes) => set((s) => {
    const sheet = s.sheets.find((x) => x.id === sheetId)
    if (!sheet) return {}
    let items = s.items
    for (const i of indexes) if (sheet.columns[i]) items = renameInOptions(items, s.catalog, sheetId, sheet.columns[i].name, null)
    return { ...withSheet(s, sheetId, (sh) => ({ ...sh, columns: sh.columns.filter((_, i) => !indexes.includes(i)) })), items }
  }),

  renameColumn: (sheetId, index, name) => set((s) => {
    const sheet = s.sheets.find((x) => x.id === sheetId)
    const col = sheet?.columns[index]
    if (!sheet || !col) return {}
    const next = uniqueName(name, sheet.columns.filter((_, i) => i !== index).map((c) => c.name))
    if (next === col.name) return {}
    return {
      ...withSheet(s, sheetId, (sh) => ({ ...sh, columns: sh.columns.map((c, i) => (i === index ? { ...c, name: next } : c)) })),
      items: renameInOptions(s.items, s.catalog, sheetId, col.name, next),
    }
  }),

  setColumnType: (sheetId, index, type) => set((s) => withSheet(s, sheetId, (sheet) => ({
    ...sheet,
    columns: sheet.columns.map((c, i) => {
      if (i !== index) return c
      // numbers typed as text become real numbers again when switching to continuous
      const values = type === 'continuous'
        ? c.values.map((v) => (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v)) ? Number(v) : v))
        : c.values
      return { ...c, type, values, typeLocked: true }
    }),
  }))),

  setColumnLevels: (sheetId, index, levels) => set((s) => withSheet(s, sheetId, (sheet) => ({
    ...sheet, columns: sheet.columns.map((c, i) => (i === index ? { ...c, levels } : c)),
  }))),

  moveColumn: (sheetId, from, to) => set((s) => withSheet(s, sheetId, (sheet) => {
    if (from === to) return null
    const columns = sheet.columns.slice()
    const [c] = columns.splice(from, 1)
    columns.splice(to, 0, c)
    return { ...sheet, columns }
  })),

  addRows: (sheetId, count, at) => set((s) => withSheet(s, sheetId, (sheet) => ({
    ...sheet, nrows: sheet.nrows + count,
    columns: sheet.columns.map((c) => {
      const values = c.values.slice()
      values.splice(at ?? values.length, 0, ...new Array(count).fill(null))
      return { ...c, values }
    }),
  }), at !== undefined)),

  deleteRows: (sheetId, rows) => set((s) => withSheet(s, sheetId, (sheet) => {
    const drop = new Set(rows)
    return {
      ...sheet, nrows: Math.max(1, sheet.nrows - drop.size),
      columns: sheet.columns.map((c) => {
        const values = c.values.filter((_, i) => !drop.has(i))
        if (!values.length) values.push(null)
        return { ...c, values }
      }),
    }
  })),

  sortRows: (sheetId, col, descending) => set((s) => withSheet(s, sheetId, (sheet) => {
    const key = sheet.columns[col]
    if (!key) return null
    const order = Array.from({ length: sheet.nrows }, (_, i) => i)
    order.sort((a, b) => {
      const x = key.values[a], y = key.values[b]
      if (isBlank(x) && isBlank(y)) return a - b
      if (isBlank(x)) return 1
      if (isBlank(y)) return -1
      const cmp = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true })
      return (descending ? -cmp : cmp) || a - b
    })
    return { ...sheet, columns: sheet.columns.map((c) => ({ ...c, values: order.map((i) => c.values[i] ?? null) })) }
  })),

  undo: () => set((s) => {
    if (!s.past.length) return {}
    const prev = s.past[s.past.length - 1]
    // bump versions so dependent results refresh
    const sheets = prev.map((p) => ({ ...p, version: (s.sheets.find((x) => x.id === p.id)?.version ?? p.version) + 1 }))
    return { sheets, past: s.past.slice(0, -1), future: [s.sheets, ...s.future].slice(0, MAX_UNDO) }
  }),
  redo: () => set((s) => {
    if (!s.future.length) return {}
    const next = s.future[0]
    const sheets = next.map((p) => ({ ...p, version: (s.sheets.find((x) => x.id === p.id)?.version ?? p.version) + 1 }))
    return { sheets, future: s.future.slice(1), past: [...s.past, s.sheets] }
  }),

  addAnalysis: (sheetId, analysisId) => set((s) => {
    const def = s.catalog?.analyses.find((a) => a.id === analysisId)
    const sheet = s.sheets.find((x) => x.id === sheetId)
    if (!def) return {}
    const options = sheet ? autofill(def.options, sheet, { twoGroups: def.category === 'Compare two groups' }) : defaultsOf(def.options)
    const item: AnalysisItem = {
      id: uid('a'), kind: 'analysis', sheetId, analysisId, options, status: 'idle',
      name: uniqueName(def.name, s.items.map((i) => i.name)),
    }
    return { items: [...s.items, item], selection: { kind: 'analysis', id: item.id } }
  }),

  addGraph: (sheetId, graphType, given) => set((s) => {
    const def = s.catalog?.graphs.find((g) => g.id === graphType)
    const sheet = s.sheets.find((x) => x.id === sheetId)
    if (!def) return {}
    let options: Record<string, any>
    if (given && Object.keys(given).length) {
      options = { ...defaultsOf(def.options) }
      for (const [k, v] of Object.entries(given)) if (v !== null && v !== undefined && def.options.some((o) => o.name === k)) options[k] = v
    } else {
      options = sheet ? autofill(def.options, sheet) : defaultsOf(def.options)
    }
    const item: GraphItem = {
      id: uid('g'), kind: 'graph', sheetId, graphType, engine: s.defaultEngine, options, status: 'idle',
      name: uniqueName(def.name, s.items.map((i) => i.name)),
    }
    return { items: [...s.items, item], selection: { kind: 'graph', id: item.id } }
  }),

  patchItem: (id, patch) => set((s) => {
    const next: Partial<State> = { items: s.items.map((it) => (it.id === id ? ({ ...it, ...patch } as Item) : it)) }
    if ('engine' in patch && patch.engine) next.defaultEngine = patch.engine as any
    return next
  }),

  setOptions: (id, patch) => set((s) => ({
    items: s.items.map((it) => (it.id === id ? ({ ...it, options: { ...it.options, ...patch } } as Item) : it)),
  })),

  resetOptions: (id) => set((s) => ({
    items: s.items.map((it) => {
      if (it.id !== id) return it
      const sheet = s.sheets.find((x) => x.id === it.sheetId)
      const schema = it.kind === 'analysis'
        ? s.catalog?.analyses.find((a) => a.id === it.analysisId)?.options
        : s.catalog?.graphs.find((g) => g.id === it.graphType)?.options
      if (!schema) return it
      // keep the data roles, reset everything else
      const fresh = defaultsOf(schema)
      for (const o of schema) if (o.type === 'variable' || o.type === 'variables' || o.name === 'layout') fresh[o.name] = it.options[o.name]
      return { ...it, options: sheet ? fresh : fresh } as Item
    }),
  })),

  renameItem: (id, name) => set((s) => ({ items: s.items.map((it) => (it.id === id ? ({ ...it, name: name.trim() || it.name } as Item) : it)) })),

  addLayout: (graphIds) => set((s) => {
    const ids = graphIds ?? s.items.filter((i) => i.kind === 'graph').slice(0, 4).map((i) => i.id)
    const layout: LayoutItem = {
      id: uid('l'), name: uniqueName('Figure 1', s.layouts.map((l) => l.name)), graphIds: ids,
      cols: ids.length === 1 ? 1 : 2, width: 7, panelHeight: 2.8, gap: 0.15, labels: 'A', labelSize: 14, status: 'idle',
    }
    return { layouts: [...s.layouts, layout], selection: { kind: 'layout', id: layout.id } }
  }),
  patchLayout: (id, patch) => set((s) => ({ layouts: s.layouts.map((l) => (l.id === id ? { ...l, ...patch } : l)) })),
  deleteLayout: (id) => set((s) => ({
    layouts: s.layouts.filter((l) => l.id !== id),
    selection: s.selection.kind === 'layout' && s.selection.id === id ? (s.sheets.length ? { kind: 'sheet', id: s.sheets[0].id } : { kind: 'welcome' }) : s.selection,
  })),

  deleteItem: (id) => set((s) => {
    const it = s.items.find((x) => x.id === id)
    const selected = (s.selection.kind === 'analysis' || s.selection.kind === 'graph') && s.selection.id === id
    return {
      items: s.items.filter((x) => x.id !== id),
      layouts: s.layouts.map((l) => (l.graphIds.includes(id) ? { ...l, graphIds: l.graphIds.filter((g) => g !== id) } : l)),
      selection: selected && it ? (s.sheets.some((x) => x.id === it.sheetId) ? { kind: 'sheet', id: it.sheetId } : { kind: 'welcome' }) : s.selection,
    }
  }),

  duplicateItem: (id) => set((s) => {
    const it = s.items.find((x) => x.id === id)
    if (!it) return {}
    const copy = { ...it, id: uid(it.kind === 'analysis' ? 'a' : 'g'), name: uniqueName(`${it.name} copy`, s.items.map((i) => i.name)), options: JSON.parse(JSON.stringify(it.options)) } as Item
    const idx = s.items.findIndex((x) => x.id === id)
    const items = s.items.slice()
    items.splice(idx + 1, 0, copy)
    return { items, selection: { kind: copy.kind, id: copy.id } as Selection }
  }),

  changeGraphType: (id, graphType) => set((s) => {
    const it = s.items.find((x) => x.id === id)
    const next = s.catalog?.graphs.find((g) => g.id === graphType)
    if (!it || it.kind !== 'graph' || !next) return {}
    const prev = s.catalog?.graphs.find((g) => g.id === it.graphType)
    const options = prev ? migrateOptions(it.options, next.options, prev.options) : defaultsOf(next.options)
    const autoNamed = prev && it.name.startsWith(prev.name)
    return {
      items: s.items.map((x) => (x.id === id ? ({ ...it, graphType, options, name: autoNamed ? uniqueName(next.name, s.items.filter((i) => i.id !== id).map((i) => i.name)) : it.name } as Item) : x)),
    }
  }),

  copyStyle: (id) => {
    const s = get()
    const it = s.items.find((x) => x.id === id)
    const def = it?.kind === 'graph' ? s.catalog?.graphs.find((g) => g.id === it.graphType) : undefined
    if (!it || it.kind !== 'graph' || !def) return
    const options: Record<string, any> = {}
    const skip = new Set(['title', 'subtitle', 'caption', 'x_label', 'y_label', 'legend_title', 'group_colors',
      'x_min', 'x_max', 'y_min', 'y_max', 'x_tick_step', 'y_tick_step', 'hlines', 'vlines'])
    for (const o of def.options) {
      if (o.section === 'Data' || o.section === 'Significance' || o.type === 'variable' || o.type === 'variables' || skip.has(o.name)) continue
      options[o.name] = it.options[o.name]
    }
    set({ styleClipboard: { options, engine: it.engine } })
    s.toast('Style copied — open another graph and choose “Paste style”.')
  },

  pasteStyle: (id) => set((s) => {
    const clip = s.styleClipboard
    const it = s.items.find((x) => x.id === id)
    const def = it?.kind === 'graph' ? s.catalog?.graphs.find((g) => g.id === it.graphType) : undefined
    if (!clip || !it || it.kind !== 'graph' || !def) return {}
    const options = { ...it.options }
    for (const o of def.options) {
      const v = clip.options[o.name]
      if (v === undefined) continue
      if (o.type === 'choice' && !o.choices?.some((c) => c.value === v)) continue
      options[o.name] = v
    }
    return { items: s.items.map((x) => (x.id === id ? ({ ...it, options, engine: clip.engine } as Item) : x)) }
  }),
}))

// ---------------------------------------------------------------- persistence
const DB = 'analysis-ui'
const STORE = 'kv'

function idb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function idbSet(key: string, value: unknown) {
  const db = await idb()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite')
    tx.objectStore(STORE).put(value, key)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
  db.close()
}

async function idbGet<T>(key: string): Promise<T | undefined> {
  const db = await idb()
  const out = await new Promise<T | undefined>((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).get(key)
    req.onsuccess = () => resolve(req.result as T | undefined)
    req.onerror = () => reject(req.error)
  })
  db.close()
  return out
}

export function projectSnapshot(includeOutputs = true): Project & { format: string; version: number } {
  const s = useStore.getState()
  return {
    format: 'analysis-ui-project', version: 1, name: s.name, sheets: s.sheets,
    layouts: s.layouts.map((l) => (includeOutputs ? l : { ...l, svg: undefined, ranKey: undefined, status: 'idle' as const })),
    items: includeOutputs ? s.items : s.items.map((it) => {
      const { ...rest } = it as any
      delete rest.results; delete rest.svg; delete rest.meta; delete rest.ranKey
      return { ...rest, status: 'idle' }
    }),
  }
}

export async function restoreSession() {
  try {
    const saved = await idbGet<any>('session')
    if (saved?.sheets?.length) {
      useStore.getState().loadProject(saved)
      if (saved.selection) useStore.setState({ selection: saved.selection })
    }
    const prefs = await idbGet<any>('prefs')
    if (prefs) useStore.setState({ dark: !!prefs.dark, defaultEngine: prefs.defaultEngine ?? 'matplotlib' })
    else if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) useStore.setState({ dark: true })
  } catch { /* a fresh start is fine */ }
  let timer: number | undefined
  useStore.subscribe((s, prev) => {
    if (s.dark !== prev.dark || s.defaultEngine !== prev.defaultEngine) {
      idbSet('prefs', { dark: s.dark, defaultEngine: s.defaultEngine }).catch(() => {})
    }
    if (s.sheets === prev.sheets && s.items === prev.items && s.layouts === prev.layouts && s.name === prev.name && s.selection === prev.selection) return
    window.clearTimeout(timer)
    timer = window.setTimeout(() => {
      idbSet('session', { ...projectSnapshot(true), selection: useStore.getState().selection }).catch(() => {})
    }, 800)
  })
}

// Handy for debugging and automated UI tests.
;(window as any).__aui = useStore
