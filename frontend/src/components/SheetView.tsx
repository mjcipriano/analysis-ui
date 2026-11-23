import DataEditor, {
  CompactSelection, GridCell, GridCellKind, GridColumn, GridSelection, Item, Rectangle, Theme,
} from '@glideapps/glide-data-grid'
import '@glideapps/glide-data-grid/dist/index.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api } from '../api'
import { isBlank, levelsOf, parseCell } from '../data'
import { useStore } from '../store'
import type { Cell, ColType, Sheet } from '../types'
import { EditableText, Icon, MenuButton, Modal, Spinner } from './ui'

const glyph = (text: string, color: string) => (p: { fgColor: string; bgColor: string }) =>
  `<svg width="20" height="20" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><rect x="1" y="4" width="18" height="12" rx="3" fill="${color}"/><text x="10" y="13.300" font-family="Arial, sans-serif" font-size="7.500" font-weight="700" text-anchor="middle" fill="#fff">${text}</text></svg>`

const headerIcons = {
  continuous: glyph('123', '#3B82F6'),
  nominal: glyph('abc', '#F59E0B'),
  ordinal: glyph('1&lt;2', '#10B981'),
}

const darkGrid: Partial<Theme> = {
  accentColor: '#818CF8', accentLight: 'rgba(129,140,248,0.18)', textDark: '#E5E7EB', textMedium: '#9CA3AF', textLight: '#6B7280',
  textBubble: '#E5E7EB', bgIconHeader: '#9CA3AF', fgIconHeader: '#111827', textHeader: '#D1D5DB', textHeaderSelected: '#111827',
  bgCell: '#111827', bgCellMedium: '#0F172A', bgHeader: '#1F2937', bgHeaderHasFocus: '#374151', bgHeaderHovered: '#374151',
  bgBubble: '#374151', bgBubbleSelected: '#818CF8', bgSearchResult: '#423c00', borderColor: 'rgba(255,255,255,0.09)',
  drilldownBorder: 'rgba(255,255,255,0.3)', linkColor: '#93C5FD',
}
const lightGrid: Partial<Theme> = {
  accentColor: '#4F46E5', accentLight: 'rgba(79,70,229,0.10)', bgHeader: '#F8FAFC', bgHeaderHasFocus: '#EEF2FF', bgHeaderHovered: '#F1F5F9',
  textHeader: '#334155', borderColor: 'rgba(15,23,42,0.09)', textDark: '#0F172A',
}

function display(v: Cell): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(+v.toPrecision(12))
  return v
}

/** Column editor shown under a header: name, type, level order and column actions. */
function ColumnPopover({ sheet, index, bounds, onClose, onTransform }: {
  sheet: Sheet; index: number; bounds: Rectangle; onClose: () => void; onTransform: (index: number) => void
}) {
  const col = sheet.columns[index]
  const { renameColumn, setColumnType, setColumnLevels, deleteColumns, addColumn, sortRows } = useStore.getState()
  const [name, setName] = useState(col?.name ?? '')
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const down = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) { commit(); onClose() } }
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    const t = setTimeout(() => window.addEventListener('mousedown', down), 0)
    window.addEventListener('keydown', key)
    return () => { clearTimeout(t); window.removeEventListener('mousedown', down); window.removeEventListener('keydown', key) }
  })
  if (!col) return null
  const commit = () => { if (name.trim() && name.trim() !== col.name) renameColumn(sheet.id, index, name) }
  const levels = col.type === 'continuous' ? [] : levelsOf(col)
  const move = (i: number, d: number) => {
    const next = levels.slice()
    const j = i + d
    if (j < 0 || j >= next.length) return
    ;[next[i], next[j]] = [next[j], next[i]]
    setColumnLevels(sheet.id, index, next)
  }
  const types: { id: ColType; label: string; hint: string }[] = [
    { id: 'continuous', label: 'Numbers', hint: 'Measurements: continuous numeric values' },
    { id: 'nominal', label: 'Categories', hint: 'Group names or labels (nominal)' },
    { id: 'ordinal', label: 'Ordered', hint: 'Categories with a natural order (ordinal)' },
  ]
  const left = Math.max(8, Math.min(bounds.x, window.innerWidth - 300))
  const top = Math.min(bounds.y + bounds.height + 4, window.innerHeight - 420)
  return createPortal(
    <div ref={ref} className="popover column-popover" style={{ left, top }}>
      <label className="pop-label">Column name</label>
      <input autoFocus value={name} onChange={(e) => setName(e.target.value)} onFocus={(e) => e.currentTarget.select()}
        onKeyDown={(e) => { if (e.key === 'Enter') { commit(); onClose() } }} onBlur={commit} />
      <label className="pop-label">Data type</label>
      <div className="segmented wide">
        {types.map((t) => (
          <button key={t.id} title={t.hint} className={col.type === t.id ? 'on' : ''} onClick={() => setColumnType(sheet.id, index, t.id)}>{t.label}</button>
        ))}
      </div>
      {levels.length > 0 && levels.length <= 30 && (
        <>
          <label className="pop-label">Order of categories <span className="muted">(used in graphs and tables)</span></label>
          <div className="levels">
            {levels.map((l, i) => (
              <div key={l} className="level-row">
                <span>{l}</span>
                <button className="icon-btn small" disabled={i === 0} onClick={() => move(i, -1)} title="Move up">▲</button>
                <button className="icon-btn small" disabled={i === levels.length - 1} onClick={() => move(i, 1)} title="Move down">▼</button>
              </div>
            ))}
          </div>
        </>
      )}
      <div className="pop-actions">
        <button className="menu-item" onClick={() => { sortRows(sheet.id, index, false); onClose() }}><Icon name="sort" size={14} /> Sort rows ascending</button>
        <button className="menu-item" onClick={() => { sortRows(sheet.id, index, true); onClose() }}><Icon name="sort" size={14} /> Sort rows descending</button>
        {col.type === 'continuous' && <button className="menu-item" onClick={() => { onClose(); onTransform(index) }}><Icon name="function" size={14} /> Transform…</button>}
        <button className="menu-item" onClick={() => { addColumn(sheet.id, index); onClose() }}><Icon name="plus" size={14} /> Insert column left</button>
        <button className="menu-item" onClick={() => { addColumn(sheet.id, index + 1); onClose() }}><Icon name="plus" size={14} /> Insert column right</button>
        <button className="menu-item danger" onClick={() => { deleteColumns(sheet.id, [index]); onClose() }}><Icon name="trash" size={14} /> Delete column</button>
      </div>
    </div>,
    document.body,
  )
}

export function TransformDialog({ sheet, initial, onClose }: { sheet: Sheet; initial?: number; onClose: () => void }) {
  const catalog = useStore((s) => s.catalog)
  const { addWireColumns, addWireSheets, toast } = useStore.getState()
  const numeric = sheet.columns.filter((c) => c.type === 'continuous')
  const cats = sheet.columns.filter((c) => c.type !== 'continuous')
  const [tab, setTab] = useState<'transform' | 'reshape'>('transform')
  const [op, setOp] = useState('log10')
  const [cols, setCols] = useState<string[]>(initial !== undefined && sheet.columns[initial] ? [sheet.columns[initial].name] : numeric.slice(0, 1).map((c) => c.name))
  const [formula, setFormula] = useState('')
  const [name, setName] = useState('Computed')
  const [reshape, setReshape] = useState<'stack' | 'unstack'>('stack')
  const [idCol, setIdCol] = useState('')
  const [valueCol, setValueCol] = useState(numeric[0]?.name ?? '')
  const [groupCol, setGroupCol] = useState(cats[0]?.name ?? '')
  const [busy, setBusy] = useState(false)
  const toggle = (n: string) => setCols((c) => (c.includes(n) ? c.filter((x) => x !== n) : [...c, n]))
  const data = { columns: sheet.columns.map((c) => ({ name: c.name, type: c.type, values: c.values, levels: c.levels })) }

  const run = async () => {
    setBusy(true)
    try {
      if (tab === 'transform') {
        const res = await api.transform({ data, op, columns: cols, formula, name })
        if (res.status !== 'ok') { toast(res.message, 'error'); return }
        addWireColumns(sheet.id, res.columns)
        toast(`${res.columns.length} column${res.columns.length === 1 ? '' : 's'} added.`)
      } else {
        const body = reshape === 'stack'
          ? { data, op: 'stack', columns: cols, id_column: idCol || null }
          : { data, op: 'unstack', columns: [], id_column: groupCol, value_column: valueCol }
        const res = await api.transform(body)
        if (res.status !== 'ok') { toast(res.message, 'error'); return }
        addWireSheets([{ name: `${sheet.name} (${reshape === 'stack' ? 'stacked' : 'by group'})`, columns: res.columns }])
      }
      onClose()
    } finally {
      setBusy(false)
    }
  }
  const groups = [...new Set(catalog?.transforms.map((t) => t.group) ?? [])]
  return (
    <Modal title="Transform data" onClose={onClose} width={600}
      footer={<>
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy} onClick={run}>{busy && <Spinner />} {tab === 'transform' ? 'Add columns' : 'Create new table'}</button>
      </>}>
      <div className="tabs">
        <button className={tab === 'transform' ? 'on' : ''} onClick={() => setTab('transform')}>Transform &amp; normalize</button>
        <button className={tab === 'reshape' ? 'on' : ''} onClick={() => setTab('reshape')}>Reshape table</button>
      </div>
      {tab === 'transform' && (
        <div className="form-grid">
          <label>Operation</label>
          <select value={op} onChange={(e) => setOp(e.target.value)}>
            {groups.map((g) => (
              <optgroup key={g} label={g}>
                {catalog?.transforms.filter((t) => t.group === g).map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
              </optgroup>
            ))}
          </select>
          {op === 'formula' ? (
            <>
              <label>Formula</label>
              <div>
                <input value={formula} onChange={(e) => setFormula(e.target.value)} placeholder="e.g.  (`Drug A` - Control) / Control * 100" />
                <div className="field-help">Use column names (in `backticks` if they contain spaces), + − * / **, and functions such as log, log10, sqrt, exp, abs.</div>
              </div>
              <label>New column name</label>
              <input value={name} onChange={(e) => setName(e.target.value)} />
            </>
          ) : (
            <>
              <label>Columns</label>
              <div className="checklist">
                {numeric.map((c) => (
                  <label key={c.id} className="field check"><input type="checkbox" checked={cols.includes(c.name)} onChange={() => toggle(c.name)} /><span>{c.name}</span></label>
                ))}
                {!numeric.length && <span className="muted">This table has no numeric columns.</span>}
              </div>
            </>
          )}
          <div />
          <div className="muted small">Results are added as new columns; your original data are never changed.</div>
        </div>
      )}
      {tab === 'reshape' && (
        <div className="form-grid">
          <label>Direction</label>
          <div className="segmented wide">
            <button className={reshape === 'stack' ? 'on' : ''} onClick={() => setReshape('stack')}>Stack columns → Group + Value</button>
            <button className={reshape === 'unstack' ? 'on' : ''} onClick={() => setReshape('unstack')}>Split by group → one column each</button>
          </div>
          {reshape === 'stack' ? (
            <>
              <label>Columns to stack</label>
              <div className="checklist">
                {numeric.map((c) => (
                  <label key={c.id} className="field check"><input type="checkbox" checked={cols.includes(c.name)} onChange={() => toggle(c.name)} /><span>{c.name}</span></label>
                ))}
              </div>
              <label>Keep ID column</label>
              <select value={idCol} onChange={(e) => setIdCol(e.target.value)}>
                <option value="">(none)</option>
                {sheet.columns.filter((c) => !cols.includes(c.name)).map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
              </select>
            </>
          ) : (
            <>
              <label>Values</label>
              <select value={valueCol} onChange={(e) => setValueCol(e.target.value)}>{numeric.map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}</select>
              <label>Grouping column</label>
              <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>{sheet.columns.map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}</select>
            </>
          )}
          <div />
          <div className="muted small">A new data table is created; the original stays as it is.</div>
        </div>
      )}
    </Modal>
  )
}

export function SheetView({ sheet, onAnalyze, onGraph }: { sheet: Sheet; onAnalyze: () => void; onGraph: () => void }) {
  const dark = useStore((s) => s.dark)
  const canUndo = useStore((s) => s.past.length > 0)
  const canRedo = useStore((s) => s.future.length > 0)
  const { editCells, pasteBlock, addColumn, addRows, deleteRows, deleteColumns, renameSheet, renameColumn, moveColumn, undo, redo, duplicateSheet, deleteSheet, toast } = useStore.getState()
  const [widths, setWidths] = useState<Record<string, number>>({})
  const [popover, setPopover] = useState<{ index: number; bounds: Rectangle } | null>(null)
  const [transform, setTransform] = useState<{ index?: number } | null>(null)
  const [selection, setSelection] = useState<GridSelection>({ columns: CompactSelection.empty(), rows: CompactSelection.empty() })

  const sheetRef = useRef(sheet)
  sheetRef.current = sheet

  const columns: GridColumn[] = useMemo(() => sheet.columns.map((c) => ({
    id: c.id, title: c.name, width: widths[c.id] ?? Math.max(110, Math.min(260, c.name.length * 9 + 60)),
    icon: c.type, hasMenu: true,
  })), [sheet.columns, widths])

  const getCellContent = useCallback(([col, row]: Item): GridCell => {
    const c = sheetRef.current.columns[col]
    const v = c?.values[row] ?? null
    const text = display(v)
    const mismatch = c?.type === 'continuous' && typeof v === 'string' && v !== ''
    return {
      kind: GridCellKind.Text, data: text, displayData: text, allowOverlay: true,
      contentAlign: typeof v === 'number' ? 'right' : 'left',
      themeOverride: mismatch ? { textDark: '#DC2626' } : undefined,
    }
  }, [])

  const onCellsEdited = useCallback((edits: readonly { location: Item; value: any }[]) => {
    const s = sheetRef.current
    const list = edits.map((e) => ({ col: e.location[0], row: e.location[1], value: parseCell(String(e.value?.data ?? '')) }))
    const before = list.map((e) => s.columns[e.col]?.type)
    editCells(s.id, list)
    const after = useStore.getState().sheets.find((x) => x.id === s.id)
    list.forEach((e, i) => {
      const c = after?.columns[e.col]
      if (c && before[i] && c.type !== before[i] && i === list.findIndex((x) => x.col === e.col)) {
        toast(`“${c.name}” is now treated as ${c.type === 'continuous' ? 'numbers' : 'categories'}.`)
      }
    })
    if (list.some((e) => e.row >= s.nrows - 1)) addRows(s.id, 10)
    return true
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const onPaste = useCallback((target: Item, values: readonly (readonly string[])[]) => {
    const s = sheetRef.current
    let block: Cell[][] = values.map((r) => r.map((t) => parseCell(t)))
    while (block.length && block[block.length - 1].every((v) => v === null)) block.pop()
    if (!block.length) return false
    const [col, row] = target
    const width = Math.max(...block.map((r) => r.length))
    // A header row pasted into the top of empty columns becomes the column names.
    const targetEmpty = s.columns.slice(col, col + width).every((c) => c.values.every(isBlank))
    const looksLikeHeader = row === 0 && block.length > 1 && targetEmpty
      && block[0].every((v) => typeof v === 'string') && block.slice(1).some((r) => r.some((v) => typeof v === 'number'))
    if (looksLikeHeader) {
      const header = block[0] as string[]
      block = block.slice(1)
      pasteBlock(s.id, col, row, block)
      header.forEach((h, i) => renameColumn(s.id, col + i, h))
    } else {
      pasteBlock(s.id, col, row, block)
    }
    return false
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const onDelete = useCallback((sel: GridSelection) => {
    const s = sheetRef.current
    const rows = sel.rows.toArray()
    if (rows.length) {
      deleteRows(s.id, rows)
      setSelection({ columns: CompactSelection.empty(), rows: CompactSelection.empty() })
      return false
    }
    const cols = sel.columns.toArray()
    if (cols.length) {
      deleteColumns(s.id, cols)
      setSelection({ columns: CompactSelection.empty(), rows: CompactSelection.empty() })
      return false
    }
    return true
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo() }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo() }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const filled = sheet.columns.reduce((n, c) => Math.max(n, c.values.reduce<number>((m, v, i) => (isBlank(v) ? m : i + 1), 0)), 0)
  const hasData = filled > 0

  return (
    <div className="sheet-view">
      <div className="content-head">
        <EditableText className="content-title" value={sheet.name} onCommit={(v) => renameSheet(sheet.id, v)} />
        <span className="muted small">{filled.toLocaleString()} rows × {sheet.columns.length} columns</span>
        <div className="content-actions">
          <button className="icon-btn" title="Undo (Ctrl+Z)" disabled={!canUndo} onClick={undo}><Icon name="undo" /></button>
          <button className="icon-btn" title="Redo (Ctrl+Shift+Z)" disabled={!canRedo} onClick={redo}><Icon name="redo" /></button>
          <button className="btn" onClick={() => addColumn(sheet.id)}><Icon name="plus" /> Column</button>
          <button className="btn" onClick={() => setTransform({})} disabled={!hasData}><Icon name="function" /> Transform</button>
          <button className="btn primary" onClick={onAnalyze} disabled={!hasData}><Icon name="results" /> Analyze</button>
          <button className="btn accent" onClick={onGraph} disabled={!hasData}><Icon name="graph" /> Graph</button>
          <MenuButton className="btn" items={[
            { label: 'Duplicate table', icon: 'copy', onClick: () => duplicateSheet(sheet.id) },
            { label: 'Add 100 rows', icon: 'plus', onClick: () => addRows(sheet.id, 100) },
            { divider: true },
            { label: 'Delete table', icon: 'trash', danger: true, onClick: () => { if (confirm(`Delete “${sheet.name}” and its analyses and graphs?`)) deleteSheet(sheet.id) } },
          ]}><Icon name="more" /></MenuButton>
        </div>
      </div>
      {!hasData && (
        <div className="hint-bar">
          <Icon name="info" size={15} />
          <span>Type or paste your data (Ctrl/⌘+V) — a pasted header row becomes the column names. Use the ▾ on a column header (or right-click it) to rename it or change its type.
            Put each group in its own column <i>or</i> use one column for values and another for group names; both work everywhere.</span>
        </div>
      )}
      <div className="grid-wrap">
        <DataEditor
          columns={columns}
          rows={sheet.nrows}
          getCellContent={getCellContent}
          onCellsEdited={onCellsEdited as any}
          onCellEdited={(location, value) => { onCellsEdited([{ location, value }]) }}
          onPaste={onPaste}
          onDelete={onDelete}
          getCellsForSelection={true}
          gridSelection={selection}
          onGridSelectionChange={setSelection}
          rowMarkers="both"
          fillHandle
          smoothScrollX
          smoothScrollY
          headerIcons={headerIcons as any}
          headerHeight={38}
          rowHeight={30}
          onHeaderMenuClick={(index, bounds) => setPopover({ index, bounds })}
          onHeaderContextMenu={(index, e) => { e.preventDefault(); setPopover({ index, bounds: e.bounds }) }}
          onColumnResize={(c, w) => setWidths((s) => ({ ...s, [c.id as string]: w }))}
          onColumnMoved={(from, to) => moveColumn(sheet.id, from, to)}
          trailingRowOptions={{ hint: 'Add row…', sticky: false, tint: true }}
          onRowAppended={() => addRows(sheet.id, 1)}
          rightElement={<button className="add-col-btn" title="Add column" onClick={() => addColumn(sheet.id)}><Icon name="plus" size={15} /></button>}
          rightElementProps={{ sticky: false, fill: false }}
          theme={dark ? darkGrid : lightGrid}
          width="100%"
          height="100%"
        />
      </div>
      {popover && <ColumnPopover sheet={sheet} index={popover.index} bounds={popover.bounds} onClose={() => setPopover(null)} onTransform={(index) => setTransform({ index })} />}
      {transform && <TransformDialog sheet={sheet} initial={transform.index} onClose={() => setTransform(null)} />}
    </div>
  )
}
