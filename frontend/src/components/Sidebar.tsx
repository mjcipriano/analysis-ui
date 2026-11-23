import { useState } from 'react'
import { useStore, Item } from '../store'
import type { Sheet } from '../types'
import { Icon, Menu, MenuEntry, Spinner } from './ui'

function Row({ active, icon, label, sub, status, onClick, menu, onRename }: {
  active: boolean; icon: string; label: string; sub?: string; status?: string; onClick: () => void; menu: MenuEntry[]; onRename: (v: string) => void
}) {
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(label)
  const commit = () => { setEditing(false); if (draft.trim() && draft !== label) onRename(draft.trim()) }
  return (
    <div className={`nav-row ${active ? 'active' : ''}`} onClick={onClick}
      onDoubleClick={() => { setDraft(label); setEditing(true) }}
      onContextMenu={(e) => { e.preventDefault(); setAt({ x: e.clientX, y: e.clientY }) }}>
      <Icon name={icon} size={15} />
      {editing
        ? <input autoFocus className="nav-edit" value={draft} onChange={(e) => setDraft(e.target.value)} onBlur={commit} onClick={(e) => e.stopPropagation()}
            onFocus={(e) => e.currentTarget.select()} onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') setEditing(false) }} />
        : <span className="nav-label" title={label}>{label}{sub && <span className="nav-sub">{sub}</span>}</span>}
      {status === 'running' && <Spinner size={11} />}
      {status === 'error' && <span className="dot error" title="Needs attention" />}
      <button className="icon-btn small nav-more" onClick={(e) => { e.stopPropagation(); const r = e.currentTarget.getBoundingClientRect(); setAt({ x: r.left, y: r.bottom + 2 }) }}><Icon name="more" size={14} /></button>
      {at && <Menu at={at} onClose={() => setAt(null)} items={[{ label: 'Rename', icon: 'edit', onClick: () => { setDraft(label); setEditing(true) } }, ...menu]} />}
    </div>
  )
}

export function Sidebar({ onAnalyze, onGraph, onNewData }: { onAnalyze: (s: Sheet) => void; onGraph: (s: Sheet) => void; onNewData: (e: React.MouseEvent) => void }) {
  const sheets = useStore((s) => s.sheets)
  const items = useStore((s) => s.items)
  const layouts = useStore((s) => s.layouts)
  const selection = useStore((s) => s.selection)
  const { addLayout, patchLayout, deleteLayout } = useStore.getState()
  const { select, renameSheet, deleteSheet, duplicateSheet, renameItem, deleteItem, duplicateItem } = useStore.getState()
  const activeSheet = (() => {
    if (selection.kind === 'sheet') return sheets.find((s) => s.id === selection.id)
    if (selection.kind === 'analysis' || selection.kind === 'graph') {
      const it = items.find((i) => i.id === selection.id)
      return sheets.find((s) => s.id === it?.sheetId)
    }
    return sheets[0]
  })()
  const multi = sheets.length > 1
  const sheetName = (it: Item) => (multi ? sheets.find((s) => s.id === it.sheetId)?.name : undefined)
  const analyses = items.filter((i) => i.kind === 'analysis')
  const graphs = items.filter((i) => i.kind === 'graph')
  const itemMenu = (it: Item): MenuEntry[] => [
    { label: 'Duplicate', icon: 'copy', onClick: () => duplicateItem(it.id) },
    { divider: true },
    { label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteItem(it.id) },
  ]
  return (
    <nav className="sidebar">
      <div className="nav-section">
        <div className="nav-head"><span>Data tables</span><button className="icon-btn small" title="New data table" onClick={onNewData}><Icon name="plus" size={14} /></button></div>
        {sheets.map((s) => (
          <Row key={s.id} icon="table" label={s.name} active={selection.kind === 'sheet' && selection.id === s.id}
            onClick={() => select({ kind: 'sheet', id: s.id })} onRename={(v) => renameSheet(s.id, v)}
            menu={[
              { label: 'Analyze…', icon: 'results', onClick: () => onAnalyze(s) },
              { label: 'New graph…', icon: 'graph', onClick: () => onGraph(s) },
              { label: 'Duplicate', icon: 'copy', onClick: () => duplicateSheet(s.id) },
              { divider: true },
              { label: 'Delete', icon: 'trash', danger: true, onClick: () => { if (confirm(`Delete “${s.name}” and its analyses and graphs?`)) deleteSheet(s.id) } },
            ]} />
        ))}
        {!sheets.length && <div className="nav-empty">No data yet</div>}
      </div>
      <div className="nav-section">
        <div className="nav-head"><span>Results</span>
          <button className="icon-btn small" title="New analysis" disabled={!activeSheet} onClick={() => activeSheet && onAnalyze(activeSheet)}><Icon name="plus" size={14} /></button></div>
        {analyses.map((it) => (
          <Row key={it.id} icon="results" label={it.name} sub={sheetName(it)} status={it.status} active={selection.kind === 'analysis' && selection.id === it.id}
            onClick={() => select({ kind: 'analysis', id: it.id })} onRename={(v) => renameItem(it.id, v)} menu={itemMenu(it)} />
        ))}
        {!analyses.length && <div className="nav-empty">{sheets.length ? 'Click + to analyze your data' : 'No analyses yet'}</div>}
      </div>
      <div className="nav-section">
        <div className="nav-head"><span>Graphs</span>
          <button className="icon-btn small" title="New graph" disabled={!activeSheet} onClick={() => activeSheet && onGraph(activeSheet)}><Icon name="plus" size={14} /></button></div>
        {graphs.map((it) => (
          <Row key={it.id} icon="graph" label={it.name} sub={sheetName(it)} status={it.status} active={selection.kind === 'graph' && selection.id === it.id}
            onClick={() => select({ kind: 'graph', id: it.id })} onRename={(v) => renameItem(it.id, v)} menu={itemMenu(it)} />
        ))}
        {!graphs.length && <div className="nav-empty">{sheets.length ? 'Click + to make a graph' : 'No graphs yet'}</div>}
      </div>
      <div className="nav-section">
        <div className="nav-head"><span>Figure layouts</span>
          <button className="icon-btn small" title="Combine graphs into a multi-panel figure" disabled={!graphs.length} onClick={() => addLayout()}><Icon name="plus" size={14} /></button></div>
        {layouts.map((l) => (
          <Row key={l.id} icon="layers" label={l.name} status={l.status} active={selection.kind === 'layout' && selection.id === l.id}
            onClick={() => select({ kind: 'layout', id: l.id })} onRename={(v) => patchLayout(l.id, { name: v })}
            menu={[{ divider: true }, { label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteLayout(l.id) }]} />
        ))}
        {!layouts.length && <div className="nav-empty">{graphs.length ? 'Click + to combine graphs' : 'No layouts yet'}</div>}
      </div>
    </nav>
  )
}
