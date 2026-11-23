import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { download, safeName, svgUrl } from '../format'
import { graphSpec, wireForItem } from '../runner'
import { useStore } from '../store'
import type { GraphItem, LayoutItem } from '../types'
import { EditableText, Icon, MenuButton, Modal, Spinner } from './ui'

/** The request body for the backend: every panel carries its own graph spec and data. */
function useLayoutSpec(layout: LayoutItem) {
  const items = useStore((s) => s.items)
  const sheets = useStore((s) => s.sheets)
  const catalog = useStore((s) => s.catalog)
  return useMemo(() => {
    const graphs = layout.graphIds.map((id) => items.find((i) => i.id === id)).filter((g): g is GraphItem => !!g && g.kind === 'graph')
    const panels = graphs.map((g) => {
      const def = catalog?.graphs.find((d) => d.id === g.graphType)
      const sheet = sheets.find((s) => s.id === g.sheetId)
      return { graph: graphSpec(g), data: { columns: def && sheet ? wireForItem(def.options, g.options, sheet) : [] } }
    })
    const spec = { panels, cols: layout.cols, width: layout.width, panel_height: layout.panelHeight, gap: layout.gap, labels: layout.labels, label_size: layout.labelSize }
    const key = JSON.stringify([layout.cols, layout.width, layout.panelHeight, layout.gap, layout.labels, layout.labelSize,
      graphs.map((g) => [g.id, g.graphType, g.engine, g.options, sheets.find((s) => s.id === g.sheetId)?.version])])
    return { spec, key, count: graphs.length }
  }, [layout.graphIds, layout.cols, layout.width, layout.panelHeight, layout.gap, layout.labels, layout.labelSize, items, sheets, catalog])
}

function ExportLayout({ layout, spec, onClose }: { layout: LayoutItem; spec: object; onClose: () => void }) {
  const toast = useStore((s) => s.toast)
  const [format, setFormat] = useState('png')
  const [dpi, setDpi] = useState(300)
  const [busy, setBusy] = useState(false)
  const go = async () => {
    setBusy(true)
    try { download(await api.exportLayout(spec, format, dpi), `${safeName(layout.name)}.${format}`); onClose() }
    catch (e: any) { toast(e?.message ?? 'Export failed.', 'error') }
    finally { setBusy(false) }
  }
  const formats = [['png', 'PNG', 'Slides, documents, web'], ['pdf', 'PDF', 'Print (image at the chosen resolution)'], ['svg', 'SVG', 'Vector — edit in Illustrator / Inkscape'], ['tiff', 'TIFF', 'Journal submission']]
  return (
    <Modal title="Export figure" onClose={onClose} width={520}
      footer={<><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" disabled={busy} onClick={go}>{busy ? <Spinner /> : <Icon name="download" />} Export</button></>}>
      <div className="export-formats">
        {formats.map(([id, label, hint]) => (
          <button key={id} className={`export-format ${format === id ? 'on' : ''}`} onClick={() => setFormat(id)}><b>{label}</b><span>{hint}</span></button>
        ))}
      </div>
      {format !== 'svg' && (
        <div className="form-grid">
          <label>Resolution</label>
          <div className="segmented">{[150, 300, 600].map((d) => <button key={d} className={dpi === d ? 'on' : ''} onClick={() => setDpi(d)}>{d} dpi</button>)}</div>
        </div>
      )}
    </Modal>
  )
}

function Num({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div className="field">
      <div className="field-label">{label}</div>
      <div className="number-field">
        <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
        <input type="number" min={min} max={max} step={step} value={value} onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v) && v >= min && v <= max) onChange(v) }} />
      </div>
    </div>
  )
}

export function LayoutView({ layout }: { layout: LayoutItem }) {
  const graphs = useStore((s) => s.items).filter((i): i is GraphItem => i.kind === 'graph')
  const { patchLayout, deleteLayout, select } = useStore.getState()
  const { spec, key, count } = useLayoutSpec(layout)
  const [exporting, setExporting] = useState(false)
  const stage = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 800, h: 600 })
  const drag = useRef<number | null>(null)

  useLayoutEffect(() => {
    const el = stage.current
    if (!el) return
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (layout.ranKey === key) return
    if (!count) { patchLayout(layout.id, { status: 'incomplete', ranKey: key }); return }
    const ctrl = new AbortController()
    const t = window.setTimeout(async () => {
      patchLayout(layout.id, { status: 'running' })
      try {
        const res = await api.renderLayout(spec, ctrl.signal)
        if (ctrl.signal.aborted) return
        if (res.status === 'ok') patchLayout(layout.id, { status: 'ok', svg: res.svg, message: undefined, ranKey: key })
        else patchLayout(layout.id, { status: 'error', message: res.message, ranKey: key })
      } catch (e: any) {
        if (!ctrl.signal.aborted) patchLayout(layout.id, { status: 'error', message: e?.message ?? 'The request failed.', ranKey: key })
      }
    }, 300)
    return () => { window.clearTimeout(t); ctrl.abort() }
  }, [key, layout.ranKey, layout.id])   // eslint-disable-line react-hooks/exhaustive-deps

  const rows = Math.max(1, Math.ceil(count / Math.max(1, Math.min(layout.cols, count || 1))))
  const hIn = rows * layout.panelHeight + (rows - 1) * layout.gap
  const natW = layout.width * 96, natH = hIn * 96
  const scale = Math.max(0.15, Math.min((box.w - 56) / natW, (box.h - 56) / natH, 2))
  const set = (patch: Partial<LayoutItem>) => patchLayout(layout.id, patch)
  const toggle = (id: string) => set({ graphIds: layout.graphIds.includes(id) ? layout.graphIds.filter((g) => g !== id) : [...layout.graphIds, id] })
  const chosen = layout.graphIds.map((id) => graphs.find((g) => g.id === id)).filter((g): g is GraphItem => !!g)
  const label = (i: number) => (layout.labels === 'A' ? String.fromCharCode(65 + (i % 26)) : layout.labels === 'a' ? String.fromCharCode(97 + (i % 26)) : layout.labels === '1' ? String(i + 1) : '•')

  return (
    <div className="split graph-split">
      <section className="content">
        <div className="content-head">
          <EditableText className="content-title" value={layout.name} onCommit={(v) => set({ name: v })} />
          <span className="muted small">{layout.width} × {hIn.toFixed(2)} in</span>
          <div className="content-actions">
            {layout.status === 'running' && <span className="status"><Spinner /> Composing…</span>}
            <button className="btn primary" disabled={!count} onClick={() => setExporting(true)}><Icon name="download" /> Export</button>
            <MenuButton className="btn" items={[{ label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteLayout(layout.id) }]}><Icon name="more" /></MenuButton>
          </div>
        </div>
        <div className="graph-stage" ref={stage}>
          {!count && (
            <div className="empty-state"><Icon name="layers" size={28} /><h3>Build a multi-panel figure</h3>
              <p>{graphs.length ? 'Tick the graphs to include on the right.' : 'Create some graphs first, then combine them here.'}</p></div>
          )}
          {count > 0 && layout.status === 'error' && (
            <div className="error-box"><Icon name="warn" size={18} /><div><div className="error-title">This figure could not be composed</div><p>{layout.message}</p></div></div>
          )}
          {count > 0 && layout.svg && layout.status !== 'error' && (
            <div className={`graph-paper ${layout.status === 'running' ? 'stale' : ''}`} style={{ width: natW * scale, height: natH * scale }}>
              <img src={svgUrl(layout.svg)} alt={layout.name} draggable={false} style={{ width: '100%', height: '100%' }} />
            </div>
          )}
          {count > 0 && !layout.svg && layout.status !== 'error' && <div className="empty-state"><Spinner size={22} /><p>Composing…</p></div>}
        </div>
      </section>
      <aside className="panel right">
        <div className="panel-head"><div><div className="panel-title">Figure layout</div><div className="panel-sub">Arrange graphs on one page with panel labels</div></div></div>
        <div className="options">
          <div className="opt-section plain"><div className="opt-section-body">
            <div className="field">
              <div className="field-label">Panels (drag to reorder)</div>
              <div className="layout-list">
                {chosen.map((g, i) => (
                  <div key={g.id} className="layout-row" draggable onDragStart={() => { drag.current = i }} onDragOver={(e) => e.preventDefault()}
                    onDrop={() => {
                      const from = drag.current
                      if (from === null || from === i) return
                      const next = layout.graphIds.filter((id) => graphs.some((x) => x.id === id))
                      const [m] = next.splice(from, 1); next.splice(i, 0, m); drag.current = null; set({ graphIds: next })
                    }}>
                    <span className="layout-badge">{label(i)}</span>
                    <button className="link layout-name" onClick={() => select({ kind: 'graph', id: g.id })} title="Open this graph">{g.name}</button>
                    <button className="icon-btn small" onClick={() => toggle(g.id)} title="Remove from figure"><Icon name="x" size={12} /></button>
                  </div>
                ))}
                {!chosen.length && <div className="muted small">No graphs yet.</div>}
              </div>
            </div>
            {graphs.filter((g) => !layout.graphIds.includes(g.id)).length > 0 && (
              <div className="field">
                <div className="field-label">Add a graph</div>
                <select value="" onChange={(e) => { if (e.target.value) toggle(e.target.value) }}>
                  <option value="">Choose…</option>
                  {graphs.filter((g) => !layout.graphIds.includes(g.id)).map((g) => <option key={g.id} value={g.id}>{g.name}</option>)}
                </select>
              </div>
            )}
          </div></div>
          <div className="opt-section"><div className="opt-section-head" style={{ cursor: 'default' }}><span>Page</span></div><div className="opt-section-body">
            <Num label="Columns" value={layout.cols} min={1} max={6} step={1} onChange={(v) => set({ cols: v })} />
            <Num label="Page width (inches)" value={layout.width} min={3} max={20} step={0.25} onChange={(v) => set({ width: v })} />
            <Num label="Panel height (inches)" value={layout.panelHeight} min={1.5} max={10} step={0.1} onChange={(v) => set({ panelHeight: v })} />
            <Num label="Gap between panels (inches)" value={layout.gap} min={0} max={1} step={0.05} onChange={(v) => set({ gap: v })} />
            <div className="field">
              <div className="field-label">Panel labels</div>
              <div className="segmented">
                {(['A', 'a', '1', 'none'] as const).map((l) => <button key={l} className={layout.labels === l ? 'on' : ''} onClick={() => set({ labels: l })}>{l === 'none' ? 'None' : l === 'A' ? 'A, B, C' : l === 'a' ? 'a, b, c' : '1, 2, 3'}</button>)}
              </div>
            </div>
            <Num label="Label size (pt)" value={layout.labelSize} min={6} max={36} step={1} onChange={(v) => set({ labelSize: v })} />
            <div className="field-help">Each graph is redrawn at the size of its panel, so text stays the same size in every panel. Style the graphs themselves in their own Format panels.</div>
          </div></div>
        </div>
      </aside>
      {exporting && <ExportLayout layout={layout} spec={spec} onClose={() => setExporting(false)} />}
    </div>
  )
}
