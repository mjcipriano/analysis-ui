import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { api } from '../api'
import { download, safeName, svgUrl } from '../format'
import { graphSpec, useGraphRunner, wireForItem } from '../runner'
import { useStore } from '../store'
import type { GraphDef, GraphItem, Palette, Sheet } from '../types'
import { OptionsForm } from './OptionsForm'
import { EditableText, Icon, MenuButton, Modal, Spinner } from './ui'

const FORMATS = [
  { id: 'png', label: 'PNG', hint: 'Best for slides, documents and the web' },
  { id: 'pdf', label: 'PDF', hint: 'Vector — best for journals and print' },
  { id: 'svg', label: 'SVG', hint: 'Vector — edit in Illustrator / Inkscape' },
  { id: 'tiff', label: 'TIFF', hint: 'High-resolution raster required by some journals' },
  { id: 'jpg', label: 'JPEG', hint: 'Small files; not ideal for line art' },
]

function ExportDialog({ item, def, sheet, onClose }: { item: GraphItem; def: GraphDef; sheet: Sheet; onClose: () => void }) {
  const toast = useStore((s) => s.toast)
  const [format, setFormat] = useState('png')
  const [dpi, setDpi] = useState(300)
  const [editable, setEditable] = useState(false)
  const [transparent, setTransparent] = useState(!!item.options.transparent)
  const [busy, setBusy] = useState(false)
  const raster = ['png', 'tiff', 'jpg'].includes(format)
  const w = Number(item.options.width ?? 5), h = Number(item.options.height ?? 4)
  const go = async () => {
    setBusy(true)
    try {
      const spec = graphSpec({ ...item, options: { ...item.options, transparent: transparent && format !== 'jpg' } })
      const blob = await api.exportGraph(spec, wireForItem(def.options, item.options, sheet), format, dpi, editable)
      download(blob, `${safeName(item.name)}.${format}`)
      onClose()
    } catch (e: any) {
      toast(e?.message ?? 'Export failed.', 'error')
    } finally {
      setBusy(false)
    }
  }
  return (
    <Modal title="Export graph" onClose={onClose} width={520}
      footer={<>
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={busy} onClick={go}>{busy ? <Spinner /> : <Icon name="download" />} Export</button>
      </>}>
      <div className="export-formats">
        {FORMATS.map((f) => (
          <button key={f.id} className={`export-format ${format === f.id ? 'on' : ''}`} onClick={() => setFormat(f.id)}>
            <b>{f.label}</b><span>{f.hint}</span>
          </button>
        ))}
      </div>
      <div className="form-grid">
        {raster && (
          <>
            <label>Resolution</label>
            <div className="segmented">
              {[150, 300, 600, 1200].map((d) => <button key={d} className={dpi === d ? 'on' : ''} onClick={() => setDpi(d)}>{d} dpi</button>)}
            </div>
            <label>Pixel size</label>
            <div className="muted">{Math.round(w * dpi)} × {Math.round(h * dpi)} px &nbsp;({w} × {h} in)</div>
          </>
        )}
        {!raster && (
          <>
            <label>Text</label>
            <label className="field check"><input type="checkbox" checked={editable} onChange={(e) => setEditable(e.target.checked)} />
              <span>Keep text editable (needs the font installed where you open the file)</span></label>
          </>
        )}
        {format !== 'jpg' && (
          <>
            <label>Background</label>
            <label className="field check"><input type="checkbox" checked={transparent} onChange={(e) => setTransparent(e.target.checked)} />
              <span>Transparent</span></label>
          </>
        )}
      </div>
    </Modal>
  )
}

function PalettePicker({ palettes, value, onPick }: { palettes: Palette[]; value: string; onPick: (id: string) => void }) {
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const current = palettes.find((p) => p.id === value) ?? palettes[0]
  useEffect(() => {
    if (!at) return
    const down = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setAt(null) }
    const t = setTimeout(() => window.addEventListener('mousedown', down), 0)
    return () => { clearTimeout(t); window.removeEventListener('mousedown', down) }
  }, [at])
  const strip = (p: Palette) => <span className="pal-strip">{p.colors.slice(0, 8).map((c, i) => <i key={i} style={{ background: c }} />)}</span>
  return (
    <>
      <button className="btn" title="Color palette" onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setAt({ x: r.left, y: r.bottom + 4 }) }}>
        {current && strip(current)} <Icon name="down" size={12} />
      </button>
      {at && createPortal(
        <div ref={ref} className="menu palette-menu" style={{ left: Math.min(at.x, window.innerWidth - 300), top: at.y }}>
          {palettes.map((p) => (
            <button key={p.id} className={`menu-item ${p.id === value ? 'on' : ''}`} onClick={() => { onPick(p.id); setAt(null) }}>
              {strip(p)}<span className="menu-label">{p.label}</span>
            </button>
          ))}
        </div>, document.body)}
    </>
  )
}

export function GraphView({ item }: { item: GraphItem }) {
  const catalog = useStore((s) => s.catalog)
  const def = catalog?.graphs.find((g) => g.id === item.graphType)
  const sheet = useStore((s) => s.sheets.find((x) => x.id === item.sheetId))
  const hasClip = useStore((s) => !!s.styleClipboard)
  const { setOptions, patchItem, renameItem, deleteItem, duplicateItem, changeGraphType, copyStyle, pasteStyle, resetOptions, toast, select } = useStore.getState()
  const { rerun } = useGraphRunner(item, def, sheet)
  const [exporting, setExporting] = useState(false)
  const [zoom, setZoom] = useState<number | 'fit'>('fit')
  const stage = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 800, h: 600 })
  const [focus, setFocus] = useState<{ name: string; nonce: number } | null>(null)

  useLayoutEffect(() => {
    const el = stage.current
    if (!el) return
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    setBox({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  const related = useMemo(() => {
    if (!def || !catalog) return []
    const data = (g: GraphDef) => g.options.filter((o) => o.section === 'Data').map((o) => o.name).join(',')
    return catalog.graphs.filter((g) => data(g) === data(def))
  }, [def, catalog])

  if (!def || !catalog) return <div className="empty-state"><h3>Unknown graph type</h3></div>

  const wIn = Number(item.options.width ?? 5), hIn = Number(item.options.height ?? 4)
  const natW = wIn * 96, natH = hIn * 96
  const fit = Math.min((box.w - 64) / natW, (box.h - 64) / natH)
  const scale = zoom === 'fit' ? Math.max(0.2, Math.min(fit, 2.2)) : zoom
  const has = (name: string) => def.options.some((o) => o.name === name)
  const geometrySections = [...new Set(def.options.map((o) => o.section ?? ''))].filter((s) => !['Data', 'Titles & labels', 'Colors & theme', 'Axes', 'Legend', 'Fonts', 'Size', 'Significance', ''].includes(s))

  /**
   * Double-clicking a region of the graph jumps to the matching setting:
   * the top edge is the title, the edges next to the axes are the axis titles,
   * the right-hand margin is the legend and the middle is the data.
   */
  const jumpTo = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height
    const flip = !!item.options.flip
    const order = fy < 0.13 ? ['title']
      : fy > 0.87 ? [flip ? 'y_label' : 'x_label', 'caption']
      : fx < 0.12 ? [flip ? 'x_label' : 'y_label']
      : fx > 0.86 ? ['legend_position', 'legend_title', 'palette']
      : [def.options.find((o) => geometrySections.includes(o.section ?? ''))?.name ?? 'palette', 'palette', 'colormap']
    const name = order.find((n) => has(n)) ?? 'title'
    setFocus({ name, nonce: Date.now() })
  }

  const copyImage = async () => {
    if (!sheet) return
    try {
      const blob = await api.exportGraph(graphSpec(item), wireForItem(def.options, item.options, sheet), 'png', 300, false)
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
      toast('Graph copied to the clipboard (300 dpi PNG).')
    } catch (e: any) {
      toast('Could not copy the image — use Export instead.', 'error')
    }
  }

  return (
    <div className="split graph-split">
      <section className="content">
        <div className="content-head">
          <EditableText className="content-title" value={item.name} onCommit={(v) => renameItem(item.id, v)} />
          <div className="content-actions">
            {item.status === 'running' && <span className="status"><Spinner /> Drawing…</span>}
            <button className="btn" title="Copy the look of this graph" onClick={() => copyStyle(item.id)}><Icon name="wand" /> Copy style</button>
            {hasClip && <button className="btn" onClick={() => pasteStyle(item.id)}><Icon name="paste" /> Paste style</button>}
            <button className="btn primary" onClick={() => setExporting(true)} disabled={!sheet}><Icon name="download" /> Export</button>
            <MenuButton className="btn" items={[
              { label: 'Copy image to clipboard', icon: 'image', onClick: copyImage },
              { label: 'Redraw', icon: 'refresh', onClick: rerun },
              { label: 'Reset appearance to defaults', icon: 'refresh', onClick: () => resetOptions(item.id) },
              { divider: true },
              { label: 'Duplicate', icon: 'copy', onClick: () => duplicateItem(item.id) },
              { label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteItem(item.id) },
            ]}><Icon name="more" /></MenuButton>
          </div>
        </div>
        <div className="graph-toolbar">
          <div className="tool">
            <span className="tool-label">Graph</span>
            <select value={item.graphType} onChange={(e) => changeGraphType(item.id, e.target.value)}>
              {related.map((g) => <option key={g.id} value={g.id}>{g.name}</option>)}
            </select>
          </div>
          <div className="tool">
            <span className="tool-label">Engine</span>
            <div className="segmented">
              <button className={item.engine === 'matplotlib' ? 'on' : ''} onClick={() => patchItem(item.id, { engine: 'matplotlib' })} title="Python · matplotlib">Matplotlib</button>
              <button className={item.engine === 'ggplot2' ? 'on' : ''} disabled={!catalog.engines.r}
                title={catalog.engines.r ? 'R · ggplot2' : 'R is not installed'} onClick={() => patchItem(item.id, { engine: 'ggplot2' })}>ggplot2</button>
            </div>
          </div>
          <div className="tool">
            <span className="tool-label">Theme</span>
            <select value={item.options.theme} onChange={(e) => setOptions(item.id, { theme: e.target.value })}>
              {catalog.themes.map((t) => <option key={t.id} value={t.id}>{t.label}</option>)}
            </select>
          </div>
          {has('palette') && (
            <div className="tool">
              <span className="tool-label">Colors</span>
              <PalettePicker palettes={catalog.palettes} value={item.options.palette} onPick={(id) => setOptions(item.id, { palette: id, group_colors: {} })} />
            </div>
          )}
          {has('colormap') && (
            <div className="tool">
              <span className="tool-label">Colors</span>
              <PalettePicker palettes={catalog.colormaps} value={item.options.colormap} onPick={(id) => setOptions(item.id, { colormap: id })} />
            </div>
          )}
          <div className="tool grow" />
          <div className="tool">
            <div className="segmented">
              <button onClick={() => setZoom(Math.max(0.2, +(scale - 0.2).toFixed(2)))} title="Zoom out">−</button>
              <button className={zoom === 'fit' ? 'on' : ''} onClick={() => setZoom('fit')} title="Fit to window">{Math.round(scale * 100)}%</button>
              <button onClick={() => setZoom(Math.min(4, +(scale + 0.2).toFixed(2)))} title="Zoom in">+</button>
            </div>
          </div>
        </div>
        <div className="graph-stage" ref={stage}>
          {!sheet && <div className="empty-state"><h3>The data table for this graph was deleted.</h3></div>}
          {sheet && item.status === 'incomplete' && (
            <div className="empty-state">
              <Icon name="sliders" size={28} />
              <h3>Choose your data</h3>
              <p>Pick columns in the <b>Data</b> section on the right: <b>{item.message}</b></p>
            </div>
          )}
          {sheet && item.status === 'error' && (
            <div className="error-box">
              <Icon name="warn" size={18} />
              <div>
                <div className="error-title">This graph could not be drawn</div>
                {(item.message ?? '').split('\n').map((l, i) => <p key={i}>{l}</p>)}
              </div>
            </div>
          )}
          {sheet && item.svg && item.status !== 'incomplete' && item.status !== 'error' && (
            <div className={`graph-paper ${item.status === 'running' ? 'stale' : ''} ${item.options.transparent ? 'checker' : ''}`}
              style={{ width: natW * scale, height: natH * scale }} onDoubleClick={jumpTo}
              title="Double-click the title, an axis or the data to edit it">
              <img src={svgUrl(item.svg)} alt={item.name} draggable={false} style={{ width: '100%', height: '100%' }} />
            </div>
          )}
          {sheet && !item.svg && (item.status === 'idle' || item.status === 'running') && <div className="empty-state"><Spinner size={22} /><p>Drawing…</p></div>}
        </div>
        {item.meta?.warnings?.length ? (
          <div className="graph-warnings"><Icon name="warn" size={14} /> {item.meta.warnings.join(' ')}</div>
        ) : null}
      </section>
      <aside className="panel right">
        <div className="panel-head">
          <div>
            <div className="panel-title">Format graph</div>
            <div className="panel-sub">
              {def.name}{sheet && <> · <button className="link" onClick={() => select({ kind: 'sheet', id: sheet.id })}>{sheet.name}</button></>}
            </div>
          </div>
        </div>
        <OptionsForm schema={def.options} values={item.options} sheet={sheet} scope={`g:${def.render_type}`} searchable
          groups={item.meta?.groups} onChange={(patch) => setOptions(item.id, patch)} focus={focus}
          openSections={['Data', 'Titles & labels', ...geometrySections.slice(0, 1)]} />
      </aside>
      {exporting && sheet && <ExportDialog item={item} def={def} sheet={sheet} onClose={() => setExporting(false)} />}
    </div>
  )
}
