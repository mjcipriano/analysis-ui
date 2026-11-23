import { useMemo, useState } from 'react'
import { useStore } from '../store'
import type { AnalysisDef, GraphDef, Sheet } from '../types'
import { Icon, Modal } from './ui'

function matches(q: string, ...texts: string[]) {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean)
  const hay = texts.join(' ').toLowerCase()
  return words.every((w) => hay.includes(w))
}

export function AnalysisPicker({ sheet, onClose }: { sheet: Sheet; onClose: () => void }) {
  const catalog = useStore((s) => s.catalog)!
  const addAnalysis = useStore((s) => s.addAnalysis)
  const [q, setQ] = useState('')
  const [cat, setCat] = useState<string | null>(null)
  const list = useMemo(() => catalog.analyses.filter((a) => (q ? matches(q, a.name, a.description, a.keywords, a.category) : !cat || a.category === cat)), [catalog, q, cat])
  const cats = catalog.analysis_categories.filter((c) => catalog.analyses.some((a) => a.category === c))
  const pick = (a: AnalysisDef) => { if (!a.available) return; addAnalysis(sheet.id, a.id); onClose() }
  return (
    <Modal title={<>Analyze <span className="muted">“{sheet.name}”</span></>} onClose={onClose} width={860}>
      <div className="picker">
        <div className="picker-side">
          <div className="picker-search">
            <Icon name="search" size={14} />
            <input autoFocus placeholder="Search analyses… (e.g. anova, t test)" value={q} onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && list.length) pick(list.find((a) => a.available) ?? list[0]) }} />
          </div>
          <button className={`picker-cat ${!cat && !q ? 'on' : ''}`} onClick={() => { setCat(null); setQ('') }}>All analyses <span>{catalog.analyses.length}</span></button>
          {cats.map((c) => (
            <button key={c} className={`picker-cat ${cat === c && !q ? 'on' : ''}`} onClick={() => { setCat(c); setQ('') }}>
              {c} <span>{catalog.analyses.filter((a) => a.category === c).length}</span>
            </button>
          ))}
        </div>
        <div className="picker-list">
          {(q || !cat ? cats : [cat]).map((c) => {
            const items = list.filter((a) => a.category === c)
            if (!items.length) return null
            return (
              <div key={c}>
                <div className="picker-group">{c}</div>
                {items.map((a) => (
                  <button key={a.id} className={`picker-item ${a.available ? '' : 'disabled'}`} onClick={() => pick(a)}
                    title={a.available ? '' : a.unavailable_reason ?? ''}>
                    <div className="picker-item-main">
                      <div className="picker-item-name">{a.name}</div>
                      <div className="picker-item-desc">{a.available ? a.description : `Unavailable — ${a.unavailable_reason}`}</div>
                    </div>
                    <span className={`engine-badge ${a.engine}`}>{a.engine === 'r' ? 'R' : 'Python'}</span>
                  </button>
                ))}
              </div>
            )
          })}
          {!list.length && <div className="muted pad">No analysis matches “{q}”.</div>}
        </div>
      </div>
    </Modal>
  )
}

/** Small schematic drawings used when no rendered thumbnail is available. */
function GraphIcon({ type }: { type: string }) {
  const c1 = '#4F6FE8', c2 = '#F0683C', c3 = '#17A589'
  const axes = <path d="M10 6v40h46" fill="none" stroke="currentColor" strokeWidth="1.500" opacity=".45" />
  const body: Record<string, JSX.Element> = {
    column_scatter: <>{axes}{[[20, 30], [18, 24], [23, 27], [21, 34]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.200" fill={c1} />)}{[[38, 16], [41, 20], [36, 12], [40, 24]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.200" fill={c2} />)}<path d="M15 29h12M33 18h12" stroke="currentColor" strokeWidth="1.600" /></>,
    column_bar: <>{axes}<rect x="15" y="24" width="10" height="22" fill={c1} /><rect x="30" y="14" width="10" height="32" fill={c2} /><rect x="45" y="30" width="8" height="16" fill={c3} /><path d="M20 24v-6M17 18h6M35 14V9M32 9h6" stroke="currentColor" strokeWidth="1.300" /></>,
    box: <>{axes}<rect x="15" y="20" width="11" height="16" fill={c1} opacity=".7" /><path d="M15 27h11M20.500 20v-7M20.500 36v6" stroke="currentColor" strokeWidth="1.300" /><rect x="35" y="12" width="11" height="18" fill={c2} opacity=".7" /><path d="M35 20h11M40.500 12V8M40.500 30v8" stroke="currentColor" strokeWidth="1.300" /></>,
    violin: <>{axes}<path d="M21 10c-7 10-7 22 0 34c7-12 7-24 0-34z" fill={c1} opacity=".75" /><path d="M41 8c-5 6-9 20 0 32c9-12 5-26 0-32z" fill={c2} opacity=".75" /></>,
    raincloud: <>{axes}<path d="M20 10c8 8 8 24 0 32z" fill={c1} opacity=".75" />{[14, 16, 13, 17, 15].map((x, i) => <circle key={i} cx={x} cy={16 + i * 5} r="1.600" fill={c1} />)}<path d="M40 8c8 8 8 22 0 30z" fill={c2} opacity=".75" />{[34, 36, 33, 37].map((x, i) => <circle key={i} cx={x} cy={13 + i * 6} r="1.600" fill={c2} />)}</>,
    before_after: <>{axes}{[[30, 16], [24, 20], [36, 30], [28, 12]].map(([a, b], i) => <g key={i}><path d={`M20 ${a}L44 ${b}`} stroke="currentColor" opacity=".5" strokeWidth="1.200" /><circle cx="20" cy={a} r="2.200" fill={c1} /><circle cx="44" cy={b} r="2.200" fill={c2} /></g>)}</>,
    grouped_bar: <>{axes}<rect x="14" y="26" width="7" height="20" fill={c1} /><rect x="21" y="18" width="7" height="28" fill={c2} /><rect x="36" y="30" width="7" height="16" fill={c1} /><rect x="43" y="12" width="7" height="34" fill={c2} /></>,
    grouped_box: <>{axes}<rect x="14" y="22" width="7" height="12" fill={c1} opacity=".75" /><rect x="22" y="16" width="7" height="14" fill={c2} opacity=".75" /><rect x="37" y="26" width="7" height="12" fill={c1} opacity=".75" /><rect x="45" y="10" width="7" height="16" fill={c2} opacity=".75" /></>,
    grouped_violin: <>{axes}<path d="M18 14c-4 8-4 16 0 24c4-8 4-16 0-24z" fill={c1} opacity=".75" /><path d="M27 10c-4 8-4 16 0 24c4-8 4-16 0-24z" fill={c2} opacity=".75" /><path d="M41 18c-4 8-4 16 0 24c4-8 4-16 0-24z" fill={c1} opacity=".75" /><path d="M50 8c-4 8-4 16 0 24c4-8 4-16 0-24z" fill={c2} opacity=".75" /></>,
    grouped_points: <>{axes}{[[16, 30], [18, 26], [17, 34]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2" fill={c1} />)}{[[25, 20], [27, 16], [26, 24]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2" fill={c2} />)}{[[40, 32], [42, 28], [41, 36]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2" fill={c1} />)}{[[49, 14], [51, 10], [50, 18]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2" fill={c2} />)}</>,
    interaction_line: <>{axes}<path d="M16 34L33 28L50 26" stroke={c1} strokeWidth="2" fill="none" /><path d="M16 30L33 20L50 10" stroke={c2} strokeWidth="2" fill="none" />{[[16, 34], [33, 28], [50, 26]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.400" fill={c1} />)}{[[16, 30], [33, 20], [50, 10]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.400" fill={c2} />)}</>,
    xy: <>{axes}{[[16, 38], [22, 30], [27, 33], [32, 24], [38, 22], [43, 15], [50, 12], [36, 30], [46, 20]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.200" fill={c1} />)}</>,
    xy_line: <>{axes}<path d="M15 36L25 28L35 30L45 18L53 14" stroke={c1} strokeWidth="2" fill="none" /><path d="M15 40L25 38L35 32L45 30L53 24" stroke={c2} strokeWidth="2" fill="none" /></>,
    xy_regression: <>{axes}<path d="M13 40L54 10" stroke={c2} strokeWidth="2" />{[[16, 36], [22, 34], [27, 28], [32, 28], [38, 20], [43, 20], [50, 12]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.200" fill={c1} />)}</>,
    dose_response: <>{axes}<path d="M13 40c14 0 14-28 40-28" stroke={c1} strokeWidth="2" fill="none" />{[[15, 39], [23, 37], [30, 28], [37, 18], [45, 13], [52, 12]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.200" fill={c1} />)}</>,
    bubble: <>{axes}<circle cx="20" cy="32" r="4" fill={c1} opacity=".7" /><circle cx="32" cy="22" r="7" fill={c2} opacity=".7" /><circle cx="46" cy="14" r="5" fill={c3} opacity=".7" /><circle cx="44" cy="34" r="3" fill={c1} opacity=".7" /></>,
    histogram: <>{axes}{[10, 18, 28, 34, 24, 14, 6].map((h, i) => <rect key={i} x={13 + i * 6} y={46 - h} width="5.500" height={h} fill={c1} />)}</>,
    density: <>{axes}<path d="M12 46c8 0 8-30 18-30s8 30 18 30z" fill={c1} opacity=".5" /><path d="M22 46c8 0 8-24 18-24s8 24 16 24z" fill={c2} opacity=".5" /></>,
    ridgeline: <><path d="M10 22c8 0 8-14 18-14s8 14 26 14z" fill={c1} opacity=".75" /><path d="M10 34c12 0 12-14 22-14s8 14 22 14z" fill={c2} opacity=".75" /><path d="M10 46c16 0 16-14 26-14s8 14 18 14z" fill={c3} opacity=".75" /></>,
    qq: <>{axes}<path d="M13 42L54 10" stroke="currentColor" opacity=".5" strokeWidth="1.200" />{[[15, 42], [20, 37], [25, 33], [30, 29], [35, 24], [40, 21], [45, 16], [51, 9]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2" fill={c1} />)}</>,
    ecdf: <>{axes}<path d="M12 44h6v-6h6v-8h6v-8h8v-6h8v-4h8" stroke={c1} strokeWidth="2" fill="none" /></>,
    survival: <>{axes}<path d="M11 10h10v8h8v6h10v8h14" stroke={c1} strokeWidth="2" fill="none" /><path d="M11 10h6v12h8v10h8v8h20" stroke={c2} strokeWidth="2" fill="none" /></>,
    pie: <><circle cx="32" cy="27" r="19" fill={c1} /><path d="M32 27V8a19 19 0 0 1 17.500 26.500z" fill={c2} /><path d="M32 27l17.500 7.500A19 19 0 0 1 24 44.200z" fill={c3} /></>,
    donut: <><circle cx="32" cy="27" r="15" fill="none" stroke={c1} strokeWidth="8" /><path d="M32 12a15 15 0 0 1 13.800 20.900" fill="none" stroke={c2} strokeWidth="8" /></>,
    count_bar: <>{axes}<rect x="15" y="30" width="10" height="16" fill={c1} /><rect x="15" y="18" width="10" height="12" fill={c2} /><rect x="32" y="24" width="10" height="22" fill={c1} /><rect x="32" y="12" width="10" height="12" fill={c2} /></>,
    heatmap: <>{[0, 1, 2, 3].map((r) => [0, 1, 2, 3, 4].map((c) => <rect key={`${r}${c}`} x={10 + c * 9.500} y={8 + r * 9.500} width="8.500" height="8.500" fill={['#2166AC', '#92C5DE', '#F7F7F7', '#F4A582', '#B2182B'][(r * 3 + c * 2) % 5]} />))}</>,
    corr_matrix: <>{[0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => c <= r ? <rect key={`${r}${c}`} x={13 + c * 10} y={8 + r * 9.500} width="9" height="8.500" fill={r === c ? '#B2182B' : ['#92C5DE', '#F4A582', '#D6604D'][(r + c) % 3]} /> : null))}</>,
    bland_altman: <>{axes}<path d="M11 26h44" stroke={c1} strokeWidth="1.500" /><path d="M11 12h44M11 40h44" stroke="currentColor" strokeDasharray="3 2" opacity=".6" />{[[16, 22], [22, 30], [28, 24], [33, 34], [38, 18], [44, 28], [50, 22]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2" fill={c1} />)}</>,
    roc: <>{axes}<path d="M10 46L54 8" stroke="currentColor" opacity=".4" strokeDasharray="3 2" /><path d="M10 46c2-26 16-36 44-38" stroke={c1} strokeWidth="2" fill="none" /></>,
    volcano: <>{axes}{[[30, 42], [33, 40], [28, 38], [35, 43], [32, 36], [26, 41], [38, 39]].map(([x, y], i) => <circle key={i} cx={x} cy={y} r="1.700" fill="#B8BDC4" />)}{[[18, 22], [15, 14], [21, 28], [13, 26]].map(([x, y], i) => <circle key={`d${i}`} cx={x} cy={y} r="2" fill="#2C6FBB" />)}{[[46, 20], [50, 12], [44, 28], [52, 24]].map(([x, y], i) => <circle key={`u${i}`} cx={x} cy={y} r="2" fill="#D7402B" />)}</>,
    forest: <><path d="M34 6v42" stroke="currentColor" opacity=".45" strokeDasharray="3 2" />{[[20, 36, 12], [16, 30, 21], [26, 44, 30], [22, 32, 39]].map(([a, b, y], i) => <g key={i}><path d={`M${a} ${y}H${b}`} stroke={c1} strokeWidth="1.600" /><rect x={(a + b) / 2 - 2.500} y={y - 2.500} width="5" height="5" fill={c1} /></g>)}</>,
    waterfall: <><path d="M8 26h50" stroke="currentColor" opacity=".45" />{[16, 12, 8, 4, -3, -7, -12, -16, -19].map((h, i) => <rect key={i} x={10 + i * 5} y={h > 0 ? 26 - h : 26} width="4" height={Math.abs(h)} fill={h > 0 ? c2 : c1} />)}</>,
  }
  return <svg viewBox="0 0 64 54" className="graph-icon">{body[type] ?? body.xy}</svg>
}

export function GraphThumb({ def }: { def: GraphDef }) {
  const [failed, setFailed] = useState(false)
  if (failed) return <GraphIcon type={def.id in ICON_IDS ? def.id : def.render_type} />
  return <img className="graph-thumb-img" src={`/thumbs/${def.id}.png`} alt="" loading="lazy" onError={() => setFailed(true)} />
}
const ICON_IDS: Record<string, true> = { xy_line: true, xy_regression: true, dose_response: true, bubble: true, donut: true, ridgeline: true, corr_matrix: true }

export function GraphGallery({ sheet, onClose }: { sheet: Sheet; onClose: () => void }) {
  const catalog = useStore((s) => s.catalog)!
  const addGraph = useStore((s) => s.addGraph)
  const engine = useStore((s) => s.defaultEngine)
  const [q, setQ] = useState('')
  const list = catalog.graphs.filter((g) => !q || matches(q, g.name, g.description, g.category))
  return (
    <Modal title={<>New graph of <span className="muted">“{sheet.name}”</span></>} onClose={onClose} width={920}>
      <div className="gallery-top">
        <div className="picker-search">
          <Icon name="search" size={14} />
          <input autoFocus placeholder="Search graph types…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <div className="tool">
          <span className="tool-label">Draw with</span>
          <div className="segmented">
            <button className={engine === 'matplotlib' ? 'on' : ''} onClick={() => useStore.setState({ defaultEngine: 'matplotlib' })}>Matplotlib (Python)</button>
            <button className={engine === 'ggplot2' ? 'on' : ''} disabled={!catalog.engines.r} onClick={() => useStore.setState({ defaultEngine: 'ggplot2' })}>ggplot2 (R)</button>
          </div>
        </div>
      </div>
      <div className="gallery">
        {catalog.graph_categories.map((c) => {
          const items = list.filter((g) => g.category === c)
          if (!items.length) return null
          return (
            <div key={c} className="gallery-group">
              <div className="picker-group">{c}</div>
              <div className="gallery-grid">
                {items.map((g) => (
                  <button key={g.id} className="gallery-card" title={g.description} onClick={() => { addGraph(sheet.id, g.id); onClose() }}>
                    <div className="gallery-thumb"><GraphThumb def={g} /></div>
                    <div className="gallery-name">{g.name}</div>
                  </button>
                ))}
              </div>
            </div>
          )
        })}
        {!list.length && <div className="muted pad">No graph type matches “{q}”.</div>}
      </div>
    </Modal>
  )
}
