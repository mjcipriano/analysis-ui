import { useEffect, useMemo, useRef, useState } from 'react'
import { isVisible, levelsOf } from '../data'
import type { Annotation, Column, OptionDef, Sheet } from '../types'
import { Icon, TypeGlyph } from './ui'

interface Props {
  schema: OptionDef[]
  values: Record<string, any>
  sheet?: Sheet
  onChange: (patch: Record<string, any>) => void
  /** resolved colours of the groups in the current graph (for the per-group colour editor) */
  groups?: { name: string; color: string }[]
  /** identifies the kind of item so collapsed sections are remembered per kind */
  scope: string
  searchable?: boolean
  /** sections opened by default (others start collapsed) */
  openSections?: string[]
  /** reveal and focus one option (e.g. after double-clicking the graph) */
  focus?: { name: string; nonce: number } | null
}

const openState = new Map<string, boolean>()

export function OptionsForm({ schema, values, sheet, onChange, groups, scope, searchable, openSections, focus }: Props) {
  const [query, setQuery] = useState('')
  const [, force] = useState(0)
  const root = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!focus) return
    const def = schema.find((o) => o.name === focus.name)
    if (!def) return
    setQuery('')
    if (def.section) openState.set(`${scope}:${def.section}`, true)
    force((n) => n + 1)
    const t = window.setTimeout(() => {
      const el = root.current?.querySelector<HTMLElement>(`[data-opt="${focus.name}"]`)
      if (!el) return
      el.scrollIntoView({ block: 'center', behavior: 'smooth' })
      el.classList.remove('flash')
      void el.offsetWidth
      el.classList.add('flash')
      const input = el.querySelector<HTMLElement>('input:not([type=range]):not([type=color]), select, textarea')
      input?.focus({ preventScroll: true })
      if (input instanceof HTMLInputElement && input.type === 'text') input.select()
    }, 60)
    return () => window.clearTimeout(t)
  }, [focus?.nonce])   // eslint-disable-line react-hooks/exhaustive-deps
  const visible = useMemo(() => schema.filter((o) => isVisible(o, values)), [schema, values])
  const q = query.trim().toLowerCase()
  const shown = q ? visible.filter((o) => `${o.label} ${o.section ?? ''} ${o.choices?.map((c) => c.label).join(' ') ?? ''}`.toLowerCase().includes(q)) : visible

  const sections: { name: string; opts: OptionDef[] }[] = []
  for (const o of shown) {
    const name = o.section ?? ''
    let s = sections.find((x) => x.name === name)
    if (!s) { s = { name, opts: [] }; sections.push(s) }
    s.opts.push(o)
  }

  const isOpen = (name: string, index: number) => {
    if (q) return true
    const key = `${scope}:${name}`
    if (openState.has(key)) return openState.get(key)!
    if (openSections) return openSections.includes(name)
    return index < 3
  }
  const toggle = (name: string, index: number) => { openState.set(`${scope}:${name}`, !isOpen(name, index)); force((n) => n + 1) }

  return (
    <div className="options" ref={root}>
      {searchable && (
        <div className="options-search">
          <Icon name="search" size={14} />
          <input placeholder="Search options…" value={query} onChange={(e) => setQuery(e.target.value)} />
          {query && <button className="icon-btn small" onClick={() => setQuery('')}><Icon name="x" size={12} /></button>}
        </div>
      )}
      {sections.map((s, i) => (
        <div className={`opt-section ${s.name ? '' : 'plain'}`} key={s.name || '_'}>
          {s.name && (
            <button className="opt-section-head" onClick={() => toggle(s.name, i)} aria-expanded={isOpen(s.name, i)}>
              <Icon name="chevron" size={13} className={isOpen(s.name, i) ? 'rot90' : ''} />
              <span>{s.name}</span>
            </button>
          )}
          {(!s.name || isOpen(s.name, i)) && (
            <div className="opt-section-body">
              {s.opts.map((o) => (
                <Field key={o.name} def={o} value={values[o.name]} values={values} schema={schema} sheet={sheet}
                  groups={groups} onChange={(v) => onChange({ [o.name]: v })} />
              ))}
            </div>
          )}
        </div>
      ))}
      {q && !shown.length && <div className="muted pad">No options match “{query}”.</div>}
    </div>
  )
}

interface FieldProps {
  def: OptionDef
  value: any
  values: Record<string, any>
  schema: OptionDef[]
  sheet?: Sheet
  groups?: { name: string; color: string }[]
  onChange: (v: any) => void
}

function Field(p: FieldProps) {
  const { def } = p
  if (def.type === 'bool') {
    return (
      <label className="field check" title={def.help} data-opt={def.name}>
        <input type="checkbox" checked={!!p.value} onChange={(e) => p.onChange(e.target.checked)} />
        <span>{def.label}</span>
      </label>
    )
  }
  return (
    <div className={`field ${def.type}`} data-opt={def.name}>
      <div className="field-label" title={def.help}>{def.label}</div>
      <div className="field-control">
        {def.type === 'variable' && <VarPicker {...p} />}
        {def.type === 'variables' && <VarsPicker {...p} />}
        {def.type === 'choice' && <ChoiceField {...p} />}
        {(def.type === 'number' || def.type === 'integer') && <NumberField {...p} />}
        {def.type === 'text' && <TextField {...p} />}
        {def.type === 'level' && <LevelField {...p} />}
        {def.type === 'color' && <ColorField value={p.value} onChange={p.onChange} />}
        {def.type === 'group_colors' && <GroupColors {...p} />}
        {def.type === 'annotations' && <AnnotationList value={p.value} onChange={p.onChange} />}
      </div>
      {def.help && def.type !== 'group_colors' && <div className="field-help">{def.help}</div>}
    </div>
  )
}

function permittedColumns(def: OptionDef, sheet?: Sheet): Column[] {
  if (!sheet) return []
  const ok = def.permitted ?? ['continuous']
  return sheet.columns.filter((c) => ok.includes(c.type))
}

function VarPicker({ def, value, sheet, onChange }: FieldProps) {
  const cols = permittedColumns(def, sheet)
  const missing = value && !sheet?.columns.some((c) => c.name === value)
  const wrongType = value && !missing && !cols.some((c) => c.name === value)
  const current = sheet?.columns.find((c) => c.name === value)
  return (
    <div className={`var-picker ${!value && def.required ? 'needs' : ''}`}>
      {current && <TypeGlyph type={current.type} />}
      <select value={value ?? ''} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">{def.required ? 'Choose a column…' : '(none)'}</option>
        {missing && <option value={value}>{value} (missing)</option>}
        {wrongType && current && <option value={value}>{value}</option>}
        {cols.map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
      </select>
    </div>
  )
}

function VarsPicker({ def, value, sheet, onChange }: FieldProps) {
  const selected: string[] = Array.isArray(value) ? value : []
  const cols = permittedColumns(def, sheet)
  const available = cols.filter((c) => !selected.includes(c.name))
  const atMax = def.max != null && selected.length >= def.max
  const drag = useRef<number | null>(null)
  const typeOf = (n: string) => sheet?.columns.find((c) => c.name === n)?.type
  return (
    <div className={`vars-picker ${selected.length < (def.min ?? 0) ? 'needs' : ''}`}>
      <div className="chips">
        {selected.map((n, i) => (
          <span key={n} className={`chip ${typeOf(n) ? '' : 'missing'}`} draggable
            onDragStart={() => { drag.current = i }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={() => {
              const from = drag.current
              if (from === null || from === i) return
              const next = selected.slice()
              const [m] = next.splice(from, 1)
              next.splice(i, 0, m)
              drag.current = null
              onChange(next)
            }}
            title="Drag to reorder">
            {typeOf(n) && <TypeGlyph type={typeOf(n)!} />}
            <span className="chip-label">{n}</span>
            <button onClick={() => onChange(selected.filter((x) => x !== n))} aria-label={`Remove ${n}`}><Icon name="x" size={11} /></button>
          </span>
        ))}
        {!selected.length && <span className="muted small">No columns chosen</span>}
      </div>
      <div className="vars-actions">
        <select value="" disabled={!available.length || atMax}
          onChange={(e) => { if (e.target.value) onChange([...selected, e.target.value]) }}>
          <option value="">{atMax ? 'Maximum reached' : available.length ? 'Add column…' : 'No more columns'}</option>
          {available.map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
        </select>
        {available.length > 1 && !atMax && def.max == null && (
          <button className="link" onClick={() => onChange([...selected, ...available.map((c) => c.name)])}>All</button>
        )}
        {selected.length > 0 && <button className="link" onClick={() => onChange([])}>Clear</button>}
      </div>
    </div>
  )
}

function ChoiceField({ def, value, onChange }: FieldProps) {
  const choices = def.choices ?? []
  const short = choices.length <= 3 && choices.every((c) => c.label.length <= 14) && choices.reduce((n, c) => n + c.label.length, 0) <= 30
  if (short) {
    return (
      <div className="segmented">
        {choices.map((c) => (
          <button key={c.value} className={value === c.value ? 'on' : ''} onClick={() => onChange(c.value)}>{c.label}</button>
        ))}
      </div>
    )
  }
  return (
    <select value={value ?? ''} onChange={(e) => onChange(e.target.value)}>
      {choices.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
    </select>
  )
}

function NumberField({ def, value, onChange }: FieldProps) {
  const [draft, setDraft] = useState(value === null || value === undefined ? '' : String(value))
  const last = useRef(value)
  useEffect(() => {
    if (value !== last.current) { last.current = value; setDraft(value === null || value === undefined ? '' : String(value)) }
  }, [value])
  const commit = (text: string) => {
    setDraft(text)
    const t = text.trim()
    if (t === '') { if (def.nullable) { last.current = null; onChange(null) } return }
    let n = Number(t)
    if (!Number.isFinite(n)) return
    if (def.type === 'integer') n = Math.round(n)
    if (def.min != null && n < def.min) return
    if (def.max != null && n > def.max) return
    last.current = n
    onChange(n)
  }
  const slider = def.min != null && def.max != null && def.step != null && (def.max - def.min) / def.step <= 400 && !def.nullable
  return (
    <div className="number-field">
      {slider && (
        <input type="range" min={def.min!} max={def.max!} step={def.step!} value={Number(value ?? def.default ?? def.min)}
          onChange={(e) => commit(e.target.value)} />
      )}
      <input type="number" value={draft} min={def.min ?? undefined} max={def.max ?? undefined} step={def.step ?? 'any'}
        placeholder={def.placeholder ?? (def.nullable ? 'auto' : '')}
        onChange={(e) => commit(e.target.value)}
        onBlur={() => setDraft(value === null || value === undefined ? '' : String(value))} />
    </div>
  )
}

function TextField({ def, value, onChange }: FieldProps) {
  const [draft, setDraft] = useState(value ?? '')
  const last = useRef(value ?? '')
  const timer = useRef<number>()
  useEffect(() => { if ((value ?? '') !== last.current) { last.current = value ?? ''; setDraft(value ?? '') } }, [value])
  const push = (v: string, now = false) => {
    setDraft(v)
    window.clearTimeout(timer.current)
    const go = () => { last.current = v; onChange(v) }
    if (now) go(); else timer.current = window.setTimeout(go, 350)
  }
  if (def.multiline) {
    return <textarea rows={3} value={draft} placeholder={def.placeholder} onChange={(e) => push(e.target.value)} onBlur={(e) => push(e.target.value, true)} />
  }
  return <input type="text" value={draft} placeholder={def.placeholder} onChange={(e) => push(e.target.value)} onBlur={(e) => push(e.target.value, true)}
    onKeyDown={(e) => { if (e.key === 'Enter') push((e.target as HTMLInputElement).value, true) }} />
}

function LevelField({ def, value, values, schema, sheet, onChange }: FieldProps) {
  const levels = useMemo(() => {
    for (const src of def.sources ?? []) {
      const sdef = schema.find((o) => o.name === src)
      if (!sdef || !isVisible(sdef, values)) continue
      const v = values[src]
      if (sdef.type === 'variables' && Array.isArray(v) && v.length) return v as string[]
      if (sdef.type === 'variable' && v) {
        const col = sheet?.columns.find((c) => c.name === v)
        if (col) return levelsOf(col)
      }
    }
    return []
  }, [def, schema, values, sheet])
  return (
    <select value={levels.includes(value) ? value : ''} onChange={(e) => onChange(e.target.value)}>
      <option value="">{levels.length ? `${levels[0]} (first)` : '(first)'}</option>
      {levels.map((l) => <option key={l} value={l}>{l}</option>)}
    </select>
  )
}

function ColorField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="color-field">
      <input type="color" value={/^#[0-9a-f]{6}$/i.test(value) ? value : '#ffffff'} onChange={(e) => onChange(e.target.value)} />
      <span className="muted small">{value || 'automatic'}</span>
      {value && <button className="link" onClick={() => onChange('')}>Reset</button>}
    </div>
  )
}

function GroupColors({ value, groups, onChange }: FieldProps) {
  const custom: Record<string, string> = value && typeof value === 'object' ? value : {}
  if (!groups?.length) return <div className="muted small">Colors appear here once the graph is drawn.</div>
  const hasCustom = Object.keys(custom).length > 0
  return (
    <div className="group-colors">
      {groups.slice(0, 40).map((g) => (
        <label key={g.name} className="group-color" title={`Change the color of ${g.name}`}>
          <input type="color" value={/^#[0-9a-f]{6}$/i.test(g.color) ? g.color : '#888888'}
            onChange={(e) => onChange({ ...custom, [g.name]: e.target.value })} />
          <span className="swatch" style={{ background: g.color }} />
          <span className="group-name">{g.name}</span>
          {custom[g.name] && (
            <button className="link" onClick={(e) => { e.preventDefault(); const next = { ...custom }; delete next[g.name]; onChange(next) }}>reset</button>
          )}
        </label>
      ))}
      {hasCustom && <button className="link" onClick={() => onChange({})}>Reset all colors</button>}
    </div>
  )
}

function AnnotationList({ value, onChange }: { value: any; onChange: (v: Annotation[]) => void }) {
  const items: Annotation[] = Array.isArray(value) ? value : []
  const patch = (i: number, p: Record<string, any>) => onChange(items.map((a, j) => (j === i ? ({ ...a, ...p } as Annotation) : a)))
  if (!items.length) return <div className="muted small">None yet.</div>
  return (
    <div className="anno-list">
      {items.map((a, i) => (
        <div key={i} className="anno-item">
          {a.kind === 'text' ? (
            <>
              <input type="text" value={a.text} onChange={(e) => patch(i, { text: e.target.value })} />
              <div className="anno-item-row">
                <input type="number" title="Size (pt)" placeholder="size" min={4} max={72} step={0.5} value={a.size ?? ''}
                  onChange={(e) => patch(i, { size: e.target.value === '' ? null : Number(e.target.value) })} />
                <button className={`btn mini ${a.bold ? 'on' : ''}`} title="Bold" onClick={() => patch(i, { bold: !a.bold })}><b>B</b></button>
                <button className={`btn mini ${a.italic ? 'on' : ''}`} title="Italic" onClick={() => patch(i, { italic: !a.italic })}><i>I</i></button>
                <select title="Alignment" value={a.align ?? 'center'} onChange={(e) => patch(i, { align: e.target.value })}>
                  <option value="left">Left</option><option value="center">Center</option><option value="right">Right</option>
                </select>
                <input type="number" title="Rotation (degrees)" placeholder="0°" step={15} value={a.angle || ''} onChange={(e) => patch(i, { angle: Number(e.target.value) || 0 })} />
              </div>
            </>
          ) : (
            <div className="anno-item-row">
              <select value={a.kind} onChange={(e) => patch(i, { kind: e.target.value })}><option value="arrow">Arrow</option><option value="line">Line</option></select>
              <input type="number" title="Thickness (pt)" min={0.25} max={12} step={0.25} value={a.width ?? 1.2} onChange={(e) => patch(i, { width: Number(e.target.value) || 1.2 })} />
            </div>
          )}
          <div className="anno-item-row">
            <input type="color" value={/^#[0-9a-f]{6}$/i.test(a.color ?? '') ? a.color : '#111111'} onChange={(e) => patch(i, { color: e.target.value })} />
            <span className="muted small">{a.color || 'theme color'}</span>
            <span style={{ flex: 1 }} />
            <button className="link" onClick={() => onChange(items.filter((_, j) => j !== i))}>Remove</button>
          </div>
        </div>
      ))}
    </div>
  )
}
