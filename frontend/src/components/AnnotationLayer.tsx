import { useEffect, useRef, useState } from 'react'
import type { Annotation } from '../types'

export type AnnotationTool = null | 'text' | 'arrow'

interface Props {
  items: Annotation[]
  tool: AnnotationTool
  onChange: (next: Annotation[]) => void
  onToolDone: () => void
}

const clamp = (v: number) => Math.min(1, Math.max(0, v))

/**
 * A transparent layer over the rendered graph. The engines draw the
 * annotations; this layer only provides the handles to place, move, edit and
 * delete them. Positions are fractions of the figure from the top-left corner.
 */
export function AnnotationLayer({ items, tool, onChange, onToolDone }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [local, setLocal] = useState<Annotation[] | null>(null)     // live copy while dragging
  const [selected, setSelected] = useState<number | null>(null)
  const [editing, setEditing] = useState<number | null>(null)
  const [draft, setDraft] = useState<{ x: number; y: number; x2: number; y2: number } | null>(null)
  const list = local ?? items

  const frac = (e: { clientX: number; clientY: number }) => {
    const r = ref.current!.getBoundingClientRect()
    return { x: clamp((e.clientX - r.left) / r.width), y: clamp((e.clientY - r.top) / r.height) }
  }

  useEffect(() => {
    if (selected === null || editing !== null) return
    const key = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); onChange(items.filter((_, i) => i !== selected)); setSelected(null) }
      if (e.key === 'Escape') setSelected(null)
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [selected, editing, items, onChange])

  /** Drag helper: `apply` receives the pointer position (fractions) and the start position. */
  const startDrag = (e: React.MouseEvent, index: number, apply: (a: Annotation, p: { x: number; y: number }, d: { dx: number; dy: number }) => Annotation) => {
    if (tool) return
    e.preventDefault(); e.stopPropagation()
    setSelected(index)
    const start = frac(e)
    const original = items[index]
    let latest: Annotation[] | null = null
    const move = (ev: MouseEvent) => {
      const p = frac(ev)
      latest = items.map((a, i) => (i === index ? apply(original, p, { dx: p.x - start.x, dy: p.y - start.y }) : a))
      setLocal(latest)
    }
    const up = () => {
      window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up)
      if (latest) onChange(latest)
      setLocal(null)
    }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }

  const onLayerDown = (e: React.MouseEvent) => {
    if (tool === 'arrow') {
      e.preventDefault()
      const s = frac(e)
      setDraft({ x: s.x, y: s.y, x2: s.x, y2: s.y })
      const move = (ev: MouseEvent) => { const p = frac(ev); setDraft({ x: s.x, y: s.y, x2: p.x, y2: p.y }) }
      const up = (ev: MouseEvent) => {
        window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up)
        const p = frac(ev)
        setDraft(null)
        if (Math.hypot(p.x - s.x, p.y - s.y) > 0.02) {
          onChange([...items, { kind: 'arrow', x: s.x, y: s.y, x2: p.x, y2: p.y, width: 1.2, color: '' }])
          setSelected(items.length)
        }
        onToolDone()
      }
      window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
    } else if (tool === 'text') {
      e.preventDefault()
      const p = frac(e)
      onChange([...items, { kind: 'text', x: p.x, y: p.y, text: 'Text', size: null, color: '', bold: false, align: 'center' }])
      setSelected(items.length); setEditing(items.length)
      onToolDone()
    } else {
      setSelected(null)
    }
  }

  const pct = (v: number) => `${v * 100}%`
  return (
    <div ref={ref} className={`anno-layer ${tool ? 'placing' : ''}`} onMouseDown={onLayerDown}
      onDoubleClick={(e) => { if (tool || (e.target as HTMLElement).closest('.anno-handle')) e.stopPropagation() }}>
      <svg className="anno-svg" width="100%" height="100%">
        {list.map((a, i) => a.kind !== 'text' && (local || selected === i) ? (
          <line key={i} x1={pct(a.x)} y1={pct(a.y)} x2={pct(a.x2)} y2={pct(a.y2)} className="anno-guide" />
        ) : null)}
        {draft && <line x1={pct(draft.x)} y1={pct(draft.y)} x2={pct(draft.x2)} y2={pct(draft.y2)} className="anno-guide draft" />}
      </svg>
      {list.map((a, i) => {
        if (a.kind === 'text') {
          if (editing === i) {
            return (
              <input key={i} className="anno-edit" autoFocus defaultValue={a.text} style={{ left: pct(a.x), top: pct(a.y) }}
                onMouseDown={(e) => e.stopPropagation()} onFocus={(e) => e.currentTarget.select()}
                onBlur={(e) => {
                  const text = e.currentTarget.value.trim()
                  onChange(text ? items.map((x, j) => (j === i ? { ...x, text } as Annotation : x)) : items.filter((_, j) => j !== i))
                  setEditing(null)
                }}
                onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') setEditing(null) }} />
            )
          }
          const w = Math.max(28, a.text.length * 6.500 + 12)
          return (
            <div key={i} className={`anno-handle text ${selected === i ? 'selected' : ''} ${local ? 'dragging' : ''}`}
              style={{ left: pct(a.x), top: pct(a.y), width: w, marginLeft: a.align === 'left' ? 0 : a.align === 'right' ? -w : -w / 2 }}
              title="Drag to move · double-click to edit · Delete to remove"
              onMouseDown={(e) => startDrag(e, i, (o, _p, d) => ({ ...o, x: clamp(o.x + d.dx), y: clamp(o.y + d.dy) }))}
              onDoubleClick={(e) => { e.stopPropagation(); setEditing(i) }}>
              {local && <span className="anno-ghost">{a.text}</span>}
            </div>
          )
        }
        const mid = { x: (a.x + a.x2) / 2, y: (a.y + a.y2) / 2 }
        return (
          <div key={i}>
            <div className={`anno-handle dot ${selected === i ? 'selected' : ''}`} style={{ left: pct(a.x), top: pct(a.y) }} title="Drag the start"
              onMouseDown={(e) => startDrag(e, i, (o, p) => ({ ...o, x: p.x, y: p.y }))} />
            <div className={`anno-handle dot ${selected === i ? 'selected' : ''}`} style={{ left: pct(a.x2), top: pct(a.y2) }} title="Drag the tip"
              onMouseDown={(e) => startDrag(e, i, (o, p) => ({ ...o, x2: p.x, y2: p.y } as Annotation))} />
            <div className={`anno-handle dot mid ${selected === i ? 'selected' : ''}`} style={{ left: pct(mid.x), top: pct(mid.y) }} title="Drag to move · Delete to remove"
              onMouseDown={(e) => startDrag(e, i, (o, _p, d) => (o.kind === 'text' ? o : { ...o, x: clamp(o.x + d.dx), y: clamp(o.y + d.dy), x2: clamp(o.x2 + d.dx), y2: clamp(o.y2 + d.dy) }))} />
          </div>
        )
      })}
    </div>
  )
}
