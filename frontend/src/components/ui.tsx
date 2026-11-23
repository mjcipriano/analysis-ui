import { ReactNode, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

const paths: Record<string, ReactNode> = {
  table: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M3 15h18M9 4v16" /></>,
  results: <><path d="M6 3h9l4 4v14H6z" /><path d="M14 3v5h5M9 13h7M9 17h5" /></>,
  graph: <><path d="M4 20V4M4 20h16" /><path d="M8 16v-5M12 16V8M16 16v-3" /></>,
  plus: <path d="M12 5v14M5 12h14" />,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  chevron: <path d="M9 6l6 6-6 6" />,
  down: <path d="M6 9l6 6 6-6" />,
  trash: <><path d="M4 7h16M10 11v6M14 11v6" /><path d="M6 7l1 13h10l1-13M9 7V4h6v3" /></>,
  copy: <><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" /></>,
  download: <><path d="M12 4v11M7 11l5 5 5-5" /><path d="M5 20h14" /></>,
  upload: <><path d="M12 16V5M7 9l5-5 5 5" /><path d="M5 20h14" /></>,
  folder: <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />,
  save: <><path d="M5 4h11l3 3v13H5z" /><path d="M8 4v5h7V4M8 20v-6h8v6" /></>,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5" /></>,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  search: <><circle cx="11" cy="11" r="6" /><path d="M20 20l-4.5-4.5" /></>,
  undo: <><path d="M9 7L4 12l5 5" /><path d="M4 12h10a5 5 0 0 1 0 10h-2" /></>,
  redo: <><path d="M15 7l5 5-5 5" /><path d="M20 12H10a5 5 0 0 0 0 10h2" /></>,
  wand: <><path d="M5 19L16 8M14 6l4 4" /><path d="M18 3v3M16.5 4.5h3M6 5v2M5 6h2M19 14v2M18 15h2" /></>,
  sliders: <><path d="M4 7h10M18 7h2M4 17h2M10 17h10" /><circle cx="16" cy="7" r="2" /><circle cx="8" cy="17" r="2" /></>,
  more: <><circle cx="5" cy="12" r="1.3" /><circle cx="12" cy="12" r="1.3" /><circle cx="19" cy="12" r="1.3" /></>,
  refresh: <><path d="M20 11a8 8 0 0 0-14.5-4M4 13a8 8 0 0 0 14.5 4" /><path d="M5 3v4h4M19 21v-4h-4" /></>,
  paste: <><rect x="6" y="5" width="12" height="16" rx="2" /><path d="M9 5V3h6v2M9 11h6M9 15h4" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8v.5" /></>,
  warn: <><path d="M12 4l9 16H3z" /><path d="M12 10v4M12 17v.5" /></>,
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  function: <><path d="M9 20c2 0 2.500-1 3-4l1-8c.5-3 1-4 3-4" /><path d="M8 11h7" /></>,
  sort: <><path d="M7 5v14M4 16l3 3 3-3M17 19V5M14 8l3-3 3 3" /></>,
  zoom: <><circle cx="11" cy="11" r="6" /><path d="M20 20l-4.5-4.5M11 8.500v5M8.500 11h5" /></>,
  fit: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
  sparkle: <path d="M12 3l1.800 5.200L19 10l-5.200 1.800L12 17l-1.800-5.200L5 10l5.200-1.800z" />,
  edit: <><path d="M4 20h4L19 9l-4-4L4 16z" /><path d="M13.500 6.500l4 4" /></>,
  image: <><rect x="3" y="5" width="18" height="14" rx="2" /><circle cx="8.500" cy="10" r="1.500" /><path d="M21 16l-5-5-8 8" /></>,
  layers: <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /></>,
}

export function Icon({ name, size = 16, className }: { name: string; size?: number; className?: string }) {
  return (
    <svg className={`icon ${className ?? ''}`} width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {paths[name] ?? null}
    </svg>
  )
}

export function Modal({ title, onClose, children, width = 720, footer }: {
  title: ReactNode; onClose: () => void; children: ReactNode; width?: number; footer?: ReactNode
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal" style={{ width }} role="dialog" aria-modal="true">
        <div className="modal-head">
          <div className="modal-title">{title}</div>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" /></button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>,
    document.body,
  )
}

export interface MenuEntry { label?: string; icon?: string; onClick?: () => void; danger?: boolean; divider?: boolean; disabled?: boolean; checked?: boolean; hint?: string }

/** A popup menu anchored at a point or below an element. */
export function Menu({ at, items, onClose }: { at: { x: number; y: number }; items: MenuEntry[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState(at)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setPos({ x: Math.max(8, Math.min(at.x, window.innerWidth - r.width - 8)), y: Math.max(8, Math.min(at.y, window.innerHeight - r.height - 8)) })
  }, [at.x, at.y])
  useEffect(() => {
    const down = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) onClose() }
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    const t = setTimeout(() => window.addEventListener('mousedown', down), 0)
    window.addEventListener('keydown', key)
    return () => { clearTimeout(t); window.removeEventListener('mousedown', down); window.removeEventListener('keydown', key) }
  }, [onClose])
  return createPortal(
    <div ref={ref} className="menu" style={{ left: pos.x, top: pos.y }} role="menu">
      {items.map((it, i) => it.divider
        ? <div key={i} className="menu-divider" />
        : (
          <button key={i} role="menuitem" className={`menu-item ${it.danger ? 'danger' : ''}`} disabled={it.disabled}
            onClick={() => { onClose(); it.onClick?.() }}>
            <span className="menu-icon">{it.checked ? <Icon name="check" size={14} /> : it.icon ? <Icon name={it.icon} size={14} /> : null}</span>
            <span className="menu-label">{it.label}</span>
            {it.hint && <span className="menu-hint">{it.hint}</span>}
          </button>
        ))}
    </div>,
    document.body,
  )
}

/** A button that opens a Menu underneath itself. */
export function MenuButton({ items, children, className, title }: { items: MenuEntry[]; children: ReactNode; className?: string; title?: string }) {
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)
  return (
    <>
      <button className={className ?? 'btn'} title={title}
        onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setAt({ x: r.left, y: r.bottom + 4 }) }}>
        {children}
      </button>
      {at && <Menu at={at} items={items} onClose={() => setAt(null)} />}
    </>
  )
}

/** Text that turns into an input on double click (or when `editing`). */
export function EditableText({ value, onCommit, className }: { value: string; onCommit: (v: string) => void; className?: string }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  if (!editing) {
    return <span className={`editable ${className ?? ''}`} title="Double-click to rename" onDoubleClick={() => setEditing(true)}>{value}</span>
  }
  const commit = () => { setEditing(false); if (draft.trim() && draft !== value) onCommit(draft.trim()); else setDraft(value) }
  return (
    <input className={`editable-input ${className ?? ''}`} autoFocus value={draft} onChange={(e) => setDraft(e.target.value)}
      onFocus={(e) => e.currentTarget.select()} onBlur={commit}
      onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') { setDraft(value); setEditing(false) } }} />
  )
}

export function Spinner({ size = 14 }: { size?: number }) {
  return <span className="spinner" style={{ width: size, height: size }} />
}

export function TypeGlyph({ type }: { type: string }) {
  const label = type === 'continuous' ? '123' : type === 'ordinal' ? '1‹2' : 'abc'
  return <span className={`type-glyph ${type}`} title={type === 'continuous' ? 'Numeric (continuous)' : type === 'ordinal' ? 'Ordered categories (ordinal)' : 'Categories (nominal)'}>{label}</span>
}
