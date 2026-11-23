/** Number formatting for result tables (the backend sends raw numbers). */

export function fmtNum(v: unknown): string {
  if (v === null || v === undefined || v === '') return ''
  if (typeof v === 'string') return v
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  const x = Number(v)
  if (!Number.isFinite(x)) return ''
  if (x === 0) return '0'
  const a = Math.abs(x)
  if (Number.isInteger(x) && a < 1e9) return x.toLocaleString('en-US')
  if (a >= 1e7 || a < 1e-4) return x.toExponential(3).replace(/e([+-])(\d)$/, 'e$10$2').replace('e+', 'e')
  if (a >= 1000) return x.toLocaleString('en-US', { maximumFractionDigits: 1 })
  if (a >= 100) return x.toFixed(1)
  if (a >= 10) return x.toFixed(2)
  if (a >= 1) return x.toFixed(3)
  return x.toPrecision(4).replace(/0+$/, '').replace(/\.$/, '')
}

export function fmtP(v: unknown): string {
  if (v === null || v === undefined || v === '') return ''
  if (typeof v === 'string') return v
  const p = Number(v)
  if (!Number.isFinite(p)) return ''
  if (p < 0.0001) return '<0.0001'
  if (p > 0.9999) return '>0.9999'
  return p.toFixed(4)
}

export function stars(v: unknown): string {
  if (typeof v === 'string') return v
  const p = Number(v)
  if (v === null || v === undefined || !Number.isFinite(p)) return ''
  return p < 0.0001 ? '****' : p < 0.001 ? '***' : p < 0.01 ? '**' : p < 0.05 ? '*' : 'ns'
}

export function fmtCell(v: unknown, format: string): string {
  switch (format) {
    case 'p': return fmtP(v)
    case 'stars': return stars(v)
    case 'int': {
      if (v === null || v === undefined || v === '') return ''
      if (typeof v === 'string') return v
      const x = Number(v)
      return Number.isFinite(x) ? Math.round(x).toLocaleString('en-US') : ''
    }
    case 'pct': {
      if (v === null || v === undefined || v === '') return ''
      if (typeof v === 'string') return v
      const x = Number(v)
      return Number.isFinite(x) ? `${(x * 100).toFixed(Math.abs(x) < 0.1 ? 2 : 1)}%` : ''
    }
    case 'text': return v === null || v === undefined ? '' : typeof v === 'number' ? fmtNum(v) : String(v)
    default: return fmtNum(v)
  }
}

export const isNumericFormat = (f: string) => f !== 'text' && f !== 'stars'

let counter = 0
export const uid = (prefix = 'id') => `${prefix}_${Date.now().toString(36)}_${(counter++).toString(36)}${Math.random().toString(36).slice(2, 6)}`

export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

export const svgUrl = (svg: string) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`

export function safeName(s: string) {
  return s.replace(/[^\w\-. ]+/g, '_').trim() || 'untitled'
}
