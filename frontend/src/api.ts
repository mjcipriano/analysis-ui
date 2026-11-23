import type { Catalog, Results, WireColumn, WireSheet } from './types'

async function post<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok && res.headers.get('content-type')?.includes('json') !== true) {
    throw new Error(`The server returned an error (${res.status}).`)
  }
  return res.json()
}

export type Reply<T> = ({ status: 'ok' } & T) | { status: 'error'; message: string }

export const api = {
  catalog: async (): Promise<Catalog> => (await fetch('/api/catalog')).json(),

  runAnalysis: (analysis: string, options: object, columns: WireColumn[], signal?: AbortSignal) =>
    post<Reply<{ results: Results }>>('/api/analyses/run', { analysis, options, data: { columns } }, signal),

  renderGraph: (graph: object, columns: WireColumn[], signal?: AbortSignal) =>
    post<Reply<{ svg: string; meta: any }>>('/api/graphs/render', { graph, data: { columns } }, signal),

  exportGraph: async (graph: object, columns: WireColumn[], format: string, dpi: number, editable_text: boolean): Promise<Blob> => {
    const res = await fetch('/api/graphs/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph, data: { columns }, format, dpi, editable_text }),
    })
    if (!res.ok) {
      let msg = `Export failed (${res.status}).`
      try { msg = (await res.json()).message ?? msg } catch { /* keep default */ }
      throw new Error(msg)
    }
    return res.blob()
  },

  renderLayout: (layout: object, signal?: AbortSignal) =>
    post<Reply<{ svg: string }>>('/api/layouts/render', { layout }, signal),

  exportLayout: async (layout: object, format: string, dpi: number): Promise<Blob> => {
    const res = await fetch('/api/layouts/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ layout, format, dpi }),
    })
    if (!res.ok) {
      let msg = `Export failed (${res.status}).`
      try { msg = (await res.json()).message ?? msg } catch { /* keep default */ }
      throw new Error(msg)
    }
    return res.blob()
  },

  importFile: async (file: File): Promise<Reply<{ sheets: WireSheet[] }>> => {
    const form = new FormData()
    form.append('file', file)
    return (await fetch('/api/import/file', { method: 'POST', body: form })).json()
  },

  importText: (text: string, name = 'Pasted data') =>
    post<Reply<{ sheets: WireSheet[] }>>('/api/import/text', { text, name }),

  transform: (body: object) => post<Reply<{ columns: WireColumn[] }>>('/api/transform', body),

  samples: async (): Promise<{ id: string; name: string; description: string; hint: string }[]> =>
    (await fetch('/api/samples')).json(),

  sample: async (id: string): Promise<Reply<{ sheets: WireSheet[] }>> => (await fetch(`/api/samples/${id}`)).json(),
}
