import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import { missingInputs, referencedColumns, toWire } from './data'
import { useStore } from './store'
import type { AnalysisDef, AnalysisItem, GraphDef, GraphItem, OptionDef, Sheet } from './types'

const DEBOUNCE_MS = 280

function wireFor(schema: OptionDef[], options: Record<string, any>, sheet: Sheet) {
  return toWire(sheet, referencedColumns(schema, options))
}

/** Keeps an analysis up to date with its options and data while it is on screen. */
export function useAnalysisRunner(item: AnalysisItem, def: AnalysisDef | undefined, sheet: Sheet | undefined) {
  const patchItem = useStore((s) => s.patchItem)
  const [nonce, setNonce] = useState(0)
  const key = useMemo(() => JSON.stringify([item.analysisId, item.options, sheet?.id, sheet?.version, nonce]), [item.analysisId, item.options, sheet?.id, sheet?.version, nonce])

  useEffect(() => {
    if (!def || !sheet || item.ranKey === key) return
    if (!def.available) {
      patchItem(item.id, { status: 'error', message: `This analysis is unavailable: ${def.unavailable_reason ?? 'engine missing'}.`, ranKey: key })
      return
    }
    const missing = missingInputs(def.options, item.options, sheet)
    if (missing.length) {
      patchItem(item.id, { status: 'incomplete', message: missing.join(' · '), ranKey: key })
      return
    }
    const ctrl = new AbortController()
    const timer = window.setTimeout(async () => {
      patchItem(item.id, { status: 'running' })
      try {
        const res = await api.runAnalysis(def.id, item.options, wireFor(def.options, item.options, sheet), ctrl.signal)
        if (ctrl.signal.aborted) return
        if (res.status === 'ok') patchItem(item.id, { status: 'ok', results: res.results, message: undefined, ranKey: key })
        else patchItem(item.id, { status: 'error', message: res.message, ranKey: key })
      } catch (e: any) {
        if (ctrl.signal.aborted) return
        patchItem(item.id, { status: 'error', message: e?.message ?? 'The request failed.', ranKey: key })
      }
    }, DEBOUNCE_MS)
    return () => { window.clearTimeout(timer); ctrl.abort() }
  }, [key, item.ranKey, def, sheet, item.id])   // eslint-disable-line react-hooks/exhaustive-deps

  return { rerun: () => setNonce((n) => n + 1) }
}

export function graphSpec(item: GraphItem) {
  return { type: item.graphType, engine: item.engine, options: item.options }
}

/** Keeps a graph preview up to date while it is on screen. */
export function useGraphRunner(item: GraphItem, def: GraphDef | undefined, sheet: Sheet | undefined) {
  const patchItem = useStore((s) => s.patchItem)
  const [nonce, setNonce] = useState(0)
  const key = useMemo(() => JSON.stringify([item.graphType, item.engine, item.options, sheet?.id, sheet?.version, nonce]), [item.graphType, item.engine, item.options, sheet?.id, sheet?.version, nonce])

  useEffect(() => {
    if (!def || !sheet || item.ranKey === key) return
    const missing = missingInputs(def.options, item.options, sheet)
    if (missing.length) {
      patchItem(item.id, { status: 'incomplete', message: missing.join(' · '), ranKey: key })
      return
    }
    const ctrl = new AbortController()
    const timer = window.setTimeout(async () => {
      patchItem(item.id, { status: 'running' })
      try {
        const res = await api.renderGraph(graphSpec(item), wireFor(def.options, item.options, sheet), ctrl.signal)
        if (ctrl.signal.aborted) return
        if (res.status === 'ok') patchItem(item.id, { status: 'ok', svg: res.svg, meta: res.meta, message: undefined, ranKey: key })
        else patchItem(item.id, { status: 'error', message: res.message, ranKey: key })
      } catch (e: any) {
        if (ctrl.signal.aborted) return
        patchItem(item.id, { status: 'error', message: e?.message ?? 'The request failed.', ranKey: key })
      }
    }, DEBOUNCE_MS)
    return () => { window.clearTimeout(timer); ctrl.abort() }
  }, [key, item.ranKey, def, sheet, item.id])   // eslint-disable-line react-hooks/exhaustive-deps

  return { rerun: () => setNonce((n) => n + 1) }
}

export function wireForItem(schema: OptionDef[], options: Record<string, any>, sheet: Sheet) {
  return wireFor(schema, options, sheet)
}
