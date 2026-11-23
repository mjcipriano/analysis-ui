import { fmtCell, isNumericFormat, svgUrl } from '../format'
import { useStore } from '../store'
import type { ResultElement, Results } from '../types'
import { Icon } from './ui'

function tableToTsv(el: Extract<ResultElement, { type: 'table' }>): string {
  const head = el.columns.map((c) => c.label).join('\t')
  const rows = el.rows.map((r) => el.columns.map((c) => fmtCell(r[c.key], c.format)).join('\t'))
  return [el.title, head, ...rows, ...el.footnotes].join('\n')
}

function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

/** A self-contained HTML report of one analysis (for saving / printing / pasting into Word). */
export function resultsToHtml(title: string, results: Results): string {
  const parts: string[] = []
  for (const el of results.elements) {
    if (el.type === 'heading') parts.push(`<h2>${escapeHtml(el.text)}</h2>`)
    else if (el.type === 'text') parts.push(`<div class="${el.kind}">${el.title ? `<b>${escapeHtml(el.title)}</b><br>` : ''}${escapeHtml(el.content).replace(/\n/g, '<br>')}</div>`)
    else if (el.type === 'plot') parts.push(`<figure><figcaption>${escapeHtml(el.title)}</figcaption>${el.svg.replace(/<\?xml[^>]*>|<!DOCTYPE[^>]*>/g, '')}</figure>`)
    else if (el.type === 'table') {
      parts.push(`<table><caption>${escapeHtml(el.title)}</caption><thead><tr>${el.columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join('')}</tr></thead><tbody>${
        el.rows.map((r) => `<tr>${el.columns.map((c) => `<td class="${isNumericFormat(c.format) ? 'num' : ''}">${escapeHtml(fmtCell(r[c.key], c.format))}</td>`).join('')}</tr>`).join('')
      }</tbody></table>${el.footnotes.map((f) => `<p class="fn">${escapeHtml(f)}</p>`).join('')}`)
    }
  }
  return `<!doctype html><html><head><meta charset="utf-8"><title>${escapeHtml(title)}</title><style>
body{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1f2937;max-width:900px;margin:32px auto;padding:0 20px}
h1{font-size:22px}h2{font-size:16px;margin-top:28px}table{border-collapse:collapse;margin:18px 0 4px}caption{text-align:left;font-weight:600;padding-bottom:6px}
th,td{padding:4px 12px;text-align:left}th{border-top:2px solid #111;border-bottom:1px solid #111;font-weight:600}tbody tr:last-child td{border-bottom:2px solid #111}
td.num{text-align:right;font-variant-numeric:tabular-nums}.fn{font-size:12px;color:#6b7280;margin:2px 0}.summary{background:#eef2ff;border-left:3px solid #4f46e5;padding:10px 14px;margin:16px 0}
.warning{background:#fffbeb;border-left:3px solid #d97706;padding:10px 14px;margin:16px 0}.note{color:#4b5563;margin:10px 0}figure{margin:18px 0}figcaption{font-weight:600;margin-bottom:6px}
</style></head><body><h1>${escapeHtml(title)}</h1>${parts.join('\n')}</body></html>`
}

export function ResultsView({ results, sheetId }: { results: Results; sheetId: string }) {
  const toast = useStore((s) => s.toast)
  const addGraph = useStore((s) => s.addGraph)
  const copy = async (text: string) => {
    try { await navigator.clipboard.writeText(text); toast('Table copied — paste it into Excel or Word.') }
    catch { toast('Copying is not allowed by the browser here.', 'error') }
  }
  // The plain-language conclusion goes first; everything else keeps its order.
  const elements = [
    ...results.elements.filter((el) => el.type === 'text' && el.kind === 'summary'),
    ...results.elements.filter((el) => !(el.type === 'text' && el.kind === 'summary')),
  ]
  return (
    <div className="results">
      {elements.map((el, i) => {
        switch (el.type) {
          case 'heading':
            return <h3 key={i} className="res-heading">{el.text}</h3>
          case 'text':
            return (
              <div key={i} className={`res-text ${el.kind}`}>
                {el.kind === 'warning' && <Icon name="warn" size={15} />}
                {el.kind === 'summary' && <Icon name="sparkle" size={15} />}
                <div>
                  {el.title && <div className="res-text-title">{el.title}</div>}
                  {el.content.split('\n').map((line, j) => <p key={j}>{line}</p>)}
                </div>
              </div>
            )
          case 'table':
            return (
              <div key={i} className="res-table-wrap">
                <div className="res-table-title">
                  <span>{el.title}</span>
                  <button className="icon-btn small ghost" title="Copy table" onClick={() => copy(tableToTsv(el))}><Icon name="copy" size={13} /></button>
                </div>
                <div className="res-table-scroll">
                  <table className="res-table">
                    <thead>
                      <tr>{el.columns.map((c) => <th key={c.key} className={isNumericFormat(c.format) ? 'num' : ''}>{c.label}</th>)}</tr>
                    </thead>
                    <tbody>
                      {el.rows.map((r, ri) => (
                        <tr key={ri}>
                          {el.columns.map((c) => {
                            const text = fmtCell(r[c.key], c.format)
                            const sig = c.format === 'p' && typeof r[c.key] === 'number' && r[c.key] < 0.05
                            return <td key={c.key} className={`${isNumericFormat(c.format) ? 'num' : ''} ${sig ? 'sig' : ''} ${c.format === 'stars' ? 'stars' : ''}`}>{text}</td>
                          })}
                        </tr>
                      ))}
                      {!el.rows.length && <tr><td colSpan={el.columns.length} className="muted">No rows</td></tr>}
                    </tbody>
                  </table>
                </div>
                {el.footnotes.map((f, fi) => <div key={fi} className="res-footnote">{f}</div>)}
              </div>
            )
          case 'plot':
            return (
              <figure key={i} className="res-plot">
                <figcaption>{el.title}</figcaption>
                <img src={svgUrl(el.svg)} alt={el.title} />
              </figure>
            )
          case 'graph':
            return (
              <button key={i} className="btn accent-soft res-graph-link" onClick={() => addGraph(sheetId, el.graph.type, el.graph.options)}>
                <Icon name="graph" size={15} /> {el.label}
              </button>
            )
          default:
            return null
        }
      })}
      {results.warnings?.length > 0 && (
        <div className="res-text warning">
          <Icon name="warn" size={15} />
          <div>{results.warnings.map((w, i) => <p key={i}>{w}</p>)}</div>
        </div>
      )}
    </div>
  )
}
