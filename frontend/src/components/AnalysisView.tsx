import { download, safeName } from '../format'
import { useAnalysisRunner } from '../runner'
import { useStore } from '../store'
import type { AnalysisItem } from '../types'
import { OptionsForm } from './OptionsForm'
import { ResultsView, resultsToHtml } from './ResultsView'
import { EditableText, Icon, MenuButton, Spinner } from './ui'

export function AnalysisView({ item }: { item: AnalysisItem }) {
  const def = useStore((s) => s.catalog?.analyses.find((a) => a.id === item.analysisId))
  const sheet = useStore((s) => s.sheets.find((x) => x.id === item.sheetId))
  const { setOptions, renameItem, deleteItem, duplicateItem, resetOptions, select } = useStore.getState()
  const { rerun } = useAnalysisRunner(item, def, sheet)

  if (!def) return <div className="empty-state"><h3>Unknown analysis</h3><p>“{item.analysisId}” is not available in this version.</p></div>

  const exportHtml = () => {
    if (!item.results) return
    download(new Blob([resultsToHtml(item.name, item.results)], { type: 'text/html' }), `${safeName(item.name)}.html`)
  }

  return (
    <div className="split">
      <aside className="panel left">
        <div className="panel-head">
          <div>
            <div className="panel-title">{def.name}</div>
            <div className="panel-sub">
              <span className={`engine-badge ${def.engine}`}>{def.engine === 'r' ? 'R' : 'Python'}</span>
              {sheet && <button className="link" onClick={() => select({ kind: 'sheet', id: sheet.id })}>{sheet.name}</button>}
            </div>
          </div>
          <button className="icon-btn" title="Reset options to defaults" onClick={() => resetOptions(item.id)}><Icon name="refresh" /></button>
        </div>
        {def.description && <div className="panel-desc">{def.description}</div>}
        <OptionsForm schema={def.options} values={item.options} sheet={sheet} scope={`a:${def.id}`}
          onChange={(patch) => setOptions(item.id, patch)} openSections={['Data', 'Options', 'Model', 'Test']} />
      </aside>
      <section className="content">
        <div className="content-head">
          <EditableText className="content-title" value={item.name} onCommit={(v) => renameItem(item.id, v)} />
          <div className="content-actions">
            {item.status === 'running' && <span className="status"><Spinner /> Calculating…</span>}
            <MenuButton className="btn" items={[
              { label: 'Save report as HTML', icon: 'download', onClick: exportHtml, disabled: !item.results },
              { label: 'Print…', icon: 'results', disabled: !item.results, onClick: () => {
                const w = window.open('', '_blank')
                if (w && item.results) { w.document.write(resultsToHtml(item.name, item.results)); w.document.close(); w.focus(); w.print() }
              } },
              { divider: true },
              { label: 'Recalculate', icon: 'refresh', onClick: rerun },
              { label: 'Duplicate', icon: 'copy', onClick: () => duplicateItem(item.id) },
              { label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteItem(item.id) },
            ]}><Icon name="more" /></MenuButton>
          </div>
        </div>
        <div className={`content-body ${item.status === 'running' ? 'stale' : ''}`}>
          {!sheet && <div className="empty-state"><h3>The data table for this analysis was deleted.</h3></div>}
          {sheet && item.status === 'incomplete' && (
            <div className="empty-state">
              <Icon name="sliders" size={28} />
              <h3>Choose your data</h3>
              <p>Pick columns on the left to see results: <b>{item.message}</b></p>
            </div>
          )}
          {sheet && item.status === 'error' && (
            <div className="error-box">
              <Icon name="warn" size={18} />
              <div>
                <div className="error-title">This analysis could not be completed</div>
                {(item.message ?? '').split('\n').map((l, i) => <p key={i}>{l}</p>)}
              </div>
            </div>
          )}
          {sheet && item.results && item.status !== 'incomplete' && item.status !== 'error' && (
            <ResultsView results={item.results} sheetId={item.sheetId} />
          )}
          {sheet && !item.results && (item.status === 'idle' || item.status === 'running') && (
            <div className="empty-state"><Spinner size={22} /><p>Calculating…</p></div>
          )}
        </div>
      </section>
    </div>
  )
}
