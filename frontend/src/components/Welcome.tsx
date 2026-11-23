import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { Icon, Modal, Spinner } from './ui'

export async function importFiles(files: FileList | File[]) {
  const { addWireSheets, toast, loadProject } = useStore.getState()
  for (const file of Array.from(files)) {
    if (/\.aui(\.json)?$/i.test(file.name)) {
      try {
        const p = JSON.parse(await file.text())
        if (p.format !== 'analysis-ui-project') throw new Error('not a project')
        loadProject(p)
        toast(`Opened “${p.name}”.`)
      } catch { toast(`“${file.name}” is not a valid Analysis UI project.`, 'error') }
      continue
    }
    try {
      const res = await api.importFile(file)
      if (res.status === 'ok') { addWireSheets(res.sheets); toast(`Imported ${file.name}.`) }
      else toast(res.message, 'error')
    } catch { toast(`Could not import ${file.name}.`, 'error') }
  }
}

export function PasteDialog({ onClose }: { onClose: () => void }) {
  const { addWireSheets, toast } = useStore.getState()
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const go = async () => {
    setBusy(true)
    try {
      const res = await api.importText(text)
      if (res.status === 'ok') { addWireSheets(res.sheets); onClose() } else toast(res.message, 'error')
    } finally { setBusy(false) }
  }
  return (
    <Modal title="Paste data" onClose={onClose} width={640}
      footer={<><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" disabled={!text.trim() || busy} onClick={go}>{busy && <Spinner />} Create table</button></>}>
      <p className="muted">Copy cells from Excel, Google Sheets or a CSV file and paste them here. The first row is used as column names when it contains text.</p>
      <textarea className="paste-area" autoFocus rows={12} value={text} onChange={(e) => setText(e.target.value)}
        placeholder={'Control\tDrug A\tDrug B\n98.2\t84.1\t70.3\n101.5\t90.7\t68.9\n…'} />
    </Modal>
  )
}

const HINT_ICON: Record<string, string> = { column: 'graph', paired: 'layers', grouped: 'graph', xy: 'sliders', survival: 'results', contingency: 'table', roc: 'results', volcano: 'sparkle', multivariate: 'layers', agreement: 'results', forest: 'results', waterfall: 'graph' }

export function Welcome() {
  const { addBlankSheet, addWireSheets, toast } = useStore.getState()
  const catalog = useStore((s) => s.catalog)
  const [samples, setSamples] = useState<{ id: string; name: string; description: string; hint: string }[]>([])
  const [loading, setLoading] = useState<string | null>(null)
  const [paste, setPaste] = useState(false)
  const file = useRef<HTMLInputElement>(null)
  useEffect(() => { api.samples().then(setSamples).catch(() => {}) }, [])
  const open = async (id: string) => {
    setLoading(id)
    try {
      const res = await api.sample(id)
      if (res.status === 'ok') addWireSheets(res.sheets); else toast(res.message, 'error')
    } finally { setLoading(null) }
  }
  return (
    <div className="welcome">
      <div className="welcome-inner">
        <h1>Analyze and graph your data</h1>
        <p className="welcome-sub">
          Point-and-click statistics with publication-quality graphs.
          {catalog && <> {catalog.analyses.length} analyses and {catalog.graphs.length} graph types, computed by <b>R</b> and <b>Python</b>.</>}
        </p>
        <div className="start-cards">
          <button className="start-card" onClick={addBlankSheet}>
            <Icon name="table" size={26} /><b>New data table</b><span>Type or paste values into an empty spreadsheet</span>
          </button>
          <button className="start-card" onClick={() => file.current?.click()}>
            <Icon name="upload" size={26} /><b>Import a file</b><span>CSV, Excel, SPSS, Stata, SAS, R, JSON, Parquet — or drop a file anywhere</span>
          </button>
          <button className="start-card" onClick={() => setPaste(true)}>
            <Icon name="paste" size={26} /><b>Paste from clipboard</b><span>Copy cells from Excel or Google Sheets</span>
          </button>
        </div>
        <input ref={file} type="file" hidden multiple accept={[...(catalog?.import_extensions ?? []), '.aui', '.json'].join(',')}
          onChange={(e) => { if (e.target.files) importFiles(e.target.files); e.target.value = '' }} />
        <h2>Or start from an example</h2>
        <div className="samples">
          {samples.map((s) => (
            <button key={s.id} className="sample" onClick={() => open(s.id)} disabled={!!loading}>
              <span className="sample-icon">{loading === s.id ? <Spinner /> : <Icon name={HINT_ICON[s.hint] ?? 'table'} size={18} />}</span>
              <span><b>{s.name}</b><span>{s.description}</span></span>
            </button>
          ))}
        </div>
      </div>
      {paste && <PasteDialog onClose={() => setPaste(false)} />}
    </div>
  )
}
