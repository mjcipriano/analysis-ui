import { useEffect, useRef, useState } from 'react'
import { AnalysisView } from './components/AnalysisView'
import { GraphView } from './components/GraphView'
import { LayoutView } from './components/LayoutView'
import { AnalysisPicker, GraphGallery } from './components/Pickers'
import { SheetView } from './components/SheetView'
import { Sidebar } from './components/Sidebar'
import { EditableText, Icon, Menu, MenuButton, Spinner } from './components/ui'
import { importFiles, PasteDialog, Welcome } from './components/Welcome'
import { download, safeName } from './format'
import { projectSnapshot, useStore } from './store'
import type { Sheet } from './types'

export default function App() {
  const catalog = useStore((s) => s.catalog)
  const catalogError = useStore((s) => s.catalogError)
  const selection = useStore((s) => s.selection)
  const sheets = useStore((s) => s.sheets)
  const items = useStore((s) => s.items)
  const layouts = useStore((s) => s.layouts)
  const name = useStore((s) => s.name)
  const dark = useStore((s) => s.dark)
  const toasts = useStore((s) => s.toasts)
  const { loadCatalog, setDark, setProjectName, newProject, select, addBlankSheet, dismissToast } = useStore.getState()
  const [analyzeFor, setAnalyzeFor] = useState<Sheet | null>(null)
  const [graphFor, setGraphFor] = useState<Sheet | null>(null)
  const [paste, setPaste] = useState(false)
  const [dataMenu, setDataMenu] = useState<{ x: number; y: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const file = useRef<HTMLInputElement>(null)

  useEffect(() => { loadCatalog() }, [])   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { document.documentElement.dataset.theme = dark ? 'dark' : 'light' }, [dark])
  useEffect(() => { document.title = `${name} — Analysis UI` }, [name])

  const saveProject = () => {
    download(new Blob([JSON.stringify(projectSnapshot(true))], { type: 'application/json' }), `${safeName(name)}.aui.json`)
  }

  if (catalogError) {
    return (
      <div className="boot">
        <Icon name="warn" size={30} />
        <h2>{catalogError}</h2>
        <p>Start it with <code>analysis-ui</code> (or <code>./run.sh</code>) and reload this page.</p>
        <button className="btn primary" onClick={loadCatalog}>Try again</button>
      </div>
    )
  }
  if (!catalog) return <div className="boot"><Spinner size={26} /><p>Starting the R and Python engines…</p></div>

  const sheet = selection.kind === 'sheet' ? sheets.find((s) => s.id === selection.id) : undefined
  const item = selection.kind === 'analysis' || selection.kind === 'graph' ? items.find((i) => i.id === selection.id) : undefined
  const layout = selection.kind === 'layout' ? layouts.find((l) => l.id === selection.id) : undefined
  const showWelcome = selection.kind === 'welcome' || (!sheet && !item && !layout)

  return (
    <div className={`app ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); setDragging(true) } }}
      onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false) }}
      onDrop={(e) => { if (e.dataTransfer.files.length) { e.preventDefault(); setDragging(false); importFiles(e.dataTransfer.files) } }}>
      <header className="topbar">
        <button className="brand" onClick={() => select({ kind: 'welcome' })} title="Home">
          <svg width="22" height="22" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#4F46E5" /><rect x="6" y="17" width="4.500" height="9" rx="1" fill="#fff" opacity=".75" /><rect x="13.750" y="10" width="4.500" height="16" rx="1" fill="#fff" /><rect x="21.500" y="14" width="4.500" height="12" rx="1" fill="#fff" opacity=".75" /><path d="M8 11.500 16 5.500l8 4" fill="none" stroke="#FDE68A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
          <span>Analysis UI</span>
        </button>
        <EditableText className="project-name" value={name} onCommit={setProjectName} />
        <div className="topbar-actions">
          <MenuButton className="btn ghost" items={[
            { label: 'New project', icon: 'plus', onClick: () => { if (!sheets.length || confirm('Start a new project? Unsaved work in the current project will be lost.')) newProject() } },
            { label: 'Open project or data file…', icon: 'folder', onClick: () => file.current?.click() },
            { label: 'Save project', icon: 'save', onClick: saveProject, disabled: !sheets.length, hint: '.aui.json' },
          ]}><Icon name="folder" /> File <Icon name="down" size={12} /></MenuButton>
          <button className="btn ghost" onClick={() => file.current?.click()}><Icon name="upload" /> Import</button>
          <span className="engines" title={`Python ready · ${catalog.engines.r ? `R ${catalog.engines.r_version} ready` : 'R not found — R analyses and ggplot2 graphs are disabled'}`}>
            <span className="engine-pill ok">Python</span>
            <span className={`engine-pill ${catalog.engines.r ? 'ok' : 'off'}`}>R</span>
          </span>
          <button className="icon-btn" title={dark ? 'Light mode' : 'Dark mode'} onClick={() => setDark(!dark)}><Icon name={dark ? 'sun' : 'moon'} /></button>
        </div>
        <input ref={file} type="file" hidden multiple accept={[...catalog.import_extensions, '.aui', '.json'].join(',')}
          onChange={(e) => { if (e.target.files) importFiles(e.target.files); e.target.value = '' }} />
      </header>
      <div className="workspace">
        <Sidebar onAnalyze={setAnalyzeFor} onGraph={setGraphFor} onNewData={(e) => { const r = e.currentTarget.getBoundingClientRect(); setDataMenu({ x: r.left, y: r.bottom + 4 }) }} />
        <main className="main">
          {showWelcome && <Welcome />}
          {!showWelcome && sheet && <SheetView key={sheet.id} sheet={sheet} onAnalyze={() => setAnalyzeFor(sheet)} onGraph={() => setGraphFor(sheet)} />}
          {!showWelcome && item?.kind === 'analysis' && <AnalysisView key={item.id} item={item} />}
          {!showWelcome && item?.kind === 'graph' && <GraphView key={item.id} item={item} />}
          {!showWelcome && layout && <LayoutView key={layout.id} layout={layout} />}
        </main>
      </div>
      {dataMenu && <Menu at={dataMenu} onClose={() => setDataMenu(null)} items={[
        { label: 'Empty table', icon: 'table', onClick: addBlankSheet },
        { label: 'Import file…', icon: 'upload', onClick: () => file.current?.click() },
        { label: 'Paste data…', icon: 'paste', onClick: () => setPaste(true) },
        { label: 'Example data…', icon: 'sparkle', onClick: () => select({ kind: 'welcome' }) },
      ]} />}
      {analyzeFor && <AnalysisPicker sheet={analyzeFor} onClose={() => setAnalyzeFor(null)} />}
      {graphFor && <GraphGallery sheet={graphFor} onClose={() => setGraphFor(null)} />}
      {paste && <PasteDialog onClose={() => setPaste(false)} />}
      {dragging && <div className="drop-overlay"><Icon name="upload" size={34} /><div>Drop files to import</div></div>}
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`} onClick={() => dismissToast(t.id)}>
            <Icon name={t.kind === 'error' ? 'warn' : 'check'} size={15} /> <span>{t.text}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
