export type ColType = 'continuous' | 'nominal' | 'ordinal'
export type Cell = number | string | null

export interface Column {
  id: string
  name: string
  type: ColType
  values: Cell[]
  /** explicit level order for nominal / ordinal columns */
  levels?: string[]
  /** true once the user picked the type by hand (stops auto-detection) */
  typeLocked?: boolean
}

export interface Sheet {
  id: string
  name: string
  columns: Column[]
  nrows: number
  /** bumped on every data change; analyses re-run when it differs */
  version: number
}

export interface OptionDef {
  type: 'variable' | 'variables' | 'choice' | 'bool' | 'number' | 'integer' | 'text' | 'level' | 'color' | 'group_colors' | 'annotations'
  name: string
  label: string
  default: any
  section?: string
  visible?: Record<string, any>
  help?: string
  permitted?: ColType[]
  required?: boolean
  min?: number | null
  max?: number | null
  step?: number | null
  nullable?: boolean
  placeholder?: string
  multiline?: boolean
  choices?: { value: string; label: string }[]
  sources?: string[]
}

export interface AnalysisDef {
  id: string
  name: string
  category: string
  engine: 'python' | 'r'
  description: string
  keywords: string
  options: OptionDef[]
  available: boolean
  unavailable_reason: string | null
}

export interface GraphDef {
  id: string
  name: string
  category: string
  description: string
  render_type: string
  icon: string
  options: OptionDef[]
}

export interface Palette { id: string; label: string; colors: string[]; sequential?: boolean }
export interface Theme { id: string; label: string; bg: string; panel_bg: string; text: string; axis: string; grid_color: string; grid: string; frame: string }

export interface Catalog {
  version: string
  engines: { python: boolean; r: boolean; r_version: string | null }
  analysis_categories: string[]
  analyses: AnalysisDef[]
  graph_categories: string[]
  graphs: GraphDef[]
  palettes: Palette[]
  colormaps: Palette[]
  themes: Theme[]
  transforms: { id: string; label: string; group: string }[]
  import_extensions: string[]
}

export type ResultElement =
  | { type: 'heading'; text: string }
  | { type: 'text'; title?: string | null; content: string; kind: 'note' | 'summary' | 'warning' }
  | { type: 'table'; title: string; columns: { key: string; label: string; format: string }[]; rows: Record<string, any>[]; footnotes: string[] }
  | { type: 'plot'; title: string; svg: string }
  | { type: 'graph'; label: string; graph: { type: string; options: Record<string, any> } }

export interface Results { elements: ResultElement[]; warnings: string[] }

export type RunStatus = 'idle' | 'running' | 'ok' | 'error' | 'incomplete'

export interface AnalysisItem {
  id: string
  kind: 'analysis'
  sheetId: string
  analysisId: string
  name: string
  options: Record<string, any>
  status: RunStatus
  results?: Results
  message?: string
  /** signature of (options, data version) of the last finished run */
  ranKey?: string
}

export interface GraphItem {
  id: string
  kind: 'graph'
  sheetId: string
  graphType: string
  engine: 'matplotlib' | 'ggplot2'
  name: string
  options: Record<string, any>
  status: RunStatus
  svg?: string
  meta?: { groups: { name: string; color: string }[]; warnings: string[] }
  message?: string
  ranKey?: string
}

export type Selection =
  | { kind: 'welcome' }
  | { kind: 'sheet'; id: string }
  | { kind: 'analysis'; id: string }
  | { kind: 'graph'; id: string }
  | { kind: 'layout'; id: string }

export interface WireColumn { name: string; type: ColType; values: Cell[]; levels?: string[] }
export interface WireSheet { name: string; columns: WireColumn[] }

export type Annotation =
  | { kind: 'text'; x: number; y: number; text: string; size?: number | null; color?: string; bold?: boolean; italic?: boolean; angle?: number; align?: 'left' | 'center' | 'right' }
  | { kind: 'arrow' | 'line'; x: number; y: number; x2: number; y2: number; color?: string; width?: number }

/** Several graphs arranged on one page with panel labels. */
export interface LayoutItem {
  id: string
  name: string
  graphIds: string[]
  cols: number
  width: number
  panelHeight: number
  gap: number
  labels: 'A' | 'a' | '1' | 'none'
  labelSize: number
  status: RunStatus
  svg?: string
  message?: string
  ranKey?: string
}
