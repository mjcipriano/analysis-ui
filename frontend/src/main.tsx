import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { restoreSession } from './store'
import './styles.css'

restoreSession().finally(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
})
