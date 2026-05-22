import { useState } from 'react'
import Header from './components/Header.jsx'
import Hero from './components/Hero.jsx'
import ConsumerForm from './components/ConsumerForm.jsx'
import TransactionForm from './components/TransactionForm.jsx'
import VerdictCard from './components/VerdictCard.jsx'
import ExplanationPanel from './components/ExplanationPanel.jsx'
import TranslationPanel from './components/TranslationPanel.jsx'
import ModelSection from './components/ModelSection.jsx'
import Footer from './components/Footer.jsx'

const MODES = [
  {
    key: 'consumer',
    label: 'Consumer mode',
    blurb: 'Describe your transaction in plain language.',
  },
  {
    key: 'analyst',
    label: 'Analyst mode',
    blurb: 'Operate directly on the anonymized entity IDs from the graph.',
  },
]

export default function App() {
  const [mode, setMode] = useState('consumer')
  const [result, setResult] = useState(null)
  const [submittedForm, setSubmittedForm] = useState(null)
  const [mapping, setMapping] = useState(null)
  const [loading, setLoading] = useState(false)

  function handleResult(r, form) {
    setResult(r)
    setSubmittedForm(form)
    if (!r) setMapping(null)
  }

  function switchMode(next) {
    if (next === mode) return
    setMode(next)
    setResult(null)
    setSubmittedForm(null)
    setMapping(null)
  }

  const activeMode = MODES.find((m) => m.key === mode)

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <Hero />

      <main id="analyzer" className="bg-navy-50">
        <div className="mx-auto max-w-7xl px-6 py-16">
          <div className="mb-8 max-w-2xl">
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-navy-500">
              Live analyzer
            </div>
            <h2 className="font-display text-3xl font-extrabold text-navy-900 md:text-4xl">
              Score a transaction in real time.
            </h2>
            <p className="mt-3 text-navy-600">{activeMode.blurb}</p>
          </div>

          <ModeToggle mode={mode} onChange={switchMode} />

          <div className="mt-6 grid gap-6 lg:grid-cols-5">
            <div className="lg:col-span-3">
              {mode === 'consumer' ? (
                <ConsumerForm
                  onResult={handleResult}
                  onLoading={setLoading}
                  onSummary={setMapping}
                />
              ) : (
                <TransactionForm onResult={handleResult} onLoading={setLoading} />
              )}
            </div>
            <div className="space-y-6 lg:col-span-2">
              <VerdictCard result={result} loading={loading} form={submittedForm} />
            </div>
          </div>

          {result && mode === 'consumer' && (
            <div className="mt-6">
              <TranslationPanel mapping={mapping || result.mapping} />
            </div>
          )}

          {result && (
            <div className="mt-6">
              <ExplanationPanel result={result} />
            </div>
          )}
        </div>
      </main>

      <ModelSection />
      <Footer />
    </div>
  )
}

function ModeToggle({ mode, onChange }) {
  return (
    <div className="inline-flex rounded-lg border border-navy-200 bg-white p-1 shadow-sm">
      {MODES.map((m) => {
        const active = m.key === mode
        return (
          <button
            key={m.key}
            type="button"
            onClick={() => onChange(m.key)}
            className={`rounded-md px-4 py-2 text-sm font-semibold transition ${
              active
                ? 'bg-navy-800 text-white shadow'
                : 'text-navy-600 hover:text-navy-900'
            }`}
          >
            {m.label}
          </button>
        )
      })}
    </div>
  )
}
