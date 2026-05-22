import { useEffect, useState } from 'react'
import EntitySelect from './EntitySelect.jsx'
import { api } from '../api.js'

const PRODUCT_LABELS = {
  C: 'C — Consumer / digital',
  H: 'H — Hosting',
  R: 'R — Recurring',
  S: 'S — Software',
}

export default function TransactionForm({ onResult, onLoading }) {
  const [options, setOptions] = useState(null)
  const [catalog, setCatalog] = useState({ device: [], card: [], address: [], email_domain: [] })
  const [form, setForm] = useState({
    amount: 75,
    product: 'C',
    card_network: 'visa',
    card_type: 'credit',
    device: '',
    card: '',
    address: '',
    email_domain: '',
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      api.options(),
      api.entities('device', { limit: 500 }),
      api.entities('card', { limit: 500 }),
      api.entities('address', { limit: 500 }),
      api.entities('email_domain', { limit: 500 }),
    ])
      .then(([opts, d, c, a, e]) => {
        setOptions(opts)
        setCatalog({ device: d.items, card: c.items, address: a.items, email_domain: e.items })
      })
      .catch((err) => setError(err.message))
  }, [])

  function update(field, value) {
    setForm((f) => ({ ...f, [field]: value }))
  }

  async function submit(e) {
    e?.preventDefault()
    setSubmitting(true)
    setError(null)
    onLoading?.(true)
    try {
      const payload = { ...form, amount: Number(form.amount) }
      // Strip empty entity selections so backend treats them as unseen
      for (const k of ['device', 'card', 'address', 'email_domain']) {
        if (!payload[k]) delete payload[k]
      }
      const result = await api.predict(payload)
      onResult(result, form)
    } catch (err) {
      setError(err.message)
      onResult(null, form)
    } finally {
      setSubmitting(false)
      onLoading?.(false)
    }
  }

  function applyPreset(kind) {
    if (kind === 'fraud_ring') {
      // Pick the top-fraud device/card so the model has rich negative context.
      const dev = catalog.device[0]?.value || ''
      const card = catalog.card[0]?.value || ''
      setForm((f) => ({
        ...f,
        amount: 1.5,
        product: 'C',
        card_network: 'visa',
        card_type: 'credit',
        device: dev,
        card,
        address: '',
        email_domain: 'gmail.com',
      }))
    } else if (kind === 'clean') {
      // Lowest-risk known entities, normal ticket size.
      const cleanDev = [...catalog.device].sort((a, b) => a.fraud_rate - b.fraud_rate || b.txn_count - a.txn_count)[0]?.value || ''
      const cleanEmail = [...catalog.email_domain].sort((a, b) => a.fraud_rate - b.fraud_rate)[0]?.value || ''
      setForm((f) => ({
        ...f,
        amount: 89,
        product: 'H',
        card_network: 'visa',
        card_type: 'debit',
        device: cleanDev,
        card: '',
        address: '',
        email_domain: cleanEmail,
      }))
    }
  }

  if (error && !options) {
    return (
      <div className="card p-6 text-sm text-red-600">
        Couldn't load model metadata: {error}
        <div className="mt-2 text-xs text-navy-500">Is the FastAPI server running on :8000?</div>
      </div>
    )
  }

  if (!options) {
    return <div className="card p-6 text-sm text-navy-500">Loading model metadata…</div>
  }

  return (
    <form onSubmit={submit} className="card p-6 md:p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="font-display text-xl font-bold text-navy-900">
            Transaction analyzer
          </h2>
          <p className="mt-1 text-sm text-navy-500">
            Score a transaction against the trained GraphSAGE model.
          </p>
        </div>
        <div className="hidden gap-2 md:flex">
          <button type="button" onClick={() => applyPreset('clean')} className="btn-ghost">
            Clean preset
          </button>
          <button type="button" onClick={() => applyPreset('fraud_ring')} className="btn-ghost">
            Fraud-ring preset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div>
          <label className="field-label">Amount (USD)</label>
          <div className="relative mt-1.5">
            <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-navy-400">$</span>
            <input
              type="number"
              min="0.01"
              step="0.01"
              required
              value={form.amount}
              onChange={(e) => update('amount', e.target.value)}
              className="field-input pl-7 font-mono"
            />
          </div>
        </div>

        <Select
          label="Product"
          value={form.product}
          onChange={(v) => update('product', v)}
          options={options.product}
          render={(v) => PRODUCT_LABELS[v] || v}
        />

        <Select
          label="Card network"
          value={form.card_network}
          onChange={(v) => update('card_network', v)}
          options={options.card_network}
        />

        <Select
          label="Card type"
          value={form.card_type}
          onChange={(v) => update('card_type', v)}
          options={options.card_type}
        />

        <EntitySelect
          label="Device"
          placeholder="Select device fingerprint…"
          options={catalog.device}
          value={form.device}
          onChange={(v) => update('device', v)}
          hint={`${catalog.device.length} known`}
        />

        <EntitySelect
          label="Card ID"
          placeholder="Select card identifier…"
          options={catalog.card}
          value={form.card}
          onChange={(v) => update('card', v)}
          hint={`${catalog.card.length} known`}
        />

        <EntitySelect
          label="Billing address"
          placeholder="Select address code…"
          options={catalog.address}
          value={form.address}
          onChange={(v) => update('address', v)}
          hint={`${catalog.address.length} known`}
        />

        <EntitySelect
          label="Email domain"
          placeholder="Select email domain…"
          options={catalog.email_domain}
          value={form.email_domain}
          onChange={(v) => update('email_domain', v)}
          hint={`${catalog.email_domain.length} known`}
        />
      </div>

      {error && (
        <div className="mt-5 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-7 flex flex-col-reverse gap-3 border-t border-navy-100 pt-5 md:flex-row md:items-center md:justify-between">
        <div className="text-xs text-navy-500">
          Predictions run against a live GraphSAGE model with 23 features and 4 edge types.
        </div>
        <div className="flex gap-2 md:hidden">
          <button type="button" onClick={() => applyPreset('clean')} className="btn-ghost flex-1">Clean</button>
          <button type="button" onClick={() => applyPreset('fraud_ring')} className="btn-ghost flex-1">Fraud ring</button>
        </div>
        <button type="submit" disabled={submitting} className="btn-primary">
          {submitting ? 'Scoring…' : 'Analyze transaction'}
          {!submitting && <ArrowRight />}
        </button>
      </div>
    </form>
  )
}

function Select({ label, value, onChange, options, render }) {
  return (
    <div>
      <label className="field-label">{label}</label>
      <div className="mt-1.5">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="field-input appearance-none bg-no-repeat bg-right pr-9"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 20 20' fill='none'><path d='M5 7L10 12L15 7' stroke='%23577' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg>\")",
            backgroundPosition: 'right 0.75rem center',
          }}
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {render ? render(o) : o}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

function ArrowRight() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
      <path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
