import { useEffect, useState } from 'react'
import { api } from '../api.js'

/**
 * Lay-friendly form. A normal customer doesn't know their internal card ID
 * or anonymized address code — they know their email, their device type,
 * and what they were buying. The backend translates these to the model's
 * vocabulary before scoring.
 */
export default function ConsumerForm({ onResult, onLoading, onSummary }) {
  const [options, setOptions] = useState(null)
  const [form, setForm] = useState({
    amount: 89.99,
    email: '',
    device_category: 'iphone',
    merchant_category: 'shopping',
    card_network: 'visa',
    card_type: 'credit',
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.consumerOptions()
      .then(setOptions)
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
      const result = await api.consumerPredict(payload)
      onResult(result, summariseForReceipt(form, result.mapping))
      onSummary?.(result.mapping)
    } catch (err) {
      setError(err.message)
      onResult(null, null)
    } finally {
      setSubmitting(false)
      onLoading?.(false)
    }
  }

  if (error && !options) {
    return (
      <div className="card p-6 text-sm text-red-600">
        Couldn't load options: {error}
        <div className="mt-2 text-xs text-navy-500">Is the FastAPI server running on :8000?</div>
      </div>
    )
  }
  if (!options) {
    return <div className="card p-6 text-sm text-navy-500">Loading…</div>
  }

  return (
    <form onSubmit={submit} className="card p-6 md:p-8">
      <div className="mb-6 flex items-start gap-3 rounded-md border border-navy-100 bg-navy-50/50 p-4">
        <InfoIcon />
        <div className="text-xs leading-relaxed text-navy-700">
          <span className="font-semibold">You're in Consumer mode.</span> Describe your
          transaction in plain language — we'll translate it to the features the
          model was trained on, then score it against the live GraphSAGE network.
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <Field label="Amount (USD)">
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
        </Field>

        <Field label="Your email address" hint="Used to extract the email domain">
          <input
            type="email"
            placeholder="you@example.com"
            value={form.email}
            onChange={(e) => update('email', e.target.value)}
            className="field-input mt-1.5"
          />
        </Field>

        <RadioGroup
          label="Device you're paying from"
          name="device_category"
          value={form.device_category}
          onChange={(v) => update('device_category', v)}
          options={options.device_categories}
        />

        <RadioGroup
          label="What are you paying for?"
          name="merchant_category"
          value={form.merchant_category}
          onChange={(v) => update('merchant_category', v)}
          options={options.merchant_categories}
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
      </div>

      {error && (
        <div className="mt-5 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-7 flex flex-col-reverse gap-3 border-t border-navy-100 pt-5 md:flex-row md:items-center md:justify-between">
        <div className="text-xs text-navy-500">
          Card number and street address aren't asked — the model wasn't trained
          on those, and we'd never store them anyway.
        </div>
        <button type="submit" disabled={submitting} className="btn-primary">
          {submitting ? 'Scoring…' : 'Check this transaction'}
          {!submitting && <ArrowRight />}
        </button>
      </div>
    </form>
  )
}

function summariseForReceipt(form, mapping) {
  return {
    amount: form.amount,
    product: mapping?.merchant_to_product?.mapped || form.merchant_category,
    card_network: form.card_network,
    card_type: form.card_type,
    device: mapping?.device_to_fingerprint?.mapped || '—',
    email_domain: mapping?.email_to_domain?.mapped || '—',
  }
}

function Field({ label, hint, children }) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <label className="field-label">{label}</label>
        {hint && <span className="text-[11px] text-navy-400">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

function Select({ label, value, onChange, options }) {
  return (
    <Field label={label}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="field-input mt-1.5 appearance-none bg-no-repeat bg-right pr-9"
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 20 20' fill='none'><path d='M5 7L10 12L15 7' stroke='%23577' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg>\")",
          backgroundPosition: 'right 0.75rem center',
        }}
      >
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </Field>
  )
}

function RadioGroup({ label, name, value, onChange, options }) {
  return (
    <Field label={label}>
      <div className="mt-1.5 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        {options.map((o) => {
          const checked = value === o.value
          return (
            <label
              key={o.value}
              className={`flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm transition ${
                checked
                  ? 'border-navy-700 bg-navy-700 text-white shadow-sm'
                  : 'border-navy-200 bg-white text-navy-700 hover:border-navy-400'
              }`}
            >
              <input
                type="radio"
                name={name}
                value={o.value}
                checked={checked}
                onChange={() => onChange(o.value)}
                className="sr-only"
              />
              <span className={`flex h-3.5 w-3.5 items-center justify-center rounded-full border-2 ${
                checked ? 'border-white' : 'border-navy-300'
              }`}>
                {checked && <span className="h-1.5 w-1.5 rounded-full bg-white" />}
              </span>
              <span className="truncate">{o.label}</span>
            </label>
          )
        })}
      </div>
    </Field>
  )
}

function InfoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="mt-0.5 shrink-0 text-navy-500">
      <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.5" />
      <path d="M10 9V14M10 6V6.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function ArrowRight() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
      <path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
