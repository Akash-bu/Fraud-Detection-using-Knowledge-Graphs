import { useEffect, useMemo, useRef, useState } from 'react'

/**
 * Searchable dropdown over a (potentially large) catalog of entities.
 * Renders fraud-rate + volume hints next to each row so an operator
 * can intentionally pick high-risk vs low-risk entities for the demo.
 */
export default function EntitySelect({
  label,
  placeholder = 'Search…',
  options,           // [{ value, txn_count, fraud_rate }]
  value,
  onChange,
  hint,
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const wrapRef = useRef(null)

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    const list = needle
      ? options.filter((o) => o.value.toLowerCase().includes(needle))
      : options
    return list.slice(0, 200)
  }, [options, query])

  useEffect(() => {
    function onClick(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [])

  const selected = options.find((o) => o.value === value)

  return (
    <div ref={wrapRef} className="relative">
      <div className="mb-1.5 flex items-center justify-between">
        <label className="field-label">{label}</label>
        {hint && <span className="text-[11px] text-navy-400">{hint}</span>}
      </div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="field-input flex items-center justify-between text-left"
      >
        <span className={value ? 'text-navy-900' : 'text-navy-400'}>
          {value ? truncate(value) : placeholder}
        </span>
        <div className="flex items-center gap-2">
          {selected && (
            <RiskPill fraudRate={selected.fraud_rate} />
          )}
          <Chevron open={open} />
        </div>
      </button>

      {open && (
        <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border border-navy-200 bg-white shadow-card">
          <div className="border-b border-navy-100 p-2">
            <input
              autoFocus
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type to filter…"
              className="w-full rounded border border-navy-100 px-2.5 py-1.5 text-sm focus:border-navy-400 focus:outline-none"
            />
          </div>
          <ul className="combobox-list max-h-72 overflow-y-auto">
            <li>
              <button
                type="button"
                onClick={() => { onChange(''); setOpen(false); setQuery('') }}
                className="flex w-full items-center justify-between border-b border-navy-50 px-3 py-2 text-left text-sm text-navy-500 hover:bg-navy-50"
              >
                <span>— None (unseen entity) —</span>
              </button>
            </li>
            {filtered.length === 0 && (
              <li className="px-3 py-4 text-center text-sm text-navy-400">No matches</li>
            )}
            {filtered.map((o) => (
              <li key={o.value}>
                <button
                  type="button"
                  onClick={() => { onChange(o.value); setOpen(false); setQuery('') }}
                  className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-navy-50"
                >
                  <span className="truncate font-mono text-xs text-navy-700">{truncate(o.value, 40)}</span>
                  <span className="flex shrink-0 items-center gap-2">
                    <span className="text-[11px] text-navy-400">{o.txn_count}×</span>
                    <RiskPill fraudRate={o.fraud_rate} />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function truncate(s, n = 28) {
  if (!s) return s
  return s.length > n ? s.slice(0, n - 1) + '…' : s
}

function RiskPill({ fraudRate }) {
  if (fraudRate == null) return null
  const pct = (fraudRate * 100).toFixed(0)
  let cls = 'bg-emerald-50 text-emerald-700'
  if (fraudRate >= 0.30) cls = 'bg-red-50 text-red-700'
  else if (fraudRate >= 0.10) cls = 'bg-amber-50 text-amber-700'
  return (
    <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold ${cls}`}>
      {pct}%
    </span>
  )
}

function Chevron({ open }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 20 20"
      fill="none"
      className={`text-navy-400 transition ${open ? 'rotate-180' : ''}`}
    >
      <path d="M5 7L10 12L15 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
