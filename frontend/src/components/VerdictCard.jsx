export default function VerdictCard({ result, loading, form, onAskAnalyst }) {
  if (loading) {
    return (
      <div className="card flex h-full flex-col items-center justify-center gap-3 p-10 text-center">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-navy-200 border-t-navy-700" />
        <div className="text-sm text-navy-500">Running GraphSAGE forward pass…</div>
      </div>
    )
  }

  if (!result) {
    return <EmptyState />
  }

  const isFraud = result.verdict === 'FRAUD'
  const pct = (result.fraud_probability * 100).toFixed(1)
  const accent = isFraud
    ? { bar: 'bg-red-600', soft: 'bg-red-50', ring: 'text-red-700', chip: 'bg-red-600' }
    : { bar: 'bg-emerald-600', soft: 'bg-emerald-50', ring: 'text-emerald-700', chip: 'bg-emerald-600' }

  return (
    <div className="card overflow-hidden">
      <div className={`${accent.soft} px-6 py-5`}>
        <div className="flex items-start justify-between">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-navy-500">
              Verdict
            </div>
            <div className={`mt-1 flex items-center gap-3 font-display text-3xl font-extrabold ${accent.ring}`}>
              <span className={`inline-block h-3 w-3 rounded-full ${accent.chip}`} />
              {result.verdict}
            </div>
            <div className="mt-1 text-sm text-navy-600">
              Decision threshold {(result.threshold * 100).toFixed(0)}%
            </div>
          </div>
          <div className="text-right">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-navy-500">
              Fraud probability
            </div>
            <div className={`font-mono text-4xl font-bold ${accent.ring}`}>
              {pct}<span className="text-lg">%</span>
            </div>
          </div>
        </div>

        <div className="mt-5">
          <div className="h-2 w-full overflow-hidden rounded-full bg-white/70">
            <div className={`h-full ${accent.bar} transition-all`} style={{ width: `${Math.max(1, pct)}%` }} />
          </div>
          <div className="mt-1 flex justify-between font-mono text-[10px] text-navy-500">
            <span>0%</span>
            <span>{(result.threshold * 100).toFixed(0)}% threshold</span>
            <span>100%</span>
          </div>
        </div>
      </div>

      <div className="border-t border-navy-100 px-6 py-4 text-sm leading-relaxed text-navy-700">
        {result.explanation?.summary}
      </div>

      {onAskAnalyst && (
        <div className="border-t border-navy-100 px-6 py-3">
          <button
            type="button"
            onClick={onAskAnalyst}
            className="group inline-flex w-full items-center justify-between rounded-md border border-navy-200 bg-white px-4 py-2.5 text-sm font-semibold text-navy-800 transition hover:border-navy-400 hover:bg-navy-50"
          >
            <span className="flex items-center gap-2">
              <SparkleIcon />
              Ask the Sentinel analyst
            </span>
            <span className="text-xs text-navy-500 group-hover:text-navy-700">
              AI-authored, evidence-grounded
            </span>
          </button>
        </div>
      )}

      <Receipt form={form} />
    </div>
  )
}

function Receipt({ form }) {
  if (!form) return null
  const rows = [
    ['Amount', `$${Number(form.amount).toFixed(2)}`],
    ['Product', form.product],
    ['Card', `${form.card_network} · ${form.card_type}`],
    ['Device', form.device || '—'],
    ['Email', form.email_domain || '—'],
  ]
  return (
    <div className="border-t border-navy-100 bg-navy-50/40 px-6 py-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-navy-500">
        Scored transaction
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-navy-500">{k}</dt>
            <dd className="truncate font-mono text-navy-800">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="card flex h-full flex-col items-center justify-center gap-3 p-10 text-center">
      <ShieldIcon />
      <h3 className="font-display text-lg font-bold text-navy-800">Awaiting transaction</h3>
      <p className="max-w-xs text-sm text-navy-500">
        Fill in the form on the left and click <span className="font-semibold text-navy-700">Analyze</span>{' '}
        to score against the live GraphSAGE model.
      </p>
    </div>
  )
}

function ShieldIcon() {
  return (
    <svg width="44" height="44" viewBox="0 0 32 32" fill="none">
      <path
        d="M16 4 L27 8 V16 C27 23 21 27 16 29 C11 27 5 23 5 16 V8 Z"
        fill="#0F2440"
        stroke="#D9A441"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <path d="M11 16 L15 20 L22 12" stroke="#D9A441" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function SparkleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none" className="text-gold-500">
      <path d="M10 2 L11.5 7 L16 8.5 L11.5 10 L10 15 L8.5 10 L4 8.5 L8.5 7 Z" fill="currentColor" />
      <circle cx="16" cy="4" r="1" fill="currentColor" />
      <circle cx="4" cy="15" r="1" fill="currentColor" />
    </svg>
  )
}
