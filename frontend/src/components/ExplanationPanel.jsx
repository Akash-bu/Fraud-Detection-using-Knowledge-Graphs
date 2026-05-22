export default function ExplanationPanel({ result }) {
  if (!result) return null
  const { signals, amount } = result.explanation

  return (
    <div className="card p-6 md:p-8">
      <div className="mb-5 flex items-baseline justify-between">
        <h3 className="font-display text-lg font-bold text-navy-900">
          Why the model said that
        </h3>
        <span className="text-xs text-navy-500">Graph neighborhood signals</span>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {signals.map((s) => (
          <SignalRow key={s.label} signal={s} />
        ))}
        <AmountRow amount={amount} />
      </div>

      <div className="mt-6 rounded-md bg-navy-50/60 px-4 py-3 text-xs leading-relaxed text-navy-600">
        <span className="font-semibold text-navy-700">How to read this:</span>{' '}
        Each row shows one of the four entities this transaction connects to in the
        knowledge graph, along with its historical fraud rate from the training set.
        The GNN propagates these patterns through 2 layers of message passing before
        producing the final score.
      </div>
    </div>
  )
}

function SignalRow({ signal }) {
  const colors = {
    high:    { bg: 'bg-red-50',     bar: 'bg-red-600',     text: 'text-red-700',     ring: 'ring-red-200' },
    medium:  { bg: 'bg-amber-50',   bar: 'bg-amber-500',   text: 'text-amber-700',   ring: 'ring-amber-200' },
    low:     { bg: 'bg-emerald-50', bar: 'bg-emerald-500', text: 'text-emerald-700', ring: 'ring-emerald-200' },
    neutral: { bg: 'bg-navy-50',    bar: 'bg-navy-300',    text: 'text-navy-600',    ring: 'ring-navy-200' },
  }[signal.severity]

  const pct = signal.fraud_rate != null
    ? Math.max(2, Math.round(signal.fraud_rate * 100))
    : 0

  return (
    <div className={`rounded-lg border border-transparent p-4 ring-1 ${colors.bg} ${colors.ring}`}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-navy-700">
          {signal.label}
        </div>
        <div className={`font-mono text-xs font-semibold ${colors.text}`}>
          {signal.fraud_rate != null ? `${(signal.fraud_rate * 100).toFixed(1)}% fraud` : 'unseen'}
        </div>
      </div>
      <div className="mt-1 truncate font-mono text-sm text-navy-900">
        {signal.value || '—'}
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white">
        <div className={`h-full ${colors.bar}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-2 text-[11px] leading-relaxed text-navy-600">
        {signal.note}
      </div>
    </div>
  )
}

function AmountRow({ amount }) {
  const flagged = !!amount.note
  return (
    <div className={`rounded-lg border border-transparent p-4 ring-1 ${flagged ? 'bg-amber-50 ring-amber-200' : 'bg-navy-50 ring-navy-200'} md:col-span-2`}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-xs font-semibold uppercase tracking-wider text-navy-700">
          Amount
        </div>
        <div className="font-mono text-xs text-navy-600">
          pop. mean ${amount.population_mean.toFixed(2)} · p99 ${amount.population_p99.toFixed(0)}
        </div>
      </div>
      <div className="mt-1 font-mono text-sm text-navy-900">
        ${amount.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </div>
      {amount.note && (
        <div className="mt-2 text-[11px] leading-relaxed text-amber-800">{amount.note}</div>
      )}
    </div>
  )
}
