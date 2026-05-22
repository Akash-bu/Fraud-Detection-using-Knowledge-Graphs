/**
 * Show the user how we translated their plain-language inputs into the
 * features the model was trained on. Trust is built by transparency, not by
 * pretending the dataset matches their real card and billing address.
 */
export default function TranslationPanel({ mapping }) {
  if (!mapping) return null
  const rows = [
    {
      label: 'Email',
      input: mapping.email_to_domain.input || '—',
      mapped: mapping.email_to_domain.mapped
        ? `domain: ${mapping.email_to_domain.mapped}`
        : 'not provided',
      note: 'Only the domain leaves your screen — never the full address.',
    },
    {
      label: 'Device',
      input: mapping.device_to_fingerprint.input,
      mapped: mapping.device_to_fingerprint.mapped
        ? `fingerprint: ${mapping.device_to_fingerprint.mapped}`
        : 'unseen (no graph link)',
      note: 'Mapped to the representative device fingerprint the model knows.',
    },
    {
      label: 'Merchant',
      input: mapping.merchant_to_product.input,
      mapped: `product code: ${mapping.merchant_to_product.mapped}`,
      note: 'Vesta product categories (C, H, R, S) are anonymized.',
    },
    {
      label: 'Card number',
      input: 'not asked',
      mapped: 'unmapped',
      note: 'The training data has internal Vesta card IDs, not real card numbers.',
    },
    {
      label: 'Street address',
      input: 'not asked',
      mapped: 'unmapped',
      note: 'The training data has anonymized region codes, not real addresses.',
    },
  ]
  return (
    <div className="card p-6 md:p-7">
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="font-display text-lg font-bold text-navy-900">
          How we read your transaction
        </h3>
        <span className="text-xs text-navy-500">Consumer → model translation</span>
      </div>
      <div className="space-y-2">
        {rows.map((r) => (
          <div key={r.label} className="grid grid-cols-12 gap-3 rounded-md border border-navy-100 px-4 py-3 text-sm">
            <div className="col-span-12 text-[11px] font-semibold uppercase tracking-wider text-navy-500 md:col-span-2">
              {r.label}
            </div>
            <div className="col-span-12 md:col-span-3">
              <div className="text-[11px] text-navy-400">You said</div>
              <div className="truncate font-mono text-navy-800">{r.input}</div>
            </div>
            <div className="col-span-12 md:col-span-3">
              <div className="text-[11px] text-navy-400">Model sees</div>
              <div className="truncate font-mono text-navy-800">{r.mapped}</div>
            </div>
            <div className="col-span-12 text-xs leading-relaxed text-navy-500 md:col-span-4">
              {r.note}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-5 rounded-md bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-800">
        <span className="font-semibold">Honest caveat:</span> the IEEE-CIS dataset
        is anonymized, so the model has never seen your specific card or address.
        Predictions here are based on the patterns of <em>similar transactions</em>{' '}
        — useful for understanding how the GNN reasons, not a substitute for your
        bank's own fraud system.
      </div>
    </div>
  )
}
