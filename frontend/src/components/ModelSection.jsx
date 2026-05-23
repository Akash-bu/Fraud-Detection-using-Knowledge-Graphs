export default function ModelSection() {
  const layers = [
    { n: '①', t: 'Ingestion', d: 'IEEE-CIS transactions + identity tables joined and cleaned.' },
    { n: '②', t: 'Knowledge graph', d: 'Neo4j Aura models Transactions linked to Devices, Cards, Addresses, Email Domains.' },
    { n: '③', t: 'Feature engineering', d: '23 features: amount, time, one-hot product/network/type, and per-entity history (txn count + fraud rate).' },
    { n: '④', t: 'GraphSAGE', d: '2-layer GNN, 64 hidden units. Each node aggregates messages from its 1- and 2-hop graph neighborhood.' },
    { n: '⑤', t: 'Calibration', d: 'Threshold 0.40 picked from a precision-recall sweep on the validation set.' },
  ]
  const rels = [
    { name: 'HAS_DEVICE', target: 'Device' },
    { name: 'USED_CARD',  target: 'Card' },
    { name: 'BILLED_TO',  target: 'Address' },
    { name: 'SENT_FROM',  target: 'Email Domain' },
  ]
  return (
    <section id="model" className="bg-white">
      <div className="mx-auto max-w-7xl px-6 py-20">
        <div className="grid gap-12 lg:grid-cols-5">
          <div className="lg:col-span-2">
            <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-navy-500">
              Under the hood
            </div>
            <h2 className="font-display text-3xl font-extrabold text-navy-900 md:text-4xl">
              A graph-native pipeline.
            </h2>
            <p className="mt-5 text-navy-600">
              Sentinel models payments as a knowledge graph and trains a GraphSAGE GNN over it.
              The network learns fraud signatures that only emerge from relationships — a single
              device shared across dozens of cards, a fresh email domain billing many addresses,
              consecutive transaction IDs from the same hardware.
            </p>
            <div className="mt-8 grid grid-cols-2 gap-4">
              <Stat label="Entity types" value="5" />
              <Stat label="Relationship types" value="4" />
              <Stat label="Training nodes" value="128K" />
              <Stat label="Training edges" value="949K" />
            </div>
          </div>

          <div className="lg:col-span-3">
            <div className="card p-6 md:p-8">
              <div className="mb-5 text-xs font-semibold uppercase tracking-wider text-navy-500">
                Knowledge graph schema
              </div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {rels.map((r) => (
                  <div key={r.name} className="flex items-center gap-3 rounded-md border border-navy-100 px-3 py-2.5">
                    <span className="rounded bg-navy-800 px-2 py-1 font-mono text-[11px] font-semibold text-white">
                      Transaction
                    </span>
                    <span className="font-mono text-[11px] text-navy-500">─[{r.name}]→</span>
                    <span className="rounded bg-gold-400/15 px-2 py-1 font-mono text-[11px] font-semibold text-gold-500">
                      {r.target}
                    </span>
                  </div>
                ))}
              </div>

              <div className="my-6 h-px bg-navy-100" />

              <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-navy-500">
                Pipeline
              </div>
              <ol className="space-y-3">
                {layers.map((l) => (
                  <li key={l.t} className="flex gap-3">
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-navy-800 font-mono text-xs text-gold-400">
                      {l.n}
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-navy-900">{l.t}</div>
                      <div className="text-sm text-navy-600">{l.d}</div>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-navy-100 bg-navy-50/40 p-4">
      <div className="text-[11px] uppercase tracking-wider text-navy-500">{label}</div>
      <div className="mt-1 font-mono text-2xl font-bold text-navy-900">{value}</div>
    </div>
  )
}
