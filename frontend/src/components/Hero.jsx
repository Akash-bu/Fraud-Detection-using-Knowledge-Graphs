export default function Hero() {
  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-navy-900 via-navy-800 to-navy-700 text-white">
      <div className="absolute inset-0 opacity-[0.08]" style={{
        backgroundImage:
          'radial-gradient(circle at 20% 20%, white 1px, transparent 1px), radial-gradient(circle at 80% 60%, white 1px, transparent 1px)',
        backgroundSize: '40px 40px, 60px 60px',
      }} />
      <div className="mx-auto grid max-w-7xl gap-12 px-6 py-20 md:grid-cols-2 md:py-24">
        <div className="relative z-10">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-navy-200">
            <span className="h-1.5 w-1.5 rounded-full bg-gold-400" />
            GraphSAGE · Trained on IEEE-CIS
          </div>
          <h1 className="font-display text-4xl font-extrabold leading-tight md:text-5xl">
            Detect fraud the way it
            <span className="bg-gradient-to-r from-gold-400 to-gold-500 bg-clip-text text-transparent"> actually moves.</span>
          </h1>
          <p className="mt-5 max-w-xl text-lg text-navy-100">
            Sentinel scores transactions through a Graph Neural Network that reasons over
            the entire network of devices, cards, addresses, and email domains — surfacing
            fraud rings that flat tabular models miss.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <a href="#analyzer" className="rounded-md bg-white px-6 py-3 text-sm font-semibold text-navy-900 shadow-sm hover:bg-navy-50">
              Launch the analyzer
            </a>
            <a href="#model" className="rounded-md border border-white/25 px-6 py-3 text-sm font-semibold text-white hover:bg-white/10">
              How the model works
            </a>
          </div>
        </div>
        <div className="relative z-10 flex items-center justify-center">
          <MetricsCard />
        </div>
      </div>
    </section>
  )
}

function MetricsCard() {
  const metrics = [
    { label: 'ROC-AUC',    value: '0.88' },
    { label: 'AUC-PR',     value: '0.53' },
    { label: 'Recall @ 0.4', value: '63.7%' },
    { label: 'Entities modeled', value: '10K+' },
  ]
  return (
    <div className="w-full max-w-md rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur">
      <div className="mb-4 flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wider text-navy-200">
          Live model · v1.0
        </div>
        <div className="rounded-full bg-emerald-400/15 px-2.5 py-0.5 text-xs font-semibold text-emerald-300">
          ● Online
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {metrics.map((m) => (
          <div key={m.label} className="rounded-lg bg-navy-900/40 p-4">
            <div className="text-xs uppercase tracking-wider text-navy-300">{m.label}</div>
            <div className="mt-1 font-mono text-2xl font-semibold text-white">{m.value}</div>
          </div>
        ))}
      </div>
      <div className="mt-5 border-t border-white/10 pt-4 text-xs text-navy-200">
        Held-out test set · 23 733 transactions · 7.3% fraud prevalence
      </div>
    </div>
  )
}
