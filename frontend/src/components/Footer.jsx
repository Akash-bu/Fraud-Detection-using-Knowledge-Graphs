export default function Footer() {
  return (
    <footer id="about" className="border-t border-navy-100 bg-navy-900 text-navy-200">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid gap-10 md:grid-cols-3">
          <div>
            <div className="font-display text-xl font-bold text-white">Sentinel</div>
            <p className="mt-3 text-sm leading-relaxed text-navy-300">
              A research-grade fraud detection stack combining Neo4j knowledge graphs
              with PyTorch Geometric GNNs. Trained on the IEEE-CIS Fraud Detection
              dataset.
            </p>
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-navy-400">
              Model
            </div>
            <ul className="mt-3 space-y-2 text-sm">
              <li>Architecture · GraphSAGE (2-layer, 64 hidden)</li>
              <li>Features · 23 per node, 4 edge types</li>
              <li>Validation · AUC-PR 0.53, ROC-AUC 0.88</li>
            </ul>
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-navy-400">
              Stack
            </div>
            <ul className="mt-3 space-y-2 text-sm">
              <li>Graph store · Neo4j Aura</li>
              <li>Model · PyTorch Geometric</li>
              <li>API · FastAPI · UI · React + Vite</li>
            </ul>
          </div>
        </div>
        <div className="mt-10 border-t border-navy-800 pt-6 text-xs text-navy-400">
          © Sentinel demo · Predictions are illustrative — not financial advice.
        </div>
      </div>
    </footer>
  )
}
