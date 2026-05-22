export default function Header() {
  return (
    <header className="border-b border-navy-100 bg-white/80 backdrop-blur sticky top-0 z-30">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          <Logo />
          <div>
            <div className="font-display text-xl font-bold leading-none text-navy-900">
              Sentinel
            </div>
            <div className="text-[11px] uppercase tracking-[0.18em] text-navy-500">
              Graph Risk Intelligence
            </div>
          </div>
        </div>
        <nav className="hidden gap-8 text-sm font-medium text-navy-600 md:flex">
          <a href="#analyzer" className="hover:text-navy-900">Analyzer</a>
          <a href="#model" className="hover:text-navy-900">Model</a>
          <a href="#about" className="hover:text-navy-900">About</a>
        </nav>
        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-2 rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700 md:inline-flex">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Model online
          </span>
          <a
            href="https://github.com/akash-bu/fraud_detection_neo4j"
            target="_blank"
            rel="noreferrer"
            className="btn-ghost"
          >
            View source
          </a>
        </div>
      </div>
    </header>
  )
}

function Logo() {
  return (
    <svg width="36" height="36" viewBox="0 0 32 32" className="shrink-0">
      <rect width="32" height="32" rx="6" fill="#0F2440" />
      <path
        d="M16 5 L26 9 V16 C26 22 21 26 16 28 C11 26 6 22 6 16 V9 Z"
        fill="none"
        stroke="#D9A441"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <circle cx="16" cy="15" r="2.2" fill="#D9A441" />
      <path
        d="M11 21 L16 16 L21 21"
        stroke="#D9A441"
        strokeWidth="1.8"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  )
}
