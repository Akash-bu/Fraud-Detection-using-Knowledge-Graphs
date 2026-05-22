// All requests go through the Vite dev-server proxy in dev (see vite.config.js),
// and assume the API is reverse-proxied at /api in production.

async function json(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}

export const api = {
  options: () => fetch('/api/options').then(json),
  entities: (kind, opts = {}) => {
    const params = new URLSearchParams()
    if (opts.q) params.set('q', opts.q)
    if (opts.limit) params.set('limit', String(opts.limit))
    const qs = params.toString()
    return fetch(`/api/entities/${kind}${qs ? `?${qs}` : ''}`).then(json)
  },
  predict: (payload) =>
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),

  consumerOptions: () => fetch('/api/consumer/options').then(json),
  consumerPredict: (payload) =>
    fetch('/api/consumer/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(json),
}
