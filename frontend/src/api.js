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

  /**
   * Stream the LLM explainer. Yields events of shape
   *   { type: 'text', content: '...' }   |   { type: 'done', usage: {...} }
   *   { type: 'error', message: '...' }
   * Pass an AbortSignal to cancel mid-stream.
   */
  async *streamExplain({ payload, prediction, messages = [] }, { signal } = {}) {
    const res = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ payload, prediction, messages }),
      signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`Explainer ${res.status}: ${text || res.statusText}`)
    }
    if (!res.body) throw new Error('No response body to stream')

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE framing: events are separated by double newline; each event is
      // one or more "data: <payload>\n" lines.
      let idx
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const chunk = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            try {
              yield JSON.parse(line.slice(6))
            } catch {
              /* ignore malformed event */
            }
          }
        }
      }
    }
  },
}
