import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { renderMarkdown } from './markdown.js'

/**
 * Right-side drawer that streams an LLM-authored explanation of the
 * current prediction and lets the operator ask follow-up questions.
 * The model only sees the evidence packet built server-side from the
 * trained-graph statistics — its citations are dotted paths into that
 * packet so the analyst can verify each claim against the same source
 * of truth (see TranslationPanel / ExplanationPanel).
 */
export default function ExplainerChat({ open, onClose, result, form }) {
  const [messages, setMessages] = useState([])
  const [streaming, setStreaming] = useState(false)
  const [input, setInput] = useState('')
  const abortRef = useRef(null)
  const scrollRef = useRef(null)

  // Reset and auto-kick on every new (open + result) pair
  useEffect(() => {
    if (!open || !result) return
    setMessages([])
    runTurn([], { initial: true })
    return () => abortRef.current?.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, result])

  // Auto-scroll to bottom as new text streams in
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, streaming])

  async function runTurn(history, { initial = false } = {}) {
    if (!result || !form) return
    const ac = new AbortController()
    abortRef.current = ac
    setStreaming(true)
    // Add an empty assistant turn we'll stream into
    setMessages((m) => [...m, { role: 'assistant', content: '', error: null }])

    try {
      const stream = api.streamExplain(
        {
          payload: form,
          prediction: {
            verdict: result.verdict,
            fraud_probability: result.fraud_probability,
            threshold: result.threshold,
          },
          messages: history,
        },
        { signal: ac.signal },
      )

      for await (const ev of stream) {
        if (ev.type === 'text') {
          setMessages((m) => {
            const copy = m.slice()
            const last = { ...copy[copy.length - 1] }
            last.content = (last.content || '') + ev.content
            copy[copy.length - 1] = last
            return copy
          })
        } else if (ev.type === 'error') {
          setMessages((m) => {
            const copy = m.slice()
            copy[copy.length - 1] = { role: 'assistant', content: '', error: ev.message }
            return copy
          })
        } else if (ev.type === 'done') {
          // Optionally surface usage in dev — kept silent in production
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return
      setMessages((m) => {
        const copy = m.slice()
        copy[copy.length - 1] = { role: 'assistant', content: '', error: err.message }
        return copy
      })
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  async function send(text) {
    const q = (text ?? input).trim()
    if (!q || streaming) return
    setInput('')
    // The conversation we send to the server *must* end with the new user
    // turn; the assistant placeholder we add locally is for UI only.
    const next = [...messages.filter((m) => !m.error), { role: 'user', content: q }]
    setMessages((m) => [...m, { role: 'user', content: q }])
    await runTurn(next)
  }

  function stop() {
    abortRef.current?.abort()
  }

  if (!open) return null

  return (
    <>
      <div className="fixed inset-0 z-40 bg-navy-900/30 backdrop-blur-sm" onClick={onClose} />
      <aside className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-white shadow-2xl">
        <Header verdict={result?.verdict} prob={result?.fraud_probability} onClose={onClose} />
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
          {messages.map((m, i) => (
            <Bubble key={i} message={m} streaming={streaming && i === messages.length - 1} />
          ))}
          {messages.length === 0 && <Loading />}
        </div>
        <SuggestedPrompts disabled={streaming} onPick={send} hasAnswer={messages.some((m) => m.role === 'assistant' && m.content)} />
        <Composer
          value={input}
          onChange={setInput}
          onSubmit={() => send()}
          onStop={stop}
          streaming={streaming}
        />
      </aside>
    </>
  )
}

function Header({ verdict, prob, onClose }) {
  const isFraud = verdict === 'FRAUD'
  return (
    <div className="flex items-start justify-between border-b border-navy-100 px-6 py-4">
      <div>
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-navy-500">
          Sentinel Analyst
        </div>
        <div className="mt-1 flex items-center gap-2 font-display text-lg font-bold text-navy-900">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${isFraud ? 'bg-red-600' : 'bg-emerald-600'}`} />
          Investigating {verdict} verdict
          {prob != null && (
            <span className="ml-1 font-mono text-sm text-navy-500">
              ({(prob * 100).toFixed(1)}%)
            </span>
          )}
        </div>
        <div className="mt-1 text-xs text-navy-500">
          AI-authored, grounded in the trained-graph evidence packet.
        </div>
      </div>
      <button
        type="button"
        onClick={onClose}
        aria-label="Close analyst chat"
        className="rounded-md p-1.5 text-navy-500 hover:bg-navy-50 hover:text-navy-900"
      >
        <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
          <path d="M5 5L15 15M15 5L5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  )
}

function Bubble({ message, streaming }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-navy-800 px-4 py-2.5 text-sm text-white shadow-sm">
          {message.content}
        </div>
      </div>
    )
  }
  if (message.error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        <div className="mb-1 font-semibold">Analyst unavailable</div>
        <div>{message.error}</div>
        <div className="mt-2 text-[11px] text-red-600">
          Set <span className="font-mono">HF_TOKEN</span> in the backend
          environment (get one at <span className="font-mono">huggingface.co/settings/tokens</span>)
          and restart <span className="font-mono">uvicorn</span>.
        </div>
      </div>
    )
  }
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="flex-1 max-w-[90%]">
        <div className="rounded-2xl rounded-tl-md bg-navy-50 px-4 py-3 text-sm text-navy-800">
          <div
            className="prose-narrow"
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
          />
          {streaming && <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-navy-400 align-middle" />}
        </div>
      </div>
    </div>
  )
}

function Loading() {
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="flex items-center gap-2 rounded-2xl rounded-tl-md bg-navy-50 px-4 py-3 text-sm text-navy-500">
        <Dot delay="0s" /><Dot delay="0.2s" /><Dot delay="0.4s" />
        <span className="ml-2">Reviewing evidence…</span>
      </div>
    </div>
  )
}

function Dot({ delay }) {
  return (
    <span
      className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-navy-400"
      style={{ animationDelay: delay }}
    />
  )
}

function Avatar() {
  return (
    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-navy-800">
      <svg width="14" height="14" viewBox="0 0 32 32">
        <path d="M16 4 L27 8 V16 C27 23 21 27 16 29 C11 27 5 23 5 16 V8 Z"
              fill="none" stroke="#D9A441" strokeWidth="2" strokeLinejoin="round" />
        <circle cx="16" cy="15" r="2.2" fill="#D9A441" />
      </svg>
    </div>
  )
}

const SUGGESTIONS = [
  'Which entity drove this verdict most?',
  'What would change the model\'s mind?',
  'Are there comparable fraud cases in the data?',
  'How confident should I be in this score?',
]

function SuggestedPrompts({ disabled, onPick, hasAnswer }) {
  if (!hasAnswer) return null
  return (
    <div className="border-t border-navy-100 px-6 py-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-navy-500">
        Follow-ups
      </div>
      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled={disabled}
            onClick={() => onPick(s)}
            className="rounded-full border border-navy-200 bg-white px-3 py-1.5 text-xs text-navy-700 hover:border-navy-400 hover:text-navy-900 disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  )
}

function Composer({ value, onChange, onSubmit, onStop, streaming }) {
  return (
    <form
      onSubmit={(e) => { e.preventDefault(); onSubmit() }}
      className="border-t border-navy-100 bg-white px-6 py-4"
    >
      <div className="flex items-end gap-2">
        <textarea
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSubmit()
            }
          }}
          placeholder="Ask a follow-up… (Shift+Enter for newline)"
          className="field-input min-h-[44px] resize-none py-2.5"
          disabled={streaming}
        />
        {streaming ? (
          <button type="button" onClick={onStop} className="btn-ghost">
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!value.trim()} className="btn-primary px-4">
            Send
          </button>
        )}
      </div>
      <div className="mt-2 text-[11px] text-navy-400">
        Sentinel cites every claim with a dotted path into the evidence — verify against the panels below.
      </div>
    </form>
  )
}
