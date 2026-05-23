/**
 * Tiny markdown renderer for the analyst chat. Supports the subset the
 * prompt actually asks the model to use: paragraphs, **bold**, *italic*,
 * `code`, bulleted lists, and the inline citation pattern
 * `[evidence.path=value]` which we render as a styled chip.
 *
 * We avoid pulling in `react-markdown` to keep the bundle small — this
 * renderer is intentionally narrow and HTML-escapes everything before
 * applying replacements, so it's safe to feed straight to
 * dangerouslySetInnerHTML.
 */

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }
const escape = (s) => s.replace(/[&<>"']/g, (c) => ESC[c])

// Match `[path.with.dots=value]` — value is anything up to the closing bracket
const CITATION_RE = /\[([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)(?:=([^\]]+))?\]/gi
const INLINE_CODE_RE = /`([^`]+)`/g
const BOLD_RE = /\*\*([^*]+)\*\*/g
const ITALIC_RE = /(^|\s)\*([^*]+)\*/g

export function renderMarkdown(src = '') {
  const lines = src.replace(/\r\n/g, '\n').split('\n')
  const out = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    if (!line.trim()) {
      i++
      continue
    }

    // Bulleted list block
    if (/^\s*[-*]\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''))
        i++
      }
      out.push('<ul>' + items.map((it) => '<li>' + inline(it) + '</li>').join('') + '</ul>')
      continue
    }

    // Paragraph: collect consecutive non-blank, non-list lines
    const para = []
    while (i < lines.length && lines[i].trim() && !/^\s*[-*]\s+/.test(lines[i])) {
      para.push(lines[i])
      i++
    }
    out.push('<p>' + inline(para.join(' ')) + '</p>')
  }

  return out.join('')
}

function inline(s) {
  let out = escape(s)
  out = out.replace(INLINE_CODE_RE, (_, code) => `<code>${code}</code>`)
  out = out.replace(BOLD_RE, '<strong>$1</strong>')
  out = out.replace(ITALIC_RE, '$1<em>$2</em>')
  out = out.replace(CITATION_RE, (_, path, value) => {
    const v = value != null ? `<span class="cite-val">${value}</span>` : ''
    return `<span class="cite" title="${path}${value ? '=' + escape(value) : ''}">`
         + `<span class="cite-path">${path}</span>${v}</span>`
  })
  return out
}
