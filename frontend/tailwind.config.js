/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          50:  '#F2F6FB',
          100: '#E1EAF5',
          200: '#B9CCE5',
          300: '#8AAACF',
          400: '#5783B4',
          500: '#2F5E94',
          600: '#1F4478',
          700: '#163358',
          800: '#0F2440',
          900: '#0A1A30',
          950: '#06111F',
        },
        gold: {
          400: '#D9A441',
          500: '#B8862C',
        },
        risk: {
          low:    '#15803D',
          medium: '#B45309',
          high:   '#B91C1C',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(15, 36, 64, 0.04), 0 8px 24px rgba(15, 36, 64, 0.06)',
        glow: '0 0 0 4px rgba(47, 94, 148, 0.18)',
      },
    },
  },
  plugins: [],
}
