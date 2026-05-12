/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        bg: 'var(--bg-page)',
        elev: 'var(--bg-elev)',
        sunk: 'var(--bg-sunk)',
        ink: 'var(--ink)',
        dim: 'var(--ink-dim)',
        mute: 'var(--ink-mute)',
        rule: 'var(--rule)',
        'rule-soft': 'var(--rule-soft)',
        accent: 'var(--accent)',
        'accent-dim': 'var(--accent-dim)',
        ok: 'var(--ok)',
        warn: 'var(--warn)',
        err: 'var(--err)',
      },
      letterSpacing: {
        tightish: '-0.01em',
        track: '0.14em',
      },
    },
  },
  plugins: [],
}
