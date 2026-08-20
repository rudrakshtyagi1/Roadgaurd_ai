/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        safe: '#22c55e',
        caution: '#eab308',
        high: '#f97316',
        critical: '#ef4444',
        panel: '#0f172a',
        surface: '#1e293b',
        border: '#334155',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
