import path from 'node:path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig, loadEnv, type Plugin } from 'vite'

// Link previews need absolute image URLs. In dev, derive the origin from the incoming request (LAN IP, tunnel), so a
// link pasted into iMessage or Slack resolves og.png from wherever the page was fetched. Builds use VITE_SITE_URL.
function siteUrl(fallback: string): Plugin {
  let origin = ''
  return {
    name: 'site-url',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        const proto = (req.headers['x-forwarded-proto'] as string | undefined)?.split(',')[0] || 'http'
        const host = (req.headers['x-forwarded-host'] as string | undefined)?.split(',')[0] || req.headers.host
        if (host) origin = `${proto}://${host}`
        next()
      })
    },
    transformIndexHtml: {
      order: 'pre',
      handler(html) {
        const o = (origin || fallback).replace(/\/$/, '')
        return html.replace(/%VITE_SITE_URL%/g, o)
      },
    },
  }
}

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss(), siteUrl(loadEnv(mode, process.cwd(), 'VITE_').VITE_SITE_URL ?? '')],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  server: { port: 5173, host: true },
}))
