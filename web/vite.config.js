import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Inline le CSS buildé dans le <head> et supprime le <link rel="stylesheet">.
// Le CSS est petit (~3 ko gzip) mais chargé en externe il bloque le rendu
// initial (flag PageSpeed « Requêtes de blocage du rendu »). Inliné, il part
// avec le HTML : zéro requête bloquante, meilleur FCP/LCP. Le JS reste externe
// (module = différé, non bloquant, et trop gros pour être inliné).
function inlineCss() {
  return {
    name: 'inline-css',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      const html = Object.values(bundle).find(
        (f) => f.type === 'asset' && f.fileName === 'index.html'
      )
      if (!html) return
      const cssFiles = Object.values(bundle).filter(
        (f) => f.type === 'asset' && f.fileName.endsWith('.css')
      )
      let source = html.source
      for (const css of cssFiles) {
        const base = css.fileName.split('/').pop()
        const link = new RegExp(
          `<link[^>]*href="[^"]*${base}"[^>]*>`
        )
        source = source.replace(link, `<style>${css.source}</style>`)
        delete bundle[css.fileName]
      }
      html.source = source
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), inlineCss()],
})
