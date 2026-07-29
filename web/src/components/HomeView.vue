<script setup>
import { startRandom, openEditor } from "../composables/useGame.js";
</script>

<template>
  <section class="home">
    <h1>Bolgrot</h1>
    <p class="lead">
      Un jeu de survie tour par tour : téléportez-vous pour éteindre les flammes
      sans qu'aucune ne vous touche. Une IA entraînée peut vous souffler le
      meilleur coup, ou jouer la partie à votre place.
    </p>
    <div class="home-actions">
      <button class="primary big" @click="startRandom">
        S'entraîner (partie aléatoire)
      </button>
      <button class="big" @click="openEditor">
        Créer ma partie (dessiner les 6 vagues)
      </button>
    </div>
    <p class="disclaimer">
      L'IA est un assistant très fort, mais la victoire n'est pas garantie à
      chaque partie. Sur un niveau perdu, « Réessayer plus fort » la relance avec
      plus de réflexion (elle finit par gagner en cherchant plus profond).
    </p>
    <div class="disclaimer">
      <b>Deux moteurs d'IA</b> — « Rapide » (léger) ou « Fort » (plus précis mais
      nettement plus lent). Temps de calcul par coup et taux de victoire de l'IA
      sur le jeu complet, selon le budget de simulations :
      <table class="perf">
        <thead>
          <tr><th>Budget</th><th>Rapide</th><th>Fort</th></tr>
        </thead>
        <tbody>
          <tr><td>300</td><td>50 % · ~0,3 s</td><td>77 % · ~1,2 s</td></tr>
          <tr><td>1000</td><td>73 % · ~0,9 s</td><td>94 % · ~4 s</td></tr>
          <tr><td>8000</td><td>~90 % · ~8 s</td><td>~95 % · ~34 s</td></tr>
        </tbody>
      </table>
      <span class="perf-note">Mesuré en prod (2 vCPU). Le « Fort » cherche mieux
      par simulation mais chaque coup coûte ~4× plus de temps.</span>
    </div>
    <div class="download">
      <div class="download-icon">🖥️</div>
      <h2>Téléchargement</h2>
      <p>
        Récupérez la version bureau : l'IA embarquée en local (indices &amp;
        autoplay hors-ligne, calcul plus rapide qu'en ligne) et un mode où vous
        placez vous-même les flammes.
      </p>
      <a class="download-btn"
         href="https://github.com/Tryllogy/Bolgrot-Solver/releases/latest/download/bolgrot-windows.zip">
        ⬇ Télécharger pour Windows (.zip)
      </a>
      <span class="download-hint">
        ≈ 170 Mo — décompressez, puis lancez <code>bolgrot.exe</code>.
      </span>
      <a class="download-src"
         href="https://github.com/Tryllogy/Bolgrot-Solver/releases/latest"
         target="_blank" rel="noopener noreferrer">
        Versions Linux / macOS &amp; code source sur GitHub
      </a>
    </div>
  </section>
</template>

<style scoped>
.home {
  padding-top: 24px;
}
.home h1 {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0 0 12px;
}
.lead {
  color: var(--text);
  font-size: 1.05rem;
  line-height: 1.6;
  max-width: 560px;
  margin: 0 auto 32px;
}
.home-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  margin-bottom: 40px;
}
.disclaimer {
  color: var(--text);
  font-size: 0.8125rem;
  line-height: 1.6;
  max-width: 560px;
  margin: 0 auto;
}
.perf {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0 6px;
  font-variant-numeric: tabular-nums;
}
.perf th,
.perf td {
  border: 1px solid var(--border);
  padding: 5px 10px;
  text-align: center;
}
.perf th {
  color: var(--text-strong);
  font-weight: 600;
}
.perf-note {
  display: block;
  font-size: 0.75rem;
}
.download {
  max-width: 560px;
  margin: 40px auto 12px;
  padding: 28px 24px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--text-strong) 6%, transparent);
  text-align: center;
}
.download-icon {
  font-size: 2.4rem;
  line-height: 1;
  margin-bottom: 8px;
}
.download h2 {
  font-size: 1.35rem;
  font-weight: 700;
  margin: 0 0 8px;
  color: var(--text-strong);
}
.download p {
  color: var(--text);
  font-size: 0.9rem;
  line-height: 1.6;
  margin: 0 auto 18px;
  max-width: 440px;
}
.download-btn {
  display: inline-block;
  padding: 12px 26px;
  border-radius: 10px;
  background: var(--accent, #3a7ca5);
  color: var(--accent-contrast);
  font-size: 1.05rem;
  font-weight: 700;
  text-decoration: none;
}
.download-btn:hover {
  filter: brightness(1.08);
}
.download-hint {
  display: block;
  margin-top: 10px;
  font-size: 0.75rem;
  color: var(--text);
}
.download-hint code {
  font-family: monospace;
  background: color-mix(in srgb, var(--text-strong) 12%, transparent);
  padding: 1px 5px;
  border-radius: 4px;
}
.download-src {
  display: block;
  margin-top: 14px;
  color: var(--text);
  font-size: 0.8125rem;
  text-decoration: underline;
}
.download-src:hover {
  color: var(--text-strong);
}
</style>
