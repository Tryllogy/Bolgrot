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
    <p class="disclaimer">
      <b>
        Attention : augmenter les sims augmente le temps de calcul (~1 s / 1000
        sims).
      </b>
    </p>
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
</style>
