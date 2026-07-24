<script setup>
import {
  editorWave,
  editorSpawns,
  editorContext,
  autoplay,
  clearEditor,
  playCustom,
  submitSpawn,
  stopAutoplay,
} from "../composables/useGame.js";
</script>

<template>
  <div class="editor-bar">
    <span>
      <strong>Vague {{ editorWave }}/6</strong> — cliquez pour poser / retirer un
      point de spawn (violet) — <strong>{{ editorSpawns.length }}/6</strong>
    </span>
    <button :disabled="!editorSpawns.length" @click="clearEditor">
      Effacer
    </button>
    <button
      class="primary"
      :disabled="!editorSpawns.length"
      @click="editorContext === 'create' ? playCustom() : submitSpawn()">
      {{ editorContext === 'create' ? 'Jouer' : 'Valider la vague' }}
    </button>
    <span v-if="autoplay" class="ai-note">
      L'IA reprendra après validation
      <button class="ghost" @click="stopAutoplay">Arrêter l'IA</button>
    </span>
  </div>
</template>

<style scoped>
.editor-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 24px;
  font-size: 14px;
  color: var(--text);
}
.editor-bar strong {
  color: var(--text-strong);
}
.ai-note {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--accent);
}
</style>
