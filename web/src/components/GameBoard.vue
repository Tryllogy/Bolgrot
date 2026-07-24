<script setup>
import {
  setCanvas,
  canvasW,
  canvasH,
  onCanvasClick,
  onCanvasMove,
  onCanvasLeave,
  toast,
  screen,
  view,
  customMode,
  newGame,
  solveLevel,
  solving,
  solveBudget,
} from "../composables/useGame.js";
</script>

<template>
  <div class="board">
    <canvas
      @click="onCanvasClick"
      @mousemove="onCanvasMove"
      @mouseleave="onCanvasLeave"
      :ref="setCanvas"
      :width="canvasW"
      :height="canvasH"></canvas>
    <Transition name="fade">
      <div v-if="toast && screen === 'play'" class="toast">{{ toast }}</div>
    </Transition>
    <div v-if="screen === 'play' && view && view.done" class="overlay">
      <div class="modal">
        <h2>{{ view.won ? "Victoire !" : "Défaite" }}</h2>
        <button class="primary" @click="newGame()">
          {{ customMode ? "Nouvelle partie perso" : "Nouvelle partie" }}
        </button>
        <button v-if="!view.won && !customMode" @click="solveLevel" :disabled="solving">
          {{ solving ? `L'IA cherche à ${solveBudget} sims…` : "Réessayer plus fort" }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.board {
  position: relative;
  display: block;
  line-height: 0;
}
canvas {
  display: block;
  width: 100%;
  height: auto;
}
.toast {
  position: absolute;
  top: 14px;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--text-strong);
  font-size: 13px;
  font-weight: 500;
  line-height: 1;
  white-space: nowrap;
  box-shadow: var(--shadow);
  pointer-events: none;
  z-index: 5;
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(8, 9, 12, 0.68);
  backdrop-filter: blur(3px);
}
.modal {
  display: flex;
  flex-direction: column;
  gap: 14px;
  align-items: center;
  padding: 28px 32px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  line-height: 1.4;
}
.modal h2 {
  font-size: 1.375rem;
  font-weight: 600;
}
</style>
