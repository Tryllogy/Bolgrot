<script setup>
import {
  view,
  selectedSpell,
  selectSpell,
  playAction,
  descLines,
  sims,
  netEngine,
  askHint,
  hintLoading,
  autoplay,
  toggleAutoplay,
  playAiMove,
  turnClock,
  turnLow,
} from "../composables/useGame.js";
</script>

<template>
  <div v-if="view" class="hud">
    <div class="stat">
      <span class="stat-label">PV</span>
      <span class="stat-value hp">{{ view.player.hp }}</span>
    </div>
    <div class="stat">
      <span class="stat-label">PA</span>
      <span class="stat-value pa">{{ view.player.pa }}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Vague</span>
      <span class="stat-value">{{ view.waves.spawned }}/{{ view.waves.total }}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Flammes</span>
      <span class="stat-value">{{ view.flames.length }}</span>
    </div>
  </div>

  <div v-if="view && !view.done" class="controls">
    <div class="spells">
      <div class="spell-slot" v-for="(sp, i) in view.spells" :key="i">
        <button
          class="spell"
          :class="{ active: selectedSpell === i }"
          :disabled="!sp.castable"
          @click="selectSpell(i)">
          <img :src="`/sprites/${sp.sprite}`" :alt="sp.name" />
          <span>{{ sp.name }}</span>
        </button>
        <span class="tooltip">
          <strong>{{ sp.name }}</strong>
          <span class="tt-cost">
            Coût : {{ sp.cost }} PA<template v-if="sp.uses_left < 90"> · {{ sp.uses_left }} usage(s)</template>
          </span>
          <span v-for="line in descLines(sp.description)" :key="line" class="tt-line">{{ line }}.</span>
          <span v-if="sp.effects.length" class="tt-effects">{{ sp.effects.join(" · ") }}</span>
        </span>
      </div>
      <div class="end-turn-wrap">
        <button class="end-turn" @click="playAction(16)">Fin de tour</button>
        <span class="turn-timer" :class="{ low: turnLow }">{{ turnClock }}</span>
      </div>
    </div>
    <div class="actions">
      <label class="budget">
        Moteur
        <select v-model="netEngine">
          <option value="az">Rapide</option>
          <option value="cnn">Fort</option>
        </select>
      </label>
      <label class="budget">
        Budget IA
        <select v-model.number="sims">
          <option :value="300">Rapide (300)</option>
          <option :value="1000">Normal (1000)</option>
          <option :value="8000">Fort (8000)</option>
        </select>
      </label>
      <button class="primary" @click="askHint" :disabled="hintLoading || autoplay">
        {{ hintLoading ? "Réflexion…" : "Indice" }}
      </button>
      <button @click="playAiMove" :disabled="hintLoading || autoplay">
        ⏭ IA : 1 coup
      </button>
      <button @click="toggleAutoplay">
        {{ autoplay ? "⏹ Stop" : "▶ Autoplay" }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.hud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-bottom: 28px;
}
.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  min-width: 78px;
  padding: 10px 18px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.stat-label {
  font-size: 10px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text);
}
.stat-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-strong);
  font-variant-numeric: tabular-nums;
}
.stat-value.hp {
  color: #e0554b;
}
.stat-value.pa {
  color: #3399ff;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: 18px;
  align-items: center;
  margin-bottom: 28px;
}
.spells {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
}
.end-turn-wrap {
  margin-left: 104px; /* espacement d'une largeur de sort */
  align-self: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.turn-timer {
  font-size: 18px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--text-strong);
}
.turn-timer.low {
  color: #e0554b;
  animation: time-pulse 1s steps(2, start) infinite;
}
@keyframes time-pulse {
  50% {
    opacity: 0.45;
  }
}
.spell-slot {
  position: relative;
}
.spell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  width: 104px;
  padding: 14px 8px;
  font-size: 12px;
  font-weight: 500;
  line-height: 1.2;
  background: var(--surface);
  color: var(--text-strong);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}
.spell img {
  width: 38px;
  height: 38px;
  image-rendering: pixelated;
}
.spell:hover:not(:disabled) {
  border-color: var(--border-strong);
}
.spell.active {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.spell:disabled {
  opacity: 0.4;
  cursor: not-allowed;
  pointer-events: none;
}

.tooltip {
  position: absolute;
  top: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  width: 232px;
  display: none;
  flex-direction: column;
  gap: 5px;
  text-align: left;
  padding: 12px 14px;
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  font-size: 12px;
  line-height: 1.45;
  z-index: 10;
  pointer-events: none;
}
.spell-slot:hover .tooltip {
  display: flex;
}
.tooltip strong {
  color: var(--text-strong);
  font-size: 13px;
  font-weight: 600;
}
.tt-cost {
  color: var(--accent);
  font-weight: 500;
}
.tt-effects {
  color: var(--text-strong);
}

.actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
}
.budget {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--text);
}
</style>
