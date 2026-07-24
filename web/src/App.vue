<script setup>
import { ref, onMounted } from "vue";
const state = ref(null);
const cells = ref(null);
const view = ref(null);
const canvas_ref = ref(null);
const legal = ref(null);
const hint = ref(null);
const hintLoading = ref(false);
const selectedSpell = ref(null);
const sims = ref(1000);
const autoplay = ref(false);
const currentSeed = ref(null);
const solving = ref(false);
const solveBudget = ref(null);
const W = 40,
  H = 20,
  OFFX = 380,
  OFFY = 10;

async function fetchLegal() {
  const r = await fetch(`http://127.0.0.1:8000/games/${state.value.game_id}/legal`);
  legal.value = await r.json();
}

function cellToPixel(x, y) {
  return { px: OFFX + (x - y) * (W / 2), py: OFFY + (x + y) * (H / 2) };
}

function drawDot(ctx, x, y, color, r = 6) {
  const { px, py } = cellToPixel(x, y);
  ctx.beginPath();
  ctx.arc(px, py, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawTile(ctx, x, y, color_fill, color_stroke) {
  const { px, py } = cellToPixel(x, y);
  ctx.beginPath();
  ctx.moveTo(px, py - H / 2);
  ctx.lineTo(px + W / 2, py);
  ctx.lineTo(px, py + H / 2);
  ctx.lineTo(px - W / 2, py);
  ctx.closePath();
  ctx.fillStyle = color_fill;
  ctx.fill();
  ctx.strokeStyle = color_stroke;
  ctx.stroke();
}

function render() {
  const ctx = canvas_ref.value.getContext("2d");
  ctx.clearRect(0, 0, canvas_ref.value.width, canvas_ref.value.height);
  const s = view.value;
  cells.value.cells.forEach(({ x, y }) => drawTile(ctx, x, y, "#3a4a5a", "#1a2530"));
  s.next_spawn.forEach((sp) => drawTile(ctx, sp[0], sp[1], "#950bbfe4", "#1a2530"));
  s.flames.forEach((f) => drawDot(ctx, f.x, f.y, "#ff6600"));
  drawDot(ctx, s.bolgrot.x, s.bolgrot.y, "#cc0000", 8);
  drawDot(ctx, s.player.x, s.player.y, "#3399ff", 8);
  if (selectedSpell.value !== null && legal.value) {
    const p = view.value.player;
    legal.value.actions
      .filter((a) => a.spell === selectedSpell.value)
      .forEach((a) => drawTile(ctx, p.x + a.dx, p.y + a.dy, "#ffee0088", "#ffcc00"));
  }
  if (hint.value) {
    const p = view.value.player;
    drawTile(ctx, p.x + hint.value.dx, p.y + hint.value.dy, "#00ff0088", "#00cc00");
  }
}

function selectSpell(i) {
  selectedSpell.value = i;
  render();
}

function descLines(desc) {
  return desc.split(".").map((s) => s.trim()).filter(Boolean);
}

function onCanvasClick(event) {
  const canvas = canvas_ref.value;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (event.clientX - rect.left) * scaleX;
  const my = (event.clientY - rect.top) * scaleY;
  const u = mx - OFFX, v = my - OFFY;
  const x = Math.round(u / W + v / H);
  const y = Math.round(v / H - u / W);
  const p = view.value.player;
  const action = legal.value.actions.find(
    (a) => a.spell === selectedSpell.value && p.x + a.dx === x && p.y + a.dy === y
  );
  if (!action) return;
  playAction(action.index)
}

async function playAction(index) {
  const r = await fetch(
    `http://127.0.0.1:8000/games/${state.value.game_id}/step?action_index=${index}`,
    { method: "POST" }
  );
  hint.value = null
  const data = await r.json();
  view.value = data.state;
  selectedSpell.value = null;
  await fetchLegal();
  render();
  if (data.done) console.log(data.state.won ? "GAGNE" : "PERDU");
} 

async function startGame(seed = null) {
  if (seed === null) seed = Math.floor(Math.random() * 1e9);
  currentSeed.value = seed;
  const game = await fetch(`http://127.0.0.1:8000/games?seed=${seed}`, { method: "POST" });
  const game_data = await game.json();
  state.value = game_data;
  view.value = game_data.state;
  selectedSpell.value = null;
  hint.value = null;
  autoplay.value = false;
  await fetchLegal();
  render();
}

async function askHint() {
  hintLoading.value = true;
  const r = await fetch(
    `http://127.0.0.1:8000/games/${state.value.game_id}/hint?sims=${sims.value}`,
    { method: "POST" }
  );
  hint.value = await r.json();
  hintLoading.value = false;
  render();
}

async function toggleAutoplay() {
  autoplay.value = !autoplay.value;
  while (autoplay.value && !view.value.done) {
    const r = await fetch(
      `http://127.0.0.1:8000/games/${state.value.game_id}/hint?sims=${sims.value}`,
      { method: "POST" }
    );
    const h = await r.json();
    await playAction(h.index);
    await new Promise((res) => setTimeout(res, 300));
  }
  autoplay.value = false;
}

async function solveOnce(budget) {
  await startGame(currentSeed.value);
  while (!view.value.done) {
    const r = await fetch(
      `http://127.0.0.1:8000/games/${state.value.game_id}/hint?sims=${budget}`,
      { method: "POST" }
    );
    const h = await r.json();
    await playAction(h.index);
    await new Promise((res) => setTimeout(res, 150));
  }
  return view.value.won;
}

async function solveLevel() {
  solving.value = true;
  for (const budget of [1000, 2000, 4000, 8000]) {
    solveBudget.value = budget;
    const won = await solveOnce(budget);
    if (won) break;
  }
  solving.value = false;
  solveBudget.value = null;
}

onMounted(async () => {
  const map = await fetch("http://127.0.0.1:8000/map");
  cells.value = await map.json();
  await startGame(currentSeed.value);
});
</script>

<template>
  <div class="wrap">
    <h1>Bolgrot</h1>
    <p class="disclaimer">
      L'IA est un assistant très fort, mais la victoire n'est pas garantie à
      chaque partie. Sur un coup donné elle propose le meilleur coup connu ; sur
      un niveau perdu, « Réessayer plus fort » relance l'IA avec plus de
      réflexion (elle finit par gagner en cherchant plus profond). Attention :
      augmenter les sims augmente le temps de calcul de chaque coup.
    </p>

    <div v-if="view" class="hud">
      <span class="chip">❤️ {{ view.player.hp }} HP</span>
      <span class="chip">⭐ {{ view.player.pa }} PA</span>
      <span class="chip">Vague {{ view.waves.spawned }}/{{ view.waves.total }}</span>
      <span class="chip">{{ view.flames.length }} flammes</span>
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
      </div>
      <div class="actions">
        <button @click="playAction(16)">Fin de tour</button>
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
        <button @click="toggleAutoplay">
          {{ autoplay ? "⏹ Stop" : "▶ Autoplay" }}
        </button>
      </div>
    </div>

    <div class="board">
      <canvas @click="onCanvasClick" ref="canvas_ref" width="760" height="600"></canvas>
      <div v-if="view && view.done" class="overlay">
        <div class="modal">
          <h2>{{ view.won ? "Victoire !" : "Défaite" }}</h2>
          <button class="primary" @click="startGame()">Nouvelle partie</button>
          <button v-if="!view.won" @click="solveLevel" :disabled="solving">
            {{ solving ? `L'IA cherche à ${solveBudget} sims…` : "Réessayer plus fort" }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.wrap {
  max-width: 840px;
  margin: 0 auto;
  padding: 0 16px 48px;
}
.disclaimer {
  color: var(--text);
  font-size: 0.85rem;
  line-height: 1.5;
  max-width: 640px;
  margin: 0 auto 20px;
}
.hud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-bottom: 16px;
}
.chip {
  background: var(--accent-bg);
  color: var(--text-h);
  border: 1px solid var(--accent-border);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 14px;
}
.controls {
  display: flex;
  flex-direction: column;
  gap: 14px;
  align-items: center;
  margin-bottom: 18px;
}
.spells {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
}
.spell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  width: 96px;
  padding: 8px;
  font-size: 12px;
  line-height: 1.2;
  background: var(--code-bg);
  color: var(--text-h);
  border: 2px solid var(--border);
  border-radius: 10px;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}
.spell img {
  width: 40px;
  height: 40px;
  image-rendering: pixelated;
}
.spell:hover:not(:disabled) {
  border-color: var(--accent-border);
}
.spell.active {
  border-color: var(--accent);
  background: var(--accent-bg);
}
.spell:disabled {
  opacity: 0.4;
  cursor: not-allowed;
  pointer-events: none;
}
.spell-slot {
  position: relative;
}
.tooltip {
  position: absolute;
  top: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  width: 230px;
  display: none;
  flex-direction: column;
  gap: 4px;
  text-align: left;
  padding: 10px 12px;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--accent-border);
  border-radius: 8px;
  box-shadow: var(--shadow);
  font-size: 12px;
  line-height: 1.35;
  z-index: 10;
  pointer-events: none;
}
.spell-slot:hover .tooltip {
  display: flex;
}
.tooltip strong {
  color: var(--text-h);
  font-size: 13px;
}
.tt-cost {
  color: var(--accent);
}
.tt-effects {
  color: var(--text-h);
  font-style: italic;
}
.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
}
.budget {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  color: var(--text);
}
button {
  font: inherit;
  font-size: 15px;
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--code-bg);
  color: var(--text-h);
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}
button:hover:not(:disabled) {
  border-color: var(--accent-border);
  box-shadow: var(--shadow);
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
button.primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
select {
  font: inherit;
  font-size: 14px;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--code-bg);
  color: var(--text-h);
}
.board {
  position: relative;
  display: inline-block;
  line-height: 0;
}
canvas {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #10141a;
  box-shadow: var(--shadow);
}
.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 12px;
}
.modal {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: center;
  padding: 24px 28px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow);
  line-height: 1.4;
}
</style>