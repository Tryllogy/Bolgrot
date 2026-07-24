<script setup>
import { ref, onMounted, nextTick } from "vue";
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
const W = 64,
  H = 32;
let OFFX = 0,
  OFFY = 0;
const canvasW = ref(760);
const canvasH = ref(600);
const FLAME_R = H * 0.38; // rayon des flammes, proportionnel à la tuile
const UNIT_R = H * 0.48; // rayon joueur / Bolgrot
const hover = ref(null); // case sous le curseur {x, y} ou null
let cellSet = new Set(); // clés "x,y" des cases valides (pour le hover)
const toast = ref(null); // texte éphémère (coup joué par l'IA)
let toastTimer = null;

function fitBoard() {
  cellSet = new Set();
  let dxmin = Infinity, dxmax = -Infinity, smin = Infinity, smax = -Infinity;
  for (const { x, y } of cells.value.cells) {
    cellSet.add(`${x},${y}`);
    dxmin = Math.min(dxmin, x - y);
    dxmax = Math.max(dxmax, x - y);
    smin = Math.min(smin, x + y);
    smax = Math.max(smax, x + y);
  }
  OFFX = -dxmin * (W / 2) + W;
  OFFY = -smin * (H / 2) + H;
  canvasW.value = (dxmax - dxmin) * (W / 2) + 2 * W;
  canvasH.value = (smax - smin) * (H / 2) + 2 * H;
}

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
  if (hover.value) {
    drawTile(ctx, hover.value.x, hover.value.y, "#ffffff26", "#ffffffaa");
  }
  s.flames.forEach((f) => drawDot(ctx, f.x, f.y, "#ff6600", FLAME_R));
  drawDot(ctx, s.bolgrot.x, s.bolgrot.y, "#cc0000", UNIT_R);
  drawDot(ctx, s.player.x, s.player.y, "#3399ff", UNIT_R);
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

function moveLabel(m) {
  if (m.spell === null || m.spell === undefined) return "Fin de tour";
  return view.value.spells[m.spell]?.name ?? "Sort";
}

function showToast(msg) {
  toast.value = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toast.value = null), 1800);
}

function pixelToCell(event) {
  const canvas = canvas_ref.value;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (event.clientX - rect.left) * scaleX;
  const my = (event.clientY - rect.top) * scaleY;
  const u = mx - OFFX, v = my - OFFY;
  return { x: Math.round(u / W + v / H), y: Math.round(v / H - u / W) };
}

function onCanvasClick(event) {
  const { x, y } = pixelToCell(event);
  const p = view.value.player;
  const action = legal.value.actions.find(
    (a) => a.spell === selectedSpell.value && p.x + a.dx === x && p.y + a.dy === y
  );
  if (!action) return;
  playAction(action.index)
}

function onCanvasMove(event) {
  const { x, y } = pixelToCell(event);
  const on = cellSet.has(`${x},${y}`) ? { x, y } : null;
  const changed = (on?.x !== hover.value?.x) || (on?.y !== hover.value?.y);
  if (changed) {
    hover.value = on;
    render();
  }
}

function onCanvasLeave() {
  if (hover.value) {
    hover.value = null;
    render();
  }
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
  showToast(`Indice : ${moveLabel(hint.value)}`);
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
    showToast(`${moveLabel(h)}`);
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
    showToast(`${moveLabel(h)}`);
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
  fitBoard();
  await nextTick();
  await startGame(currentSeed.value);
});
</script>

<template>
  <div class="wrap">
    <h1>Bolgrot</h1>
    <p class="disclaimer">
      L'IA est un assistant très fort, mais la victoire n'est pas garantie à
      chaque partie. Sur un coup donné elle propose le meilleur coup connu ; sur
      un niveau perdu, "Réessayer plus fort" relance l'IA avec plus de
      réflexion (elle finit par gagner en cherchant plus profond). Attention :
      augmenter les sims augmente le temps de calcul de chaque coup.
      (~1s/1000)
    </p>

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
        <button class="end-turn" @click="playAction(16)">Fin de tour</button>
      </div>
      <div class="actions">
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
      <canvas
        @click="onCanvasClick"
        @mousemove="onCanvasMove"
        @mouseleave="onCanvasLeave"
        ref="canvas_ref"
        :width="canvasW"
        :height="canvasH"></canvas>
      <Transition name="fade">
        <div v-if="toast" class="toast">{{ toast }}</div>
      </Transition>
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
  max-width: 1000px;
  margin: 0 auto;
  padding: 56px 24px 72px;
  text-align: center;
}
.wrap :deep(h1) {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text-strong);
  margin: 0 0 12px;
}
.disclaimer {
  color: var(--text);
  font-size: 0.8125rem;
  line-height: 1.6;
  max-width: 560px;
  margin: 0 auto 36px;
}

/* HUD ---------------------------------------------------------------------- */
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

/* Controls ----------------------------------------------------------------- */
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
.end-turn {
  margin-left: 104px; /* espacement d'une largeur de sort */
  align-self: center;
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

/* Tooltip ------------------------------------------------------------------ */
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

/* Actions ------------------------------------------------------------------ */
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
button {
  font: inherit;
  font-size: 14px;
  font-weight: 500;
  padding: 9px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-strong);
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s, opacity 0.15s;
}
button:hover:not(:disabled) {
  border-color: var(--border-strong);
  background: var(--surface-2);
}
button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
button.primary {
  background: var(--accent);
  color: var(--accent-contrast);
  border-color: var(--accent);
}
button.primary:hover:not(:disabled) {
  filter: brightness(1.06);
  background: var(--accent);
  border-color: var(--accent);
}
select {
  font: inherit;
  font-size: 13px;
  font-weight: 500;
  text-transform: none;
  letter-spacing: normal;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-strong);
  cursor: pointer;
}

/* Board -------------------------------------------------------------------- */
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
.modal :deep(h2) {
  font-size: 1.375rem;
  font-weight: 600;
  color: var(--text-strong);
}
</style>