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
  <h1>Bolgrot</h1>
  <p class="disclaimer">
    L'IA est un assistant tres fort, mais la victoire n'est pas garantie
    chaque partie. Sur un coup donne elle propose le meilleur coup connu ; sur un
    niveau perdu, "Reessayer plus fort" relance l'IA avec plus de reflexion
    (elle finit par gagner en cherchant plus profond).
    Attention cependant, augmenter les sims augmente le temps necessaire pour que
    l'IA trouve le meilleur coup.
  </p>
  <div v-if="view">
    HP: {{ view.player.hp }} | PA: {{ view.player.pa }} | Vague:
    {{ view.waves.spawned }}/{{ view.waves.total }} | Flammes: {{ view.flames.length }}
  </div>
  <div v-if="view">
    <button
      v-for="(sp, i) in view.spells"
      :key="i"
      :disabled="!sp.castable"
      @click="selectSpell(i)">
      {{ sp.name }}
    </button>
  </div>
  <div>
    <button v-if="view && !view.done" @click="playAction(16)">
      Fin de tour
    </button>
  </div>
  <div v-if="view && !view.done">
    <select v-model.number="sims">
      <option :value="300">Rapide (300)</option>
      <option :value="1000">Normal (1000)</option>
      <option :value="8000">Fort (8000)</option>
    </select>
    <button @click="askHint" :disabled="hintLoading || autoplay">
      {{ hintLoading ? "Reflexion" : "Indice" }}
    </button>
    <button @click="toggleAutoplay">
      {{ autoplay ? "⏹ Stop" : "▶ Autoplay" }}
    </button>
  </div>
  <div v-if="view && view.done">
    <h2>{{ view.won ? "Vous avez gagne !" : "Vous avez perdu" }}</h2>
    <button @click="startGame()">Nouvelle partie</button>
    <button v-if="!view.won" @click="solveLevel" :disabled="solving">
      {{ solving ? `L'IA cherche a ${solveBudget} sims` : "Reessayer plus fort (IA)" }}
    </button>
  </div>
  <canvas @click="onCanvasClick" ref="canvas_ref" width="760" height="600"></canvas>
</template>

<style>
  .disclaimer { color: #888; font-size: 0.85rem; max-width: 640px; }
</style>