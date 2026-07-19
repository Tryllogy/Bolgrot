<script setup>
import { ref, onMounted } from "vue";
const state = ref(null);
const cells = ref(null);
const view = ref(null);
const canvas_ref = ref(null);
const W = 40,
  H = 20,
  OFFX = 380,
  OFFY = 10;

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

onMounted(async () => {
  const map = await fetch("http://127.0.0.1:8000/map", { method: "GET" });
  const map_data = await map.json();
  const game = await fetch("http://127.0.0.1:8000/games", { method: "POST" });
  const game_data = await game.json();
  state.value = game_data;
  cells.value = map_data;
  view.value = game_data.state;
  const ctx = canvas_ref.value.getContext("2d");
  map_data.cells.forEach(({ x, y }) => {
    drawTile(ctx, x, y, "#3a4a5a", "#1a2530");
  });
  const s = game_data.state;
  s.next_spawn.forEach((spawn) => {
    drawTile(ctx, spawn[0], spawn[1], "#950bbfe4", "#1a2530");
  });
  s.flames.forEach((f) => drawDot(ctx, f.x, f.y, "#ff6600"));
  drawDot(ctx, s.bolgrot.x, s.bolgrot.y, "#cc0000", 8);
  drawDot(ctx, s.player.x, s.player.y, "#3399ff", 8);
});
</script>

<template>
  <h1>Bolgrot</h1>
  <!-- <pre>{{ state }}</pre> -->
  <div v-if="view">
    HP: {{ view.player.hp }} | PA: {{ view.player.pa }} | Vague:
    {{ view.waves.spawned }}/{{ view.waves.total }} | Flammes: {{ view.flames.length }}
  </div>
  <canvas ref="canvas_ref" width="760" height="600"></canvas>
</template>
