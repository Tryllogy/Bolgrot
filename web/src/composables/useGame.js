import { ref, computed, nextTick } from "vue";
import { api } from "./useApi.js";
import { useBoard } from "./useBoard.js";

// Store singleton : l'état et la logique de jeu vivent ici, une seule fois.
// Les composants importent ce dont ils ont besoin (pas de props/emits).
const board = useBoard();
const { drawDot, drawTile, fitBoard, pixelToCell, hasCell, FLAME_R, UNIT_R } =
  board;
export const canvasW = board.canvasW;
export const canvasH = board.canvasH;

// --- État réactif ---
export const state = ref(null);
export const cells = ref(null);
export const view = ref(null);
export const canvas_ref = ref(null);
export const legal = ref(null);
export const hint = ref(null);
export const hintLoading = ref(false);
export const selectedSpell = ref(null);
export const sims = ref(1000);
export const autoplay = ref(false);
export const currentSeed = ref(null);
export const solving = ref(false);
export const solveBudget = ref(null);
export const hover = ref(null); // case sous le curseur {x, y} ou null
export const toast = ref(null); // texte éphémère (coup joué par l'IA)
let toastTimer = null;

// Compte à rebours du tour : 1 tour = 2 min (comme le jeu de base, TIME_TURN).
const TURN_SECONDS = 120;
export const turnTime = ref(TURN_SECONDS); // secondes restantes sur le tour
let turnInterval = null;
export const turnClock = computed(() => {
  const t = Math.max(0, turnTime.value);
  return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
});
export const turnLow = computed(() => turnTime.value <= 15); // alerte fin de tour
export const screen = ref("home"); // "home" | "play" | "editor"
export const basePlayer = ref(null); // [x, y] départ joueur
export const baseBolgrot = ref(null); // [x, y] Bolgrot
export const editorSpawns = ref([]); // spawns posés dans l'éditeur [{x, y}]
export const customMode = ref(false); // partie perso (vagues dessinées)
export const editorContext = ref("create"); // "create" 1re vague | "next"
export const editorWave = ref(1); // numéro de vague en cours de placement

// Enregistre l'élément <canvas> (fonction-ref depuis GameBoard).
export function setCanvas(el) {
  canvas_ref.value = el;
}

async function fetchLegal() {
  legal.value = await api.legal(state.value.game_id);
}

export function render() {
  if (!canvas_ref.value) return; // rien à dessiner sur l'accueil
  const ctx = canvas_ref.value.getContext("2d");
  ctx.clearRect(0, 0, canvas_ref.value.width, canvas_ref.value.height);
  cells.value.cells.forEach(({ x, y }) => drawTile(ctx, x, y, "#3a4a5a", "#1a2530"));

  if (screen.value === "editor") {
    const g = view.value; // présent quand on place une vague EN COURS de partie
    if (g) g.flames.forEach((f) => drawDot(ctx, f.x, f.y, "#ff6600", FLAME_R));
    if (hover.value) {
      drawTile(ctx, hover.value.x, hover.value.y, "#ffffff26", "#ffffffaa");
    }
    editorSpawns.value.forEach((f) => drawTile(ctx, f.x, f.y, "#950bbfe4", "#1a2530"));
    const bx = g ? g.bolgrot.x : baseBolgrot.value[0];
    const by = g ? g.bolgrot.y : baseBolgrot.value[1];
    const px = g ? g.player.x : basePlayer.value[0];
    const py = g ? g.player.y : basePlayer.value[1];
    drawDot(ctx, bx, by, "#cc0000", UNIT_R);
    drawDot(ctx, px, py, "#3399ff", UNIT_R);
    return;
  }

  const s = view.value;
  if (!s) return;
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

export function selectSpell(i) {
  // Re-cliquer le sort déjà sélectionné le désélectionne (efface la previsu).
  selectedSpell.value = selectedSpell.value === i ? null : i;
  render();
}

export function descLines(desc) {
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

export function onCanvasClick(event) {
  const { x, y } = pixelToCell(canvas_ref.value, event);
  if (screen.value === "editor") {
    editorToggle(x, y);
    return;
  }
  const p = view.value.player;
  const action = legal.value.actions.find(
    (a) => a.spell === selectedSpell.value && p.x + a.dx === x && p.y + a.dy === y
  );
  if (!action) {
    // Clic à côté d'une cible : on désélectionne (efface la previsu).
    if (selectedSpell.value !== null) {
      selectedSpell.value = null;
      render();
    }
    return;
  }
  playAction(action.index);
}

function editorToggle(x, y) {
  if (!hasCell(x, y)) return; // hors map
  const g = view.value; // partie en cours (placement d'une vague suivante)
  const occupied = g
    ? (x === g.player.x && y === g.player.y) ||
      (x === g.bolgrot.x && y === g.bolgrot.y) ||
      g.flames.some((f) => f.x === x && f.y === y)
    : (x === basePlayer.value[0] && y === basePlayer.value[1]) ||
      (x === baseBolgrot.value[0] && y === baseBolgrot.value[1]);
  if (occupied) return; // case déjà prise (joueur / Bolgrot / flamme)
  const i = editorSpawns.value.findIndex((f) => f.x === x && f.y === y);
  if (i >= 0) editorSpawns.value.splice(i, 1);
  else if (editorSpawns.value.length < 6) editorSpawns.value.push({ x, y });
  render();
}

export function onCanvasMove(event) {
  const { x, y } = pixelToCell(canvas_ref.value, event);
  const on = hasCell(x, y) ? { x, y } : null;
  const changed = on?.x !== hover.value?.x || on?.y !== hover.value?.y;
  if (changed) {
    hover.value = on;
    render();
  }
}

export function onCanvasLeave() {
  if (hover.value) {
    hover.value = null;
    render();
  }
}

// Le timer ne tourne que pendant une partie jouée à la main : pas sur l'accueil
// ni l'éditeur, pas quand la partie est finie, et pas pendant l'IA (autoplay /
// résolution) qui enchaîne les tours elle-même.
function turnTimerActive() {
  return (
    screen.value === "play" &&
    !!view.value &&
    !view.value.done &&
    !autoplay.value &&
    !solving.value
  );
}

// Un seul intervalle, monté une fois (voir init), gardé par turnTimerActive.
// À 0 : fin de tour automatique (une nouvelle vague apparaît), puis on repart à
// 120 s. Le temps n'est PAS remis à zéro par les actions individuelles — les
// 2 min sont le budget de tout le tour ; seule une fin de tour (manuelle ou
// automatique) le relance. On repart à 120 AVANT le playAction asynchrone pour
// que le tick suivant ne déclenche pas une deuxième fin de tour.
function ensureTurnTimer() {
  if (turnInterval) return;
  turnInterval = setInterval(async () => {
    if (!turnTimerActive()) return;
    if (turnTime.value > 0) {
      turnTime.value -= 1;
    } else {
      turnTime.value = TURN_SECONDS;
      await playAction(16);
    }
  }, 1000);
}

export async function playAction(index) {
  hint.value = null;
  const data = await api.step(state.value.game_id, index);
  view.value = data.state;
  if (index === 16) turnTime.value = TURN_SECONDS; // fin de tour → nouveau tour
  selectedSpell.value = null;
  await fetchLegal();
  render();
  // Partie perso : après une fin de tour, le joueur dessine la vague suivante.
  if (
    customMode.value &&
    index === 16 &&
    !view.value.done &&
    view.value.waves.spawned < view.value.waves.total
  ) {
    openNextWaveEditor();
  }
}

function applyGame(data) {
  state.value = data;
  view.value = data.state;
  selectedSpell.value = null;
  hint.value = null;
  autoplay.value = false;
  turnTime.value = TURN_SECONDS; // nouvelle partie → tour plein
}

async function startGame(seed = null) {
  if (seed === null) seed = Math.floor(Math.random() * 1e9);
  currentSeed.value = seed;
  customMode.value = false;
  applyGame(await api.newGame(seed));
  await fetchLegal();
  render();
}

async function startCustomGame(spawns) {
  customMode.value = true;
  applyGame(await api.customGame(spawns.map((f) => [f.x, f.y])));
  await fetchLegal();
  render();
}

// --- Navigation entre écrans ---
export async function startRandom() {
  screen.value = "play";
  await nextTick();
  await startGame();
}

export async function openEditor() {
  view.value = null; // éditeur "création" : plateau vierge, positions de base
  customMode.value = false;
  editorContext.value = "create";
  editorWave.value = 1;
  editorSpawns.value = [];
  hover.value = null;
  screen.value = "editor";
  await nextTick();
  render();
}

export function clearEditor() {
  editorSpawns.value = [];
  render();
}

export async function playCustom() {
  if (editorSpawns.value.length === 0) return;
  screen.value = "play";
  await nextTick();
  await startCustomGame(editorSpawns.value);
}

// Entre deux tours d'une partie perso : le joueur dessine la vague suivante.
function openNextWaveEditor() {
  editorContext.value = "next";
  editorWave.value = view.value.waves.spawned + 1;
  editorSpawns.value = [];
  hover.value = null;
  screen.value = "editor";
  render();
}

export async function submitSpawn() {
  if (editorSpawns.value.length === 0) return;
  const data = await api.setSpawn(
    state.value.game_id,
    editorSpawns.value.map((f) => [f.x, f.y])
  );
  view.value = data.state;
  screen.value = "play";
  await nextTick();
  await fetchLegal();
  render();
  if (autoplay.value) await runAutoplay(); // reprise de l'IA (hybride perso)
}

export function stopAutoplay() {
  autoplay.value = false;
}

export function goHome() {
  autoplay.value = false;
  screen.value = "home";
}

export async function newGame() {
  if (customMode.value) await openEditor();
  else await startGame();
}

export async function askHint() {
  hintLoading.value = true;
  hint.value = await api.hint(state.value.game_id, sims.value);
  hintLoading.value = false;
  showToast(`Indice : ${moveLabel(hint.value)}`);
  render();
}

// Joue tant qu'on est en partie (screen "play"). En partie perso, playAction
// bascule sur l'éditeur à la fin d'un tour : la boucle s'arrête alors SANS
// éteindre autoplay, et submitSpawn la relance après la pose de la vague.
async function runAutoplay() {
  while (autoplay.value && !view.value.done && screen.value === "play") {
    const h = await api.hint(state.value.game_id, sims.value);
    showToast(`${moveLabel(h)}`);
    await playAction(h.index);
    await new Promise((res) => setTimeout(res, 300));
  }
  if (view.value?.done) autoplay.value = false;
}

export async function toggleAutoplay() {
  autoplay.value = !autoplay.value;
  if (autoplay.value) await runAutoplay();
}

async function solveOnce(budget) {
  await startGame(currentSeed.value);
  while (!view.value.done) {
    const h = await api.hint(state.value.game_id, budget);
    showToast(`${moveLabel(h)}`);
    await playAction(h.index);
    await new Promise((res) => setTimeout(res, 150));
  }
  return view.value.won;
}

export async function solveLevel() {
  solving.value = true;
  for (const budget of [1000, 2000, 4000, 8000]) {
    solveBudget.value = budget;
    const won = await solveOnce(budget);
    if (won) break;
  }
  solving.value = false;
  solveBudget.value = null;
}

// Appelé une fois au montage de l'app : charge la map + positions de base.
export async function init() {
  const md = await api.map();
  cells.value = md;
  basePlayer.value = md.player;
  baseBolgrot.value = md.bolgrot;
  fitBoard(cells.value.cells);
  ensureTurnTimer(); // démarre le compte à rebours de tour (gardé par l'état)
}
