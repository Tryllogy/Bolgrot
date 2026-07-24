// Toutes les requêtes vers le backend, centralisées.
// En dev (vite) le back tourne sur un autre port ; en prod le front buildé est
// servi par FastAPI lui-même, donc l'URL est relative (même origine).
const BASE_URL = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";

async function jget(path) {
  const r = await fetch(BASE_URL + path);
  return r.json();
}

async function jpost(path, body) {
  const opts = { method: "POST" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(BASE_URL + path, opts);
  return r.json();
}

export const api = {
  map: () => jget("/map"),
  newGame: (seed) => jpost(`/games?seed=${seed}`),
  customGame: (spawns) => jpost("/games/custom", { spawns }),
  legal: (id) => jget(`/games/${id}/legal`),
  step: (id, index) => jpost(`/games/${id}/step?action_index=${index}`),
  hint: (id, sims) => jpost(`/games/${id}/hint?sims=${sims}`),
  setSpawn: (id, tiles) => jpost(`/games/${id}/spawn`, { tiles }),
};
