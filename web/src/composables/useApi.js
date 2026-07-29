// Toutes les requêtes vers le backend, centralisées.
// En dev (vite) le back tourne sur un autre port ; en prod le front buildé est
// servi par FastAPI lui-même, donc l'URL est relative (même origine).
const BASE_URL = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";

// Erreur HTTP typée : `.status` porte le code, `.notFound` isole le 404 d'un
// game_id perdu (l'instance serveur qui l'avait a disparu) pour que l'appelant
// puisse rejouer la session au lieu de lire un corps d'erreur comme une réponse.
class ApiError extends Error {
  constructor(status) {
    super(`HTTP ${status}`);
    this.status = status;
    this.notFound = status === 404;
  }
}

async function jget(path) {
  const r = await fetch(BASE_URL + path);
  if (!r.ok) throw new ApiError(r.status);
  return r.json();
}

async function jpost(path, body) {
  const opts = { method: "POST" };
  if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(BASE_URL + path, opts);
  if (!r.ok) throw new ApiError(r.status);
  return r.json();
}

export const api = {
  map: () => jget("/map"),
  newGame: (seed) => jpost(`/games?seed=${seed}`),
  customGame: (spawns) => jpost("/games/custom", { spawns }),
  legal: (id) => jget(`/games/${id}/legal`),
  step: (id, index) => jpost(`/games/${id}/step?action_index=${index}`),
  hint: (id, sims, engine = "az") =>
    jpost(`/games/${id}/hint?sims=${sims}&engine=${engine}`),
  setSpawn: (id, tiles) => jpost(`/games/${id}/spawn`, { tiles }),
  restore: (session) => jpost("/games/restore", session),
};
