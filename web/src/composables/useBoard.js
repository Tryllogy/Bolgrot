import { ref } from "vue";

// Géométrie iso + primitives de dessin du plateau (canvas). Ne connaît rien de
// l'état de jeu : on lui passe les données à dessiner.
export function useBoard() {
  const W = 64,
    H = 32;
  let OFFX = 0,
    OFFY = 0;
  const canvasW = ref(760);
  const canvasH = ref(600);
  const FLAME_R = H * 0.38; // rayon flammes, proportionnel à la tuile
  const UNIT_R = H * 0.48; // rayon joueur / Bolgrot
  let cellSet = new Set(); // clés "x,y" des cases valides

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

  function drawTile(ctx, x, y, colorFill, colorStroke) {
    const { px, py } = cellToPixel(x, y);
    ctx.beginPath();
    ctx.moveTo(px, py - H / 2);
    ctx.lineTo(px + W / 2, py);
    ctx.lineTo(px, py + H / 2);
    ctx.lineTo(px - W / 2, py);
    ctx.closePath();
    ctx.fillStyle = colorFill;
    ctx.fill();
    ctx.strokeStyle = colorStroke;
    ctx.stroke();
  }

  // Ajuste le canvas au plus près des bornes iso de la map (cellules {x,y}).
  function fitBoard(cells) {
    cellSet = new Set();
    let dxmin = Infinity,
      dxmax = -Infinity,
      smin = Infinity,
      smax = -Infinity;
    for (const { x, y } of cells) {
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

  // Projection inverse : pixel de l'événement souris -> case (x, y).
  function pixelToCell(canvas, event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (event.clientX - rect.left) * scaleX;
    const my = (event.clientY - rect.top) * scaleY;
    const u = mx - OFFX,
      v = my - OFFY;
    return { x: Math.round(u / W + v / H), y: Math.round(v / H - u / W) };
  }

  const hasCell = (x, y) => cellSet.has(`${x},${y}`);

  return {
    canvasW,
    canvasH,
    FLAME_R,
    UNIT_R,
    drawDot,
    drawTile,
    fitBoard,
    pixelToCell,
    hasCell,
  };
}
