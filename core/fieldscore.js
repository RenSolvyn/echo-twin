// Field test scoring: compare a solved room with tape-measured truth (TESTING.md).
// truth: { walls: [m, ...] perpendicular distances from the turning spot (between the feet) to each wall,
//          headings?: [deg | null, ...] compass bearing of each wall (top edge pointed at it, v0.6.2),
//          roomHeight?: m, phoneHeight?: m floor to phone at the end of the lift }
// v0.6.2: with headings and a scan whose compass agrees with its turn, each tape wall is paired with the sound
// wall in its direction (within 30 deg), never by distance; pairing by distance alone flatters the result (a
// wrong wall at a similar distance counts), so it is used only without a compass, and the report says so.

export function parseTruth(text, roomHeight, phoneHeight) {
  const walls = String(text || '').split(/[\s,;]+/).map(Number).filter(v => Number.isFinite(v) && v > 0.2 && v < 15);
  const t = { walls };
  if (Number.isFinite(+roomHeight) && +roomHeight > 1.5) t.roomHeight = +roomHeight;
  if (Number.isFinite(+phoneHeight) && +phoneHeight > 0.3) t.phoneHeight = +phoneHeight;
  return walls.length >= 2 ? t : null;
}

// Rows from the form: [{ d: text, h: deg | null }] -> truth (walls with their headings kept aligned) or null.
export function truthFromRows(rows, roomHeight, phoneHeight) {
  const ok = (rows || []).map(r => ({ d: Number(String(r.d ?? '').trim().replace(',', '.')), h: Number.isFinite(r.h) ? r.h : null }))
    .filter(r => Number.isFinite(r.d) && r.d > 0.2 && r.d < 15);
  if (ok.length < 2) return null;
  const t = { walls: ok.map(r => r.d) };
  if (ok.some(r => r.h != null)) t.headings = ok.map(r => r.h);
  if (Number.isFinite(+roomHeight) && +roomHeight > 1.5) t.roomHeight = +roomHeight;
  if (Number.isFinite(+phoneHeight) && +phoneHeight > 0.3) t.phoneHeight = +phoneHeight;
  return t;
}

const DEG = Math.PI / 180;
const wrapDeg = (a) => ((a % 360) + 540) % 360 - 180;

// Ties the scan's heading frame (yaw of the bottom edge, arbitrary zero) to the compass, from the frames that
// carry both (f.yaw rad, f.hdg deg of the top edge). Physics says yaw = c - (hdg + 180) (counterclockwise yaw,
// clockwise compass); the sign is still taken from the data: over a full turn only the right sign keeps c
// constant. Returns { ok, c, s, spreadDeg, n, spanDeg, why }. ok needs 20 frames, 270 deg of turn and a spread
// (circular standard deviation of c) of 25 deg or less. Time O(n).
export function compassFrame(frames) {
  const F = (frames || []).filter(f => Number.isFinite(f.hdg) && Number.isFinite(f.yaw));
  const n = F.length;
  if (n < 20) return { ok: false, n, why: n ? 'too few compass readings' : 'no compass readings in this scan' };
  let lo = Infinity, hi = -Infinity; for (const f of F) { lo = Math.min(lo, f.yaw); hi = Math.max(hi, f.yaw); }
  const spanDeg = (hi - lo) / DEG;
  const fit = (s) => { let x = 0, y = 0; for (const f of F) { const c = (f.yaw / DEG - s * (f.hdg + 180)) * DEG; x += Math.cos(c); y += Math.sin(c); } const R = Math.hypot(x, y) / n; return { s, R, c: Math.atan2(y, x) / DEG }; };
  const a = fit(-1), b = fit(1), best = a.R >= b.R ? a : b;
  const spreadDeg = Math.sqrt(-2 * Math.log(Math.max(1e-9, Math.min(1, best.R)))) / DEG;
  const out = { c: best.c, s: best.s, spreadDeg, n, spanDeg };
  if (spanDeg < 270) return { ...out, ok: false, why: 'the turn was too short to tie the compass to it' };
  if (spreadDeg > 25) return { ...out, ok: false, why: `the compass disagreed with the turn by ${Math.round(spreadDeg)} deg (metal or electronics nearby)` };
  return { ...out, ok: true };
}

// Direction (rad, scan frame) of a wall whose compass bearing (top edge pointed at it) is h deg.
export function wallDirection(h, frame) { return (frame.c + frame.s * h) * DEG; }

// ctx: { frame (compassFrame), cloud ({ nB, ridge, strength }) }. Time O(T x E).
export function scoreField(room, truth, tol = 0.15, ctx = {}) {
  if (!room || !room.ok || !truth || !truth.walls || !truth.walls.length) return null;
  const estW = room.walls.filter(w => !w.inferred), est = estW.map(w => w.d);
  const T = truth.walls.length, fr = ctx.frame, H = truth.headings || [];
  const byDir = !!(fr && fr.ok) && H.some(h => Number.isFinite(h));
  const dir = truth.walls.map((m, i) => (byDir && Number.isFinite(H[i]) ? wallDirection(H[i], fr) : null));
  const ui = new Set(), uj = new Set(), match = [];
  // 1. by direction: one to one, smallest angle first, within 30 deg
  const ang = [];
  dir.forEach((a, i) => { if (a != null) estW.forEach((w, j) => { const d = Math.abs(wrapDeg((w.phi - a) / DEG)); if (d <= 30) ang.push({ i, j, d }); }); });
  ang.sort((p, q) => p.d - q.d);
  for (const p of ang) { if (ui.has(p.i) || uj.has(p.j)) continue; ui.add(p.i); uj.add(p.j); match.push({ i: p.i, j: p.j, err: Math.abs(est[p.j] - truth.walls[p.i]), by: 'direction' }); }
  // 2. walls without a usable direction: by distance among what is left (flagged)
  const pairs = [];
  truth.walls.forEach((m, i) => { if (dir[i] == null) est.forEach((e, j) => pairs.push({ i, j, err: Math.abs(e - m) })); });
  pairs.sort((a, b) => a.err - b.err);
  for (const p of pairs) { if (ui.has(p.i) || uj.has(p.j)) continue; ui.add(p.i); uj.add(p.j); match.push({ ...p, by: 'distance' }); }
  const good = match.filter(p => p.err <= tol);
  const errs = good.map(p => p.err).sort((a, b) => a - b);
  const cl = ctx.cloud;
  const out = {
    measured: T, estimated: est.length, inferred: room.walls.length - est.length,
    recall: good.length / T, precision: est.length ? good.length / est.length : 0,
    errMedCm: errs.length ? 100 * errs[errs.length >> 1] : null,
    matchedBy: byDir ? (dir.every(d => d != null) ? 'direction' : 'mixed') : 'distance',
    compass: fr ? { ok: fr.ok, spreadDeg: fr.spreadDeg ?? null, why: fr.why || null } : null,
    perWall: truth.walls.map((m, i) => {
      const p = match.find(q => q.i === i), w = { measured: m, estimated: p ? est[p.j] : null, errCm: p ? 100 * p.err : null, by: p ? p.by : (dir[i] != null ? 'direction' : 'distance'), heading: Number.isFinite(H[i]) ? H[i] : null };
      if (cl && dir[i] != null) {
        let b = Math.round((dir[i] / DEG) / (360 / cl.nB)) % cl.nB; if (b < 0) b += cl.nB;
        const r = cl.ridge[b];
        w.glow = Number.isFinite(r) ? r : null; w.glowErrCm = Number.isFinite(r) ? 100 * Math.abs(r - m) : null; w.glowStrength = cl.strength ? cl.strength[b] : null;
      }
      return w;
    })
  };
  if (cl && byDir) { const g = out.perWall.filter(w => w.glowErrCm != null); out.glowFound = g.filter(w => w.glowErrCm <= 100 * tol).length; out.glowScored = g.length; }
  if (truth.phoneHeight && room.floor && Number.isFinite(room.floor.h)) out.floorErrCm = 100 * Math.abs(room.floor.h - truth.phoneHeight);
  if (truth.roomHeight && truth.phoneHeight && room.ceil && Number.isFinite(room.ceil.h)) out.ceilErrCm = 100 * Math.abs(room.ceil.h - (truth.roomHeight - truth.phoneHeight));
  return out;
}

export function fieldText(s) {
  if (!s) return 'Enter at least two measured wall distances.';
  const f = (v) => (v == null ? '-' : v.toFixed(1));
  const how = s.matchedBy === 'direction' ? 'paired by compass' : s.matchedBy === 'mixed' ? 'paired by compass where aimed, else by distance' : 'paired by distance only (flatters the result)';
  const lines = [
    `walls found ${Math.round(s.recall * s.measured)} of ${s.measured} within 15 cm; ${s.estimated} measured by sound, ${s.inferred} inferred`,
    `${how}`,
    `median error of found walls ${f(s.errMedCm)} cm`,
    ...s.perWall.map((w, k) => `  wall ${k + 1}${w.heading != null ? ' (' + Math.round(w.heading) + ' deg)' : ''}: tape ${w.measured.toFixed(2)} m, sound ${w.estimated == null ? '-' : w.estimated.toFixed(2) + ' m'}, error ${f(w.errCm)} cm` +
      (w.glowErrCm !== undefined ? `, glow ${w.glow == null ? '-' : w.glow.toFixed(2) + ' m'} (${f(w.glowErrCm)} cm)` : ''))
  ];
  if (s.glowScored) lines.splice(3, 0, `glow outline within 15 cm on ${s.glowFound} of ${s.glowScored} aimed walls`);
  if (s.compass && !s.compass.ok && s.compass.why) lines.push(`compass not used: ${s.compass.why}`);
  if (s.floorErrCm != null) lines.push(`floor distance error ${f(s.floorErrCm)} cm`);
  if (s.ceilErrCm != null) lines.push(`ceiling distance error ${f(s.ceilErrCm)} cm`);
  return lines.join('\n');
}
