// Field test scoring: compare a solved room with tape-measured truth (TESTING.md).
// truth: { walls: [m, ...] perpendicular distances from the turning spot (between the feet) to each wall,
//          roomHeight?: m, phoneHeight?: m floor to phone at the end of the lift }
// Matching is one to one by smallest error, so it needs no compass; turn order is kept for the report only.

export function parseTruth(text, roomHeight, phoneHeight) {
  const walls = String(text || '').split(/[\s,;]+/).map(Number).filter(v => Number.isFinite(v) && v > 0.2 && v < 15);
  const t = { walls };
  if (Number.isFinite(+roomHeight) && +roomHeight > 1.5) t.roomHeight = +roomHeight;
  if (Number.isFinite(+phoneHeight) && +phoneHeight > 0.3) t.phoneHeight = +phoneHeight;
  return walls.length >= 2 ? t : null;
}

export function scoreField(room, truth, tol = 0.15) {
  if (!room || !room.ok || !truth || !truth.walls || !truth.walls.length) return null;
  const est = room.walls.filter(w => !w.inferred).map(w => w.d);
  const pairs = [];
  truth.walls.forEach((m, i) => est.forEach((e, j) => pairs.push({ i, j, err: Math.abs(e - m) })));
  pairs.sort((a, b) => a.err - b.err);
  const ui = new Set(), uj = new Set(), match = [];
  for (const p of pairs) { if (ui.has(p.i) || uj.has(p.j)) continue; ui.add(p.i); uj.add(p.j); match.push(p); }
  const good = match.filter(p => p.err <= tol);
  const errs = good.map(p => p.err).sort((a, b) => a - b);
  const out = {
    measured: truth.walls.length, estimated: est.length, inferred: room.walls.length - est.length,
    recall: good.length / truth.walls.length, precision: est.length ? good.length / est.length : 0,
    errMedCm: errs.length ? 100 * errs[errs.length >> 1] : null,
    perWall: truth.walls.map((m, i) => { const p = match.find(q => q.i === i); return { measured: m, estimated: p ? est[p.j] : null, errCm: p ? 100 * p.err : null }; })
  };
  if (truth.phoneHeight && room.floor && Number.isFinite(room.floor.h)) out.floorErrCm = 100 * Math.abs(room.floor.h - truth.phoneHeight);
  if (truth.roomHeight && truth.phoneHeight && room.ceil && Number.isFinite(room.ceil.h)) out.ceilErrCm = 100 * Math.abs(room.ceil.h - (truth.roomHeight - truth.phoneHeight));
  return out;
}

export function fieldText(s) {
  if (!s) return 'Enter at least two measured wall distances.';
  const f = (v) => (v == null ? '-' : v.toFixed(1));
  const lines = [
    `walls found ${Math.round(s.recall * s.measured)} of ${s.measured} within 15 cm; ${s.estimated} measured by sound, ${s.inferred} inferred`,
    `median error of found walls ${f(s.errMedCm)} cm`,
    ...s.perWall.map((w, k) => `  wall ${k + 1}: tape ${w.measured.toFixed(2)} m, sound ${w.estimated == null ? '-' : w.estimated.toFixed(2) + ' m'}, error ${f(w.errCm)} cm`)
  ];
  if (s.floorErrCm != null) lines.push(`floor distance error ${f(s.floorErrCm)} cm`);
  if (s.ceilErrCm != null) lines.push(`ceiling distance error ${f(s.ceilErrCm)} cm`);
  return lines.join('\n');
}
