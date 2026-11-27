// Tape calibration (SANDBOX.md E2). Every range from one scan carries the same speed-of-sound scale (air
// temperature) and the same offset (speaker and microphone geometry, latency), so to first order a true
// distance is s x estimated + o for every wall, floor and ceiling. One taped wall fixes o; two or more taped
// walls whose distances differ by at least 0.8 m fix s and o.

import { angDiff, DEG, roomPolygon } from './geom.js';

export const CALIB = { minSpread: 0.8, sMin: 0.97, sMax: 1.03, oMax: 0.15, startTol: 30 * DEG };

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// pairs: [{ est, tape }] in metres. Returns { scale, offset, n, kind }. Time O(n).
export function fitCalibration(pairs) {
  const P = (pairs || []).filter(p => Number.isFinite(p.est) && Number.isFinite(p.tape) && p.tape > 0.2);
  if (!P.length) return null;
  const ests = P.map(p => p.est);
  const spread = Math.max(...ests) - Math.min(...ests);
  if (P.length >= 2 && spread >= CALIB.minSpread) {
    const mx = ests.reduce((a, b) => a + b, 0) / P.length, my = P.reduce((a, p) => a + p.tape, 0) / P.length;
    let sxy = 0, sxx = 0;
    for (const p of P) { sxy += (p.est - mx) * (p.tape - my); sxx += (p.est - mx) ** 2; }
    const s = clamp(sxx > 0 ? sxy / sxx : 1, CALIB.sMin, CALIB.sMax);
    const o = clamp(P.reduce((a, p) => a + p.tape - s * p.est, 0) / P.length, -CALIB.oMax, CALIB.oMax);
    return { scale: s, offset: o, n: P.length, kind: 'scale and offset' };
  }
  const diffs = P.map(p => p.tape - p.est).sort((a, b) => a - b);
  return { scale: 1, offset: clamp(diffs[diffs.length >> 1], -CALIB.oMax, CALIB.oMax), n: P.length, kind: 'offset' };
}

// Applies d' = s d + o to every wall, s to the hold radius, and s h + o to the floor and ceiling distances;
// rebuilds the plan so its edges follow the corrected walls. Returns a new room. Time O(walls^2).
export function applyCalibration(room, cal) {
  if (!room || !room.ok || !cal) return room;
  const s = cal.scale ?? 1, o = cal.offset ?? 0;
  const walls = room.walls.map(w => ({ ...w, d: s * w.d + o }));
  const pg = roomPolygon(walls);
  const fix = (H) => H ? { ...H, h: s * H.h + o } : H;
  return { ...room, walls, poly: pg.poly, edgeWall: pg.edgeWall, openEdges: pg.edgeWall.map(id => id < 0), rho: room.rho != null ? s * room.rho : room.rho, floor: fix(room.floor), ceil: fix(room.ceil), calibration: cal };
}

// Index of the measured wall whose normal is closest to a heading (the wall the user faced at the start),
// or -1 when none is within 30 deg.
export function pickStartWall(walls, yaw) {
  let best = -1, bd = Infinity;
  (walls || []).forEach((w, i) => { if (w.inferred) return; const d = angDiff(w.phi, yaw); if (d < bd) { bd = d; best = i; } });
  return bd <= CALIB.startTol ? best : -1;
}

// Set only by the sealed test (SANDBOX.md E2). Not adopted 2026-11-27: one taped wall made the other walls worse
// (median +2.0 cm, 95 % interval +0.7 to +3.5), because only 34 % of wall-error variance is shared within a room.
// The app does not offer the correction; the code stays for the field round's offline check.
export const CALIB_ENABLED = false;
export const CALIB_SAVE = false;
