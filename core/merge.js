// Two standing spots (SANDBOX.md E1). Each spot is solved alone; spot B is then registered to spot A from
// the walls alone and the two convex cells are merged. The gyro keeps tracking heading while the user walks,
// so both scans share one heading frame up to a small offset (gyro drift and walking jolts).
//
// Wall convention (core/geom.js): outward normal n = (cos phi, sin phi), the cell is n . x <= d, with d measured
// from that spot's turning centre. A wall seen from spot B at offset t (in A's frame) satisfies
//   same plane heard from both spots:    d_A = d_B + n . t
//   one wall heard from both sides:      n . t = d_A + d_B + thickness   (normals opposite)
// Hypotheses come from pairs of non-parallel matches (two equations, two unknowns); each one is scored by the
// walls it explains within 10 cm minus free-space conflicts (a wall of one spot standing deep inside the closed
// cell the other spot heard as empty). A registration needs two non-parallel supporting walls and a clear
// winner; otherwise it fails and nothing is merged.

import { wrap, angDiff, DEG, rotate, pointInConvex, roomPolygon } from './geom.js';

export const MERGE = {
  // Set only by the sealed test (SANDBOX.md E1). Not adopted 2026-11-27: 0 of 50 convex and L-shape pairs
  // registered and 2 of 30 two-room pairs merged wrongly, so the app records a second spot but never merges it.
  enabled: false,
  angTol: 18 * DEG,      // normals of one plane seen from two spots (wall angles are 10 to 15 deg off today, E114)
  sd: 0.03,              // m, distance noise of one wall from one spot
  sphi: 8 * DEG,         // angle noise of one wall from one spot (E114)
  gate: 2.5,             // residual gate in sigmas
  conflictCost: 0.5,     // evidence lost per free-space conflict
  thick: 0.12, thickMin: 0.05, thickMax: 0.30, // m, wall heard from both sides
  minMove: 0.5, maxMove: 10, // m between spots
  maxHeading: 45 * DEG,  // heading offset between the two scans' frames: 3 % gyro scale error over 2 turns plus the walk
  conflictDepth: 0.30,   // m inside the other cell
  conflictRange: 6.5,    // m, free space is only evidence where it was heard
  margin: 0.8,           // second-best distinct hypothesis must score at most this fraction of the best
  minScore: 2.5          // evidence (nats) beyond the two walls that fix the walk (dev: 2.0 let one wrong merge through)
};

const nrm = (phi) => [Math.cos(phi), Math.sin(phi)];
const measured = (room) => room.walls.map((w, i) => ({ ...w, i })).filter(w => !w.inferred);

// Distance from a point to the boundary of a convex polygon (positive inside). Time O(edges).
function depthInside(poly, x, y) {
  if (!pointInConvex(poly, x, y)) return -1;
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    const ex = x1 - x0, ey = y1 - y0, L = Math.hypot(ex, ey) || 1;
    m = Math.min(m, Math.abs((x - x0) * ey - (y - y0) * ex) / L);
  }
  return m;
}

// Midpoints of the measured edges of a solved cell, in its own frame.
function wallMidpoints(room) {
  const out = [];
  const P = room.poly || [];
  for (let i = 0; i < P.length; i++) {
    const wi = room.edgeWall ? room.edgeWall[i] : -1;
    if (wi < 0 || !room.walls[wi] || room.walls[wi].inferred) continue;
    const a = P[i], b = P[(i + 1) % P.length];
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 0.4) continue;
    out.push([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
  }
  return out;
}

const toA = (p, dh, t) => { const q = rotate(p, -dh); return [q[0] + t[0], q[1] + t[1]]; };
const toB = (p, dh, t) => rotate([p[0] - t[0], p[1] - t[1]], dh);

// Residual of one B wall against one A wall under heading offset dh and translation t, or null if the
// normals do not line up. The plane relation uses the mean normal of the two walls; its angle error e moves
// n . t by about e (n_perp . t), so the expected residual grows with the walk across the normal:
//   sigma^2 = 2 sd^2 + 0.5 sphi^2 (n_perp . t)^2.
// Returns { kind: 'same' | 'back', res, sigma }.
function pairResidual(wa, wb, dh, t) {
  const phiB = wrap(wb.phi - dh);
  const same = angDiff(phiB, wa.phi) < MERGE.angTol, back = !same && angDiff(phiB, wa.phi + Math.PI) < MERGE.angTol;
  if (!same && !back) return null;
  const pb = same ? phiB : wrap(phiB + Math.PI);
  const phi = Math.atan2(Math.sin(wa.phi) + Math.sin(pb), Math.cos(wa.phi) + Math.cos(pb));
  const n = nrm(phi), perp = -n[1] * t[0] + n[0] * t[1];
  const nt = n[0] * t[0] + n[1] * t[1];
  const sigma = Math.sqrt(2 * MERGE.sd * MERGE.sd + 0.5 * MERGE.sphi * MERGE.sphi * perp * perp);
  if (same) return { kind: 'same', res: Math.abs(wa.d - wb.d - nt), sigma };
  const th = nt - wa.d - wb.d;
  return { kind: 'back', res: Math.max(0, MERGE.thickMin - th, th - MERGE.thickMax), th, sigma };
}

// Log-likelihood ratio of one matched wall against clutter spread over 3 m of residual.
const llr = (m) => Math.max(0, Math.log(3 / (m.sigma * Math.sqrt(2 * Math.PI))) - 0.5 * (m.res / m.sigma) ** 2);

// Score of a hypothesis: evidence from matched walls beyond the two most informative ones (a walk has two
// unknowns, so any two non-parallel walls fit exactly), minus free-space conflicts both ways.
// Time O(|A| |B| + edges).
function scoreHyp(A, B, wa, wb, dh, t) {
  const usedA = new Set(), matches = [];
  const opts = [];
  wb.forEach((b, j) => wa.forEach((a, i) => { const r = pairResidual(a, b, dh, t); if (r && r.res < MERGE.gate * r.sigma) opts.push({ i, j, ...r }); }));
  opts.sort((x, y) => x.res / x.sigma - y.res / y.sigma);
  const usedB = new Set();
  for (const o of opts) { if (usedA.has(o.i) || usedB.has(o.j)) continue; usedA.add(o.i); usedB.add(o.j); o.llr = llr(o); matches.push(o); }
  const ev = matches.map(m => m.llr).sort((x, y) => y - x);
  let s = ev.slice(2).reduce((a, b) => a + b, 0);
  let conflicts = 0;
  const closedA = A.closed && A.poly && A.poly.length >= 3, closedB = B.closed && B.poly && B.poly.length >= 3;
  if (closedA) for (const m of wallMidpoints(B)) {
    const p = toA(m, dh, t);
    if (Math.hypot(p[0], p[1]) < MERGE.conflictRange && depthInside(A.poly, p[0], p[1]) > MERGE.conflictDepth) conflicts++;
  }
  if (closedB) for (const m of wallMidpoints(A)) {
    const p = toB(m, dh, t);
    if (Math.hypot(p[0], p[1]) < MERGE.conflictRange && depthInside(B.poly, p[0], p[1]) > MERGE.conflictDepth) conflicts++;
  }
  s -= MERGE.conflictCost * conflicts;
  // support must include two non-parallel walls, or translation is not determined
  let spread = 0;
  for (let x = 0; x < matches.length; x++) for (let y = x + 1; y < matches.length; y++) {
    const da = angDiff(wa[matches[x].i].phi, wa[matches[y].i].phi);
    spread = Math.max(spread, Math.min(da, Math.PI - da));
  }
  return { s, matches, conflicts, determined: spread > 25 * DEG };
}

// Least-squares translation from matched planes for a fixed heading offset. Time O(matches).
function solveT(wa, wb, matches, dh) {
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
  for (const m of matches) {
    const A = wa[m.i], B = wb[m.j];
    const n = nrm(A.phi);
    const rhs = m.kind === 'same' ? A.d - B.d : A.d + B.d + MERGE.thick;
    const w = 1 / Math.max(1e-4, m.sigma ? m.sigma * m.sigma : (A.sD || 0.02) ** 2 + (B.sD || 0.02) ** 2);
    a11 += w * n[0] * n[0]; a12 += w * n[0] * n[1]; a22 += w * n[1] * n[1];
    b1 += w * n[0] * rhs; b2 += w * n[1] * rhs;
  }
  const det = a11 * a22 - a12 * a12;
  if (Math.abs(det) < 1e-9) return null;
  return [(a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det];
}

// Register room B (spot B) to room A (spot A). Returns { ok, t, dh, matches, score, second, conflicts, reason }.
// Time O(|A|^2 |B|^2 (|A| |B| + edges)); rooms have at most about 8 measured walls.
export function registerSpots(A, B) {
  const wa = measured(A), wb = measured(B);
  if (wa.length < 2 || wb.length < 2) return { ok: false, reason: 'each spot needs at least two measured walls' };
  const hyps = [];
  for (let i1 = 0; i1 < wa.length; i1++) for (let i2 = i1 + 1; i2 < wa.length; i2++) {
    const sep = angDiff(wa[i1].phi, wa[i2].phi);
    if (Math.min(sep, Math.PI - sep) < 25 * DEG) continue;
    for (let j1 = 0; j1 < wb.length; j1++) for (let j2 = 0; j2 < wb.length; j2++) {
      if (j1 === j2) continue;
      for (const k1 of ['same', 'back']) for (const k2 of ['same', 'back']) {
        const off1 = k1 === 'same' ? 0 : Math.PI, off2 = k2 === 'same' ? 0 : Math.PI;
        const e1 = wrap(wb[j1].phi - wa[i1].phi - off1), e2 = wrap(wb[j2].phi - wa[i2].phi - off2);
        if (angDiff(e1, e2) > MERGE.angTol || Math.abs(e1) > MERGE.maxHeading || Math.abs(e2) > MERGE.maxHeading) continue;
        const dh = Math.atan2(Math.sin(e1) + Math.sin(e2), Math.cos(e1) + Math.cos(e2));
        const t = solveT(wa, wb, [{ i: i1, j: j1, kind: k1 }, { i: i2, j: j2, kind: k2 }], dh);
        if (!t) continue;
        const L = Math.hypot(t[0], t[1]);
        if (L < MERGE.minMove || L > MERGE.maxMove) continue;
        hyps.push({ t, dh });
      }
    }
  }
  if (!hyps.length) return { ok: false, reason: 'no pair of walls heard from both spots' };
  // score, refine on all matches (heading from matched normals, then translation), rescore
  const scored = hyps.map(h => {
    let { t, dh } = h;
    let sc = scoreHyp(A, B, wa, wb, dh, t);
    for (let it = 0; it < 2 && sc.matches.length >= 2; it++) {
      let sx = 0, sy = 0;
      for (const m of sc.matches) { const e = wrap(wb[m.j].phi - wa[m.i].phi - (m.kind === 'same' ? 0 : Math.PI)); sx += Math.cos(e); sy += Math.sin(e); }
      const dh2 = Math.atan2(sy, sx);
      const t2 = solveT(wa, wb, sc.matches, dh2);
      if (!t2) break;
      const sc2 = scoreHyp(A, B, wa, wb, dh2, t2);
      if (sc2.s < sc.s - 1e-9) break;
      t = t2; dh = dh2; sc = sc2;
    }
    return { t, dh, ...sc };
  }).filter(h => h.determined);
  if (!scored.length) return { ok: false, reason: 'the walls heard from both spots are parallel, so the walk is not determined' };
  scored.sort((x, y) => y.s - x.s);
  const best = scored[0];
  const rival = scored.find(h => Math.hypot(h.t[0] - best.t[0], h.t[1] - best.t[1]) > 0.5 || angDiff(h.dh, best.dh) > 10 * DEG);
  const second = rival ? rival.s : 0;
  const out = { t: best.t, dh: best.dh, matches: best.matches, score: best.s, second, conflicts: best.conflicts, nHyp: scored.length };
  if (best.s < MERGE.minScore) return { ok: false, reason: 'too few walls agree between the two spots', ...out };
  if (second > MERGE.margin * best.s) return { ok: false, reason: 'two different walks explain the walls about equally well', ...out };
  return { ok: true, ...out };
}

// Merge B into A's frame with a successful registration. Matched planes are averaged (inverse variance);
// walls heard from both sides keep both faces. Returns a room the views and scorers can use directly:
// walls (A frame), cells (one convex polygon per spot), spots ([[0,0], t]), floor and ceiling from A.
export function mergeSpots(A, B, reg) {
  const { t, dh } = reg;
  const tw = (w) => { const phi = wrap(w.phi - dh); const n = nrm(phi); return { ...w, phi, d: w.d + n[0] * t[0] + n[1] * t[1] }; };
  const wa = measured(A), wb = measured(B);
  const pairOf = new Map(reg.matches.filter(m => m.kind === 'same').map(m => [wb[m.j].i, wa[m.i].i]));
  const walls = A.walls.map(w => ({ ...w, spots: [0] }));
  const bIndex = [];
  B.walls.forEach((w, j) => {
    const W = tw(w);
    const ai = pairOf.get(j);
    if (ai != null) {
      const a = walls[ai];
      const va = (a.sD || 0.02) ** 2, vb = (W.sD || 0.02) ** 2;
      const phi = Math.atan2(Math.sin(a.phi) / va + Math.sin(W.phi) / vb, Math.cos(a.phi) / va + Math.cos(W.phi) / vb);
      walls[ai] = { ...a, phi, d: (a.d / va + W.d / vb) / (1 / va + 1 / vb), sD: Math.sqrt(1 / (1 / va + 1 / vb)), spots: [0, 1] };
      bIndex[j] = ai;
    } else { bIndex[j] = walls.length; walls.push({ ...W, spots: [1] }); }
  });
  // each spot's cell rebuilt from the merged walls it heard, so a shared wall is one line in both cells
  const cells = [], cellWallIndex = [];
  for (const k of [0, 1]) {
    const idx = walls.map((w, i) => w.spots.includes(k) ? i : -1).filter(i => i >= 0);
    const sp = k ? t : [0, 0];
    // roomPolygon clips a box around the origin, so build the cell around the spot and shift it back
    const local = idx.map(i => { const w = walls[i]; return { ...w, d: w.d - (Math.cos(w.phi) * sp[0] + Math.sin(w.phi) * sp[1]) }; });
    const pg = roomPolygon(local);
    cells.push(pg.poly.map(p => [p[0] + sp[0], p[1] + sp[1]]));
    cellWallIndex.push(pg.edgeWall.map(e => e >= 0 ? idx[e] : -1));
  }
  return {
    ...A, ok: true, merged: true, walls, cells, cellWallIndex,
    cellOpen: cellWallIndex.map(m => m.map(e => e < 0)), spots: [[0, 0], t], registration: { t, dh, matches: reg.matches.length, score: reg.score, second: reg.second, conflicts: reg.conflicts },
    rhos: [A.rho, B.rho], closed: !!(A.closed && B.closed)
  };
}

// Splits a recording into spots (frame.spot, default 0) for callers that solve each spot.
export function framesBySpot(frames) {
  const by = new Map();
  for (const f of frames) { const s = f.spot || 0; if (!by.has(s)) by.set(s, []); by.get(s).push(f); }
  return [...by.keys()].sort((a, b) => a - b).map(k => by.get(k));
}

// Audit only (tools/sandbox5.mjs diagnostics): score of a given walk, refined like a hypothesis.
export function _scoreWalk(A, B, dh, t) { const wa = measured(A), wb = measured(B); return { ...scoreHyp(A, B, wa, wb, dh, t), wa, wb }; }
