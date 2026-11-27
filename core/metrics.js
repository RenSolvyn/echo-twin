// Evaluation against simulator ground truth. The estimator frame differs from the world frame by an
// unknown heading offset (a gauge freedom), so walls are aligned by one global rotation first.

import { wrap, angDiff, DEG, iou, rotate, roomPolygon } from './geom.js';

function circMean(angles, weights) {
  let s = 0, c = 0;
  angles.forEach((a, i) => { const w = weights ? weights[i] : 1; s += w * Math.sin(a); c += w * Math.cos(a); });
  return Math.atan2(s, c);
}

function matchWalls(est, truth, rot) {
  const pairs = [];
  const usedT = new Set();
  const order = est.map((w, i) => i).sort((a, b) => (est[b].s || 0) - (est[a].s || 0));
  for (const i of order) {
    let best = -1, bc = Infinity;
    truth.forEach((t, j) => {
      if (usedT.has(j)) return;
      const da = angDiff(est[i].phi - rot, t.phi), dd = Math.abs(est[i].d - t.d);
      if (da > 25 * DEG || dd > 0.6) return;
      const cost = da / (10 * DEG) + dd / 0.15;
      if (cost < bc) { bc = cost; best = j; }
    });
    if (best >= 0) { usedT.add(best); pairs.push([i, best]); }
  }
  return pairs;
}

export function evaluate(res, truth) {
  const out = { ok: !!(res && res.ok) };
  if (!out.ok) return { ok: false, recall: 0, precision: 0, iou: 0, wallErr: [], angErr: [], nees: [] };
  // global rotation: initial guess from the strongest wall against its nearest truth by distance
  let rot = 0;
  {
    let best = Infinity;
    for (const t of truth.walls) for (const w of res.walls) {
      const r0 = wrap(w.phi - t.phi);
      const pr = matchWalls(res.walls, truth.walls, r0);
      const c = -pr.length * 10 + pr.reduce((s, [i, j]) => s + Math.abs(res.walls[i].d - truth.walls[j].d), 0);
      if (c < best) { best = c; rot = r0; }
    }
    for (let it = 0; it < 3; it++) {
      const pr = matchWalls(res.walls, truth.walls, rot);
      if (!pr.length) break;
      rot = wrap(rot + circMean(pr.map(([i, j]) => wrap(res.walls[i].phi - rot - truth.walls[j].phi))));
    }
  }
  const pairs = matchWalls(res.walls, truth.walls, rot);
  const good = pairs.filter(([i, j]) => Math.abs(res.walls[i].d - truth.walls[j].d) <= 0.15 && angDiff(res.walls[i].phi - rot, truth.walls[j].phi) <= 10 * DEG);
  const observable = truth.walls.filter(t => t.observable).length;
  const goodObs = good.filter(([, j]) => truth.walls[j].observable).length;
  out.recall = observable ? goodObs / observable : 1;
  out.precision = res.walls.length ? good.length / res.walls.length : 0;
  out.wallErr = pairs.map(([i, j]) => Math.abs(res.walls[i].d - truth.walls[j].d));
  out.angErr = pairs.map(([i, j]) => angDiff(res.walls[i].phi - rot, truth.walls[j].phi) / DEG);
  // NEES per matched wall with the marginal 2x2 covariance (phi, d)
  out.nees = [];
  if (res.P) {
    for (const [i, j] of good) {
      const wi = res.wallIndex ? res.wallIndex[i] : i;
      const o = res.wallOffset ?? 2;
      const a = o + 2 * wi, b = o + 1 + 2 * wi;
      if (!res.P[a]) continue;
      const pa = res.P[a][a], pb = res.P[b][b], pab = res.P[a][b];
      const ea = wrap(res.walls[i].phi - rot - truth.walls[j].phi), eb = res.walls[i].d - truth.walls[j].d;
      const det = pa * pb - pab * pab;
      if (det <= 0) continue;
      out.nees.push((ea * ea * pb - 2 * ea * eb * pab + eb * eb * pa) / det);
    }
  }
  const estPoly = res.poly.map(p => rotate(p, -rot));
  out.iou = iou(estPoly, truth.poly);
  out.floorErr = res.floor ? Math.abs(res.floor.h - truth.floor) : null;
  out.ceilErr = res.ceil ? Math.abs(res.ceil.h - truth.ceil) : null;
  out.rot = rot;
  out.nWalls = res.walls.length;
  out.nTrue = truth.walls.length;
  out.nObservable = observable;
  return out;
}

export function evaluateBaseline(b, truth) {
  if (!b.ok) return { iou: 0 };
  // the baseline polygon is in the IMU frame too; use the best rotation over a fine grid
  let best = 0;
  for (let a = -Math.PI; a < Math.PI; a += 2 * DEG) {
    const P = b.poly.map(p => rotate(p, a));
    // star polygon may be non-convex; use its convex hull area overlap as an upper bound for fairness
    const v = iou(convexHull(P), truth.poly);
    if (v > best) best = v;
  }
  return { iou: best };
}

export function convexHull(pts) {
  const p = pts.map(q => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], hi = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (hi.length >= 2 && cross(hi[hi.length - 2], hi[hi.length - 1], q) <= 0) hi.pop(); hi.push(q); }
  hi.pop(); lo.pop();
  return lo.concat(hi);
}

export function quantile(arr, q) {
  if (!arr.length) return NaN;
  const s = arr.slice().sort((a, b) => a - b);
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
