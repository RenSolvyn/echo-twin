// Wall selection by exact search over wall lines with a detection likelihood (v0.4, SANDBOX.md).
//
// Each predicted echo path (wall, adjacent corner, wall or corner plus floor or ceiling bounce) is scored
// by the single-target likelihood ratio under Poisson clutter: max over Pd of sum_k log(1 - Pd + Pd g_k),
// g_k = sum_z w_z N(r_z; r_k, sigma) / lambda. Paths that predict echoes which are not there pay log(1 - Pd).
// A convex plan is a cycle of pool walls in angle order in which every wall is an edge; with wall terms
// unary and corner terms pairwise, the best cycle is found exactly by second-order dynamic programming.
import { wrap, angDiff, DEG, TAU, intersectWalls } from './geom.js';
import { EST, smooth, houghAcc, houghPeaks } from './estimate.js';

const PD_GRID = [];
for (let p = 0.02; p < 0.99; p += 0.04) PD_GRID.push(p);
const SQ2PI = Math.sqrt(2 * Math.PI);

// Candidates grouped by chirp. Time O(n). Space O(n).
export function frameTable(cands, rMin = 0.55, rMax = EST.maxRange) {
  const byK = new Map();
  let wsum = 0;
  for (const c of cands) {
    if (c.r < rMin || c.r > rMax || !(c.w > 0)) continue;
    let f = byK.get(c.k);
    if (!f) { f = { th: c.theta, om: c.omega || 0, t: c.t, r: [], w: [] }; byK.set(c.k, f); }
    f.r.push(c.r); f.w.push(c.w); wsum += c.w;
  }
  const F = [...byK.keys()].sort((a, b) => a - b).map(k => byK.get(k));
  const n = Math.max(1, F.length);
  return { F, n, rMin, rMax, lambda: Math.max(1e-6, wsum / (n * (rMax - rMin))) };
}

// Predicted range of one path in one chirp (same model as the smoother's predict()).
function centre(g, t) {
  const tr = g.traj;
  if (!tr) return [0, 0];
  const s = Math.max(0, Math.min(tr.K - 1 - 1e-9, (t - tr.t0) / tr.dt)), j = Math.floor(s), f = s - j;
  return [(1 - f) * tr.cx[j] + f * tr.cx[j + 1], (1 - f) * tr.cy[j] + f * tr.cy[j + 1]];
}

function pathRange(p, f, g) {
  const th = f.th + g.drift * (f.t - g.tMid), rho = g.rho, om = f.om, kappa = g.kappa;
  const cc = centre(g, f.t);
  if (p.type === 'wall' || p.type === 'wallH') {
    const a = th - p.phi, u = p.d - rho * Math.cos(a) - (Math.cos(p.phi) * cc[0] + Math.sin(p.phi) * cc[1]), dop = kappa * rho * om * Math.sin(a);
    if (p.type === 'wall') return u + dop;
    const R = Math.sqrt(u * u + p.h * p.h);
    return R + dop * u / R;
  }
  if (p.type === 'w2' || p.type === 'w2H') {
    // exact second-order specular path off walls a and b (image source method, monostatic): the double
    // image of the phone is p2 = Hb Ha p + t, so v = p2 - p = L p + t and the half path is |v| / 2
    const c = Math.cos(th), s = Math.sin(th), px = rho * c + cc[0], py = rho * s + cc[1], vxp = -rho * om * s, vyp = rho * om * c;
    const L = p.L, vx = L[0] * px + L[1] * py + p.t[0], vy = L[2] * px + L[3] * py + p.t[1];
    const dvx = L[0] * vxp + L[1] * vyp, dvy = L[2] * vxp + L[3] * vyp;
    const nv = Math.hypot(vx, vy), half = 0.5 * nv, rdot = (vx * dvx + vy * dvy) / (2 * nv + 1e-12);
    if (p.type === 'w2') return half + kappa * rdot;
    const R3 = Math.sqrt(half * half + p.h * p.h);
    return R3 + kappa * rdot * half / R3;
  }
  const px = rho * Math.cos(th) + cc[0], py = rho * Math.sin(th) + cc[1];
  const dx = p.X[0] - px, dy = p.X[1] - py, R = Math.hypot(dx, dy);
  const vx = -rho * om * Math.sin(th), vy = rho * om * Math.cos(th);
  const rdot = -(dx * vx + dy * vy) / R;
  if (p.type === 'corner') return R + kappa * rdot;
  const RR = Math.sqrt(R * R + p.h * p.h);
  return RR + kappa * rdot * R / RR;
}

// Second-order path off walls a then b. Reflection across n.x = d: x' = H x + 2 d n, H = I - 2 n n^T.
// The monostatic path a then b has the same length as b then a, so one path per unordered pair.
export function w2Path(a, b, h = null) {
  const na = [Math.cos(a.phi), Math.sin(a.phi)], nb = [Math.cos(b.phi), Math.sin(b.phi)];
  const H = (n) => [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[1], 1 - 2 * n[1] * n[1]];
  const Ha = H(na), Hb = H(nb);
  const M = [Hb[0] * Ha[0] + Hb[1] * Ha[2], Hb[0] * Ha[1] + Hb[1] * Ha[3], Hb[2] * Ha[0] + Hb[3] * Ha[2], Hb[2] * Ha[1] + Hb[3] * Ha[3]];
  const L = [M[0] - 1, M[1], M[2], M[3] - 1];
  const Hbna = [Hb[0] * na[0] + Hb[1] * na[1], Hb[2] * na[0] + Hb[3] * na[1]];
  const t = [2 * a.d * Hbna[0] + 2 * b.d * nb[0], 2 * a.d * Hbna[1] + 2 * b.d * nb[1]];
  return h == null ? { type: 'w2', L, t } : { type: 'w2H', L, t, h };
}

// Max over Pd of sum_k log(1 - Pd + Pd g_k). Concave in Pd, so golden-section search; chirps with g = 0
// are counted once as n0 log(1 - Pd). Time O(nonzero chirps x 18).
function bestPd(gs, withPd = false) {
  let n0 = 0; const nz = [];
  for (const v of gs) { if (v > 1e-9) nz.push(v); else n0++; }
  if (!nz.length) return withPd ? [0, 0] : 0;
  const f = (pd) => { let s = n0 * Math.log(1 - pd); for (const v of nz) s += Math.log(1 - pd + pd * v); return s; };
  let a = 0.005, b = 0.98; const r = 0.6180339887;
  let c = b - r * (b - a), d = a + r * (b - a), fc = f(c), fd = f(d);
  for (let it = 0; it < 18; it++) {
    if (fc > fd) { b = d; d = c; fd = fc; c = b - r * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + r * (b - a); fd = f(d); }
  }
  const pd = 0.5 * (a + b), best = Math.max(0, f(pd));
  return withPd ? [best, best > 0 ? pd : 0] : best;
}

// Likelihood ratio of one path on its own (no explaining away). Time O(chirps x candidates per chirp).
export function pathScore(p, T, g, sigma) {
  const inv2 = 1 / (2 * sigma * sigma), norm = 1 / (SQ2PI * sigma * T.lambda), gate = 3 * sigma;
  const gs = [];
  for (const f of T.F) {
    const pr = pathRange(p, f, g);
    if (!(pr >= T.rMin && pr <= T.rMax)) continue;
    let v = 0;
    for (let i = 0; i < f.r.length; i++) { const e = f.r[i] - pr; if (e > -gate && e < gate) v += f.w[i] * Math.exp(-e * e * inv2); }
    gs.push(v * norm);
  }
  return bestPd(gs);
}

function pathsOf(walls, hs) {
  const P = [];
  walls.forEach((w, i) => {
    P.push({ type: 'wall', phi: w.phi, d: w.d, wi: i });
    for (const h of hs) P.push({ type: 'wallH', phi: w.phi, d: w.d, h, wi: i });
  });
  // every pair of walls: exact double reflections (adjacent pairs are the corner tracks, parallel pairs the
  // constant width tracks); a pair whose track is absent costs nothing beyond its share of the penalty
  for (let i = 0; i < walls.length; i++) for (let j = i + 1; j < walls.length; j++) {
    P.push(w2Path(walls[i], walls[j]));
    for (const h of hs) P.push(w2Path(walls[i], walls[j], h));
  }
  return P;
}

// Joint score of a whole hypothesis: every candidate counts for the one path that explains it best.
// Returns penalized log-likelihood in nats. Time O(chirps x candidates x paths).
export function jointScore(walls, hs, g, T, sigma) {
  const paths = pathsOf(walls, hs), np = paths.length;
  if (!np) return -Infinity;
  const inv2 = 1 / (2 * sigma * sigma), norm = 1 / (SQ2PI * sigma * T.lambda), gate = 3 * sigma;
  const gs = paths.map(() => []);
  const pr = new Float64Array(np), acc = new Float64Array(np), valid = new Uint8Array(np);
  for (const f of T.F) {
    for (let j = 0; j < np; j++) { pr[j] = pathRange(paths[j], f, g); valid[j] = pr[j] >= T.rMin && pr[j] <= T.rMax ? 1 : 0; acc[j] = 0; }
    for (let i = 0; i < f.r.length; i++) {
      let bj = -1, bk = 0;
      for (let j = 0; j < np; j++) {
        if (!valid[j]) continue;
        const e = f.r[i] - pr[j];
        if (e <= -gate || e >= gate) continue;
        const k = Math.exp(-e * e * inv2);
        if (k > bk) { bk = k; bj = j; }
      }
      if (bj >= 0) acc[bj] += f.w[i] * bk;
    }
    for (let j = 0; j < np; j++) if (valid[j]) gs[j].push(acc[j] * norm);
  }
  let s = 0, used = 0;
  for (const v of gs) { const [ll, pd] = bestPd(v, true); s += ll; if (pd >= 0.1) used++; }
  const nParams = 2 * walls.length + used;
  return s - 0.5 * nParams * Math.log(T.n);
}

// Length of the edge of wall b between its neighbours a (before) and c (after) in angle order.
function edgeLen(a, b, c) {
  const X1 = intersectWalls(a, b), X2 = intersectWalls(b, c);
  if (!X1 || !X2) return -1;
  return (X2[0] - X1[0]) * -Math.sin(b.phi) + (X2[1] - X1[1]) * Math.cos(b.phi);
}

// Pool: Hough peaks refined on their own wall-path score, deduplicated. Time O(peaks x grid x chirps).
function buildPool(peaks, extra, T, g, sigma) {
  const pool = [];
  const seeds = peaks.filter(p => p.d >= EST.minWallDist).concat(extra.filter(Boolean));
  for (const pk of seeds) {
    let best = { phi: pk.phi, d: pk.d, s: pathScore({ type: 'wall', phi: pk.phi, d: pk.d }, T, g, sigma) };
    for (const [stepA, stepD, nA, nD] of [[1 * DEG, 0.02, 5, 4], [0.5 * DEG, 0.005, 2, 3]]) {
      const c0 = { ...best };
      for (let i = -nA; i <= nA; i++) for (let j = -nD; j <= nD; j++) {
        const phi = wrap(c0.phi + i * stepA), d = c0.d + j * stepD;
        if (d < EST.minWallDist) continue;
        const s = pathScore({ type: 'wall', phi, d }, T, g, sigma);
        if (s > best.s) best = { phi, d, s };
      }
    }
    if (pool.some(q => angDiff(q.phi, best.phi) < 4 * DEG && Math.abs(q.d - best.d) < 0.05)) continue;
    pool.push(best);
  }
  return pool.sort((a, b) => wrap(a.phi) - wrap(b.phi));
}

// Exact best cycles by second-order dynamic programming; returns the best cycle per start pair.
function dpCycles(pool, hs, T, g, sigma) {
  const P = pool.length, lnN = Math.log(T.n), minEdge = 0.4;
  const U = pool.map(w => {
    let s = pathScore({ type: 'wall', phi: w.phi, d: w.d }, T, g, sigma);
    for (const h of hs) s += pathScore({ type: 'wallH', phi: w.phi, d: w.d, h }, T, g, sigma);
    return s - 0.5 * (2 + 1 + hs.length) * lnN;
  });
  const B = Array.from({ length: P }, () => new Float64Array(P).fill(-Infinity));
  for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
    if (i === j) continue;
    let gap = wrap(pool[j].phi - pool[i].phi); if (gap <= 0) gap += TAU;
    if (gap < 25 * DEG || gap > 179 * DEG) continue;
    const X = intersectWalls(pool[i], pool[j]);
    if (!X || Math.hypot(X[0], X[1]) > 9) continue;
    let s = pathScore(w2Path(pool[i], pool[j]), T, g, sigma);
    for (const h of hs) s += pathScore(w2Path(pool[i], pool[j], h), T, g, sigma);
    B[i][j] = s - 0.5 * (1 + hs.length) * lnN;
  }
  const E = new Map();
  const edge = (a, b, c) => { const key = (a * P + b) * P + c; let v = E.get(key); if (v === undefined) { v = edgeLen(pool[a], pool[b], pool[c]); E.set(key, v); } return v; };
  const out = [];
  const val = Array.from({ length: P }, () => new Float64Array(P));
  const bp = Array.from({ length: P }, () => new Int16Array(P));
  for (let s = 0; s < P; s++) for (let w1 = s + 1; w1 < P; w1++) {
    if (!Number.isFinite(B[s][w1])) continue;
    for (const row of val) row.fill(-Infinity);
    val[s][w1] = U[s] + U[w1] + B[s][w1]; bp[s][w1] = -1;
    let best = null;
    for (let cur = w1; cur < P; cur++) for (let prev = s; prev < cur; prev++) {
      const v = val[prev][cur];
      if (!Number.isFinite(v)) continue;
      if (cur !== w1 && Number.isFinite(B[cur][s]) && edge(prev, cur, s) >= minEdge && edge(cur, s, w1) >= minEdge) {
        const tot = v + B[cur][s];
        if (!best || tot > best.v) best = { v: tot, prev, cur };
      }
      for (let n = cur + 1; n < P; n++) {
        if (!Number.isFinite(B[cur][n]) || edge(prev, cur, n) < minEdge) continue;
        const nv = v + U[n] + B[cur][n];
        if (nv > val[cur][n]) { val[cur][n] = nv; bp[cur][n] = prev; }
      }
    }
    if (!best) continue;
    const seq = [best.cur];
    let a = best.prev, b = best.cur;
    while (a >= 0) { seq.push(a); const pa = bp[a][b]; b = a; a = pa; }
    out.push({ v: best.v, idx: seq.reverse() });
  }
  return out.sort((x, y) => y.v - x.v);
}

const sameSet = (A, B) => A.length === B.length && A.every(a => B.some(b => angDiff(a.phi, b.phi) < 6 * DEG && Math.abs(a.d - b.d) < 0.08));

// ctx: { cands, rho, hs, kappa, tMid, turnDur, nD, opt, contenders: [wall arrays] }
// Returns { walls, rho, hs, drift, score, margin, alternatives, pool } or null.
export function dpSelect(ctx) {
  const t0 = Date.now();
  const sigSearch = 0.03, sigFinal = 0.025, maxSmooth = ctx.opt.fast ? 1 : 3;
  const T = frameTable(ctx.cands);
  if (T.F.length < 20) return null;
  let g = { rho: ctx.rho, drift: 0, tMid: ctx.tMid, kappa: ctx.kappa };
  let hs = ctx.hs.slice();
  const finals = [];
  const consider = (walls, rho0, hsIn) => {
    if (walls.length < 3) return;
    let sm;
    try { sm = smooth(ctx.cands, walls.map(w => ({ phi: w.phi, d: w.d, s: 1 })), rho0, { ...ctx.opt, kappa: ctx.kappa, tMid: ctx.tMid, hs: hsIn, turnDur: ctx.turnDur }); } catch (e) { return; }
    const ws = sm.walls.map(w => ({ phi: w.phi, d: w.d }));
    const hsS = sm.hs.map(h => h.h ?? h);
    const gS = { rho: sm.rho, drift: sm.drift, tMid: ctx.tMid, kappa: ctx.kappa, traj: sm.traj };
    finals.push({ walls: ws, rho: sm.rho, hs: hsS, drift: sm.drift, traj: sm.traj, score: jointScore(ws, hsS, gS, T, sigFinal) });
  };
  // the greedy answer is a contender; the other greedy starts only seed the pool (smoothing is the costly step)
  if (ctx.contenders && ctx.contenders[0] && ctx.contenders[0].length >= 3) consider(ctx.contenders[0], ctx.rho, hs);
  let poolOut = [];
  for (let pass = 0; pass < (ctx.opt.fast ? 1 : 2); pass++) {
    let peaks;
    if (pass === 0) peaks = ctx.peaks;
    else {
      const best = finals.slice().sort((a, b) => b.score - a.score)[0];
      if (!best) break;
      g = { rho: best.rho, drift: best.drift, tMid: ctx.tMid, kappa: ctx.kappa, traj: best.traj };
      hs = best.hs;
      const corr = ctx.cands.map(c => ({ ...c, theta: c.theta + best.drift * (c.t - ctx.tMid) }));
      peaks = houghPeaks(houghAcc(corr, best.rho, ctx.kappa, ctx.nD), ctx.nD, 24, 0.02);
    }
    const extra = finals.flatMap(f => f.walls).concat((ctx.contenders || []).slice(1).flat());
    const pool = buildPool(peaks, extra, T, g, sigSearch);
    poolOut = pool;
    if (pool.length < 3) continue;
    const cyc = dpCycles(pool, hs, T, g, sigSearch);
    const tops = [];
    for (const c of cyc) {
      const ws = c.idx.map(i => ({ phi: pool[i].phi, d: pool[i].d }));
      if (tops.some(t => sameSet(t.walls, ws))) continue;
      tops.push({ walls: ws, js: jointScore(ws, hs, g, T, sigSearch) });
      if (tops.length >= 10) break;
    }
    tops.sort((a, b) => b.js - a.js);
    for (const t of tops.slice(0, pass === 0 ? maxSmooth : 2)) if (!finals.some(f => sameSet(f.walls, t.walls))) consider(t.walls, g.rho, hs);
  }
  if (!finals.length) return null;
  finals.sort((a, b) => b.score - a.score);
  const best = finals[0];
  const alt = finals.find(f => !sameSet(f.walls, best.walls));
  return { ...best, margin: alt ? best.score - alt.score : null, alternatives: finals.length, poolSize: poolOut.length, ms: Date.now() - t0 };
}
