// Room estimation from a guided scan (lift, then a slow turn in place).
//
// Physics: holding the phone flat and turning moves it on a circle of radius rho (the hold radius).
// A wall with outward normal angle phi at distance d from the turning axis echoes at
//   r(theta) = d - rho cos(theta - phi) + kappa rho omega sin(theta - phi)
// (last term: range-Doppler coupling of the linear FM sweep from the phone's own motion).
// Each wall is a sinusoid in heading-range space, so:
//   1. subtract the heading-invariant profile (floor, ceiling, body, ringing),
//   2. vote every echo candidate into a Hough accumulator over (phi, d) for a grid of rho,
//   3. pick walls with image-source explanations for corners, wall-floor and body ghosts,
//   4. refine everything jointly with a robust Levenberg-Marquardt smoother on S1 x R per wall
//      (a small factor graph: echo factors, corner factors, priors), with covariance,
//   5. close the plan as an intersection of wall half-planes.

import { median } from './dsp.js';
import { DEFAULTS } from './config.js';
import { wrap, angDiff, DEG, TAU, intersectWalls, roomPolygon, pointInConvex } from './geom.js';
import { zeros, cholesky, cholSolve, inverseSPD } from './linalg.js';
import { features, amplitudeScore, N_FEAT } from './model.js';
import { dpSelect } from './select.js';
import { nearFar, attachCos, cosWeight } from './stereo.js';

export const EST = {
  minRange: DEFAULTS.minRange,
  maxRange: DEFAULTS.maxRange,
  binSize: DEFAULTS.binSize,
  gateDeg: 180,        // no facing assumption: speaker directivity differs per phone (iPhone earpiece faces up and back)
  farRatio: 0.35,
  minWallDist: 0.7,    // operating limit: stand at least this far from every wall
  rhoMin: 0.12, rhoMax: 0.6, // operating limit: hold radius (phone held out from the body)
  minPlaneDist: 0.6,   // operating limit: floor and ceiling at least this far from the phone
  manhattan: true,
  manhattanTolDeg: 8,
  manhattanSigmaDeg: 1.5,
  selector: 'greedy', // 'greedy' (v0.3) or 'dp' (v0.4 exact search with detection likelihood, core/select.js)
  oddsMin: null, // confidence threshold on the hypothesis odds margin (nats), set only by the pre-registered calibration
  trajKnots: 0, // centre-of-rotation path knots in the smoother; 7 tested on dev rooms (recall unchanged, wall error 1.4 to 3.6 cm), so off
  candidates: 'persistence', // 'model' or 'persistence'; persistence adopted 2026-11-27 (SANDBOX.md front-end experiment and end-to-end check)
  persistS0: 2 // frozen 2026-11-27: best median IoU of {2, 4, 8} on own-sim validation (0.609, 0.600, 0.582)
};

// ---------- heading-invariant profile and candidates ----------

export function invariantProfile(frames) {
  const n = frames[0].profile.length, K = frames.length;
  const m = new Float32Array(n);
  const col = new Float64Array(K);
  for (let j = 0; j < n; j++) {
    for (let k = 0; k < K; k++) col[k] = frames[k].profile[j];
    col.sort();
    m[j] = col[K >> 1];
  }
  return m;
}

function peaksAbove(p, iMin, iMax, thr, maxPeaks) {
  const out = [];
  iMin = Math.max(1, iMin); iMax = Math.min(p.length - 2, iMax);
  for (let i = iMin; i <= iMax; i++) {
    const v = p[i];
    if (v > thr && v > p[i - 1] && v >= p[i + 1]) {
      const a = p[i - 1], c = p[i + 1], den = a - 2 * v + c;
      const fr = den !== 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) : 0;
      out.push({ i: i + fr, bin: i, amp: v });
    }
  }
  out.sort((x, y) => y.amp - x.amp);
  const kept = [];
  for (const pk of out) {
    if (kept.every(k => Math.abs(k.i - pk.i) >= 2)) kept.push(pk);
    if (kept.length >= maxPeaks) break;
  }
  return kept;
}

export function headingRates(frames) {
  for (let k = 0; k < frames.length; k++) {
    const a = frames[Math.max(0, k - 1)], b = frames[Math.min(frames.length - 1, k + 1)];
    const dt = b.t - a.t;
    frames[k].omega = dt > 0 ? (b.yaw - a.yaw) / dt : 0;
  }
}

// Heading-local background: per range bin, the median over frames within +-halfDeg of heading.
// Walls change range with heading (hold-radius aperture) so they survive; floor, ceiling, body and
// ringing keep their range and only change loudness slowly with heading, so they are removed.
// Time: O(frames x window x bins).
export function localBackground(frames, halfDeg = 45) {
  const K = frames.length, n = frames[0].profile.length;
  const half = halfDeg * DEG;
  const bg = new Array(K);
  const win = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    const idx = [];
    for (let j = 0; j < K; j++) {
      const dy = Math.abs(wrap(frames[j].yaw - frames[k].yaw));
      if (dy <= half) idx.push(j);
    }
    const b = new Float32Array(n);
    const m = idx.length;
    for (let i = 0; i < n; i++) {
      for (let q = 0; q < m; q++) win[q] = frames[idx[q]].profile[i];
      const w = win.subarray(0, m); w.sort();
      b[i] = w[m >> 1];
    }
    bg[k] = b;
  }
  return bg;
}

// Echo candidates in the residual (heading-varying) part of each profile.
// Time: O(frames x bins). opt.scorer(features) -> score in [0,1], or the amplitude baseline.
export function extractCandidates(frames, inv, opt) {
  const binSize = opt.binSize ?? EST.binSize;
  const iMin = Math.round((opt.minRange ?? EST.minRange) / binSize);
  const iMax = Math.round((opt.maxRange ?? EST.maxRange) / binSize);
  const out = [];
  const feat = new Float32Array(N_FEAT);
  for (let k = 0; k < frames.length; k++) {
    const f = frames[k], raw = f.profile;
    const ref = Array.isArray(inv) ? inv[k] : inv;
    const res = new Float32Array(raw.length);
    for (let j = 0; j < raw.length; j++) { const v = raw[j] - ref[j]; res[j] = v > 0 ? v : 0; }
    const floor = median(raw, iMin, iMax + 1) || 1e-9;
    const pks = peaksAbove(res, iMin, iMax, floor * 2.5, 14);
    const frameMax = pks.length ? pks[0].amp : 1;
    pks.forEach((pk, rank) => {
      const ctx = { floor, frameMax, rank, bwKHz: opt.bwKHz ?? 4, nBins: raw.length };
      const needFeat = opt.scorer || opt.keepFeatures;
      if (needFeat) features(raw, res, pk, ctx, feat);
      const score = opt.scorer ? opt.scorer(feat) : amplitudeScore(pk, ctx);
      const c = { k, t: f.t, theta: f.yaw, omega: f.omega || 0, r: pk.i * binSize, amp: pk.amp, score, ampScore: amplitudeScore(pk, ctx) };
      if (opt.keepFeatures) c.feat = Float32Array.from(feat);
      out.push(c);
    });
  }
  return out;
}

// 0-dimensional superlevel persistence of a 1D signal over [iMin, iMax]: each local maximum dies when its
// component merges into an older (higher) one; persistence = birth height - merge height. By the stability
// theorem, noise of sup-norm e moves every persistence by at most 2e, so ranking by persistence over the
// noise level is level-free. Measured best of five detectors on own-sim and independent data (SANDBOX.md).
// Time O(n log n). Space O(n).
export function persistencePeaks(p, iMin, iMax, maxPeaks = 14) {
  const n = p.length, order = [];
  for (let i = Math.max(1, iMin); i <= Math.min(n - 2, iMax); i++) order.push(i);
  order.sort((a, b) => p[b] - p[a]);
  const parent = new Int32Array(n).fill(-1), birth = new Float64Array(n), peakOf = new Int32Array(n);
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const pers = [];
  for (const i of order) {
    parent[i] = i; birth[i] = p[i]; peakOf[i] = i;
    for (const j of [i - 1, i + 1]) {
      if (j < 0 || j >= n || parent[j] < 0) continue;
      let a = find(i), b = find(j);
      if (a === b) continue;
      if (birth[a] < birth[b]) [a, b] = [b, a];
      pers.push({ i: peakOf[b], v: birth[b] - p[i] });
      parent[b] = a;
    }
  }
  const roots = new Set(order.filter(i => parent[i] >= 0).map(i => find(i)));
  let lo = Infinity; for (const i of order) lo = Math.min(lo, p[i]);
  for (const r of roots) pers.push({ i: peakOf[r], v: birth[r] - lo });
  pers.sort((a, b) => b.v - a.v);
  return pers.slice(0, maxPeaks);
}

function robustSigma(arr) {
  const a = Array.from(arr).sort((x, y) => x - y), m = a[a.length >> 1];
  const d = a.map(v => Math.abs(v - m)).sort((x, y) => x - y);
  return 1.4826 * d[d.length >> 1] || 1e-12;
}

// Candidates from persistence peaks of the heading-varying residual, scored by persistence over a robust
// noise level (median absolute deviation of profile minus background). Time O(frames x bins log bins).
export function persistenceCandidates(frames, bg, opt = {}) {
  const binSize = opt.binSize ?? EST.binSize;
  const iMin = Math.round((opt.minRange ?? EST.minRange) / binSize);
  const iMax = Math.round((opt.maxRange ?? EST.maxRange) / binSize);
  const out = [];
  frames.forEach((f, k) => {
    const raw = f.profile, b = Array.isArray(bg) ? bg[k] : bg;
    const res = new Float32Array(raw.length), dif = [];
    for (let j = 0; j < raw.length; j++) {
      res[j] = Math.max(0, raw[j] - b[j]);
      if (j >= iMin && j <= iMax) dif.push(raw[j] - b[j]);
    }
    const sg = robustSigma(dif);
    for (const pk of persistencePeaks(res, iMin, iMax, opt.maxPeaks ?? 14)) {
      out.push({ k, t: f.t, theta: f.yaw, omega: f.omega || 0, r: pk.i * binSize, amp: raw[pk.i], score: pk.v / sg });
    }
  });
  return out;
}

// ---------- floor and ceiling ----------

// Floor and ceiling from the lift. The IMU gives the shape of the vertical motion dz(t) (relative to the
// end of the lift, from zero-velocity-corrected integration); the floor echo must follow r = h - k dz(t)
// and the ceiling r = h + k dz(t), with k near 1. Matched search over (h, k) on prominence-normalised
// profiles, so crossing tracks and ringing ladders do not matter. Falls back to the turn-phase bands.
function prominence(p, binSize) {
  const W = Math.round(0.15 / binSize), out = new Float32Array(p.length);
  const tmp = [];
  for (let i = 0; i < p.length; i++) {
    tmp.length = 0;
    for (let j = Math.max(0, i - W); j <= Math.min(p.length - 1, i + W); j += 2) tmp.push(p[j]);
    tmp.sort((a, b) => a - b);
    out[i] = p[i] / (tmp[tmp.length >> 1] || 1e-9);
  }
  return out;
}

// Mean log-amplitude (above clutter) along r(t) = h + sign * k * (dz(t) - dz_end) over the lift, best k.
export function liftTrackScore(liftFrames, h, sign, binSize = EST.binSize) {
  const lf = liftFrames.filter(f => Number.isFinite(f.dz)).sort((a, b) => a.t - b.t);
  if (lf.length < 6) return null;
  const span = Math.max(...lf.map(f => f.dz)) - Math.min(...lf.map(f => f.dz));
  if (span < 0.08) return null;
  const nb = lf[0].profile.length;
  const B = new Float32Array(nb), col = new Float64Array(lf.length);
  for (let i = 0; i < nb; i++) { for (let q = 0; q < lf.length; q++) col[q] = lf[q].profile[i]; col.sort(); B[i] = col[lf.length >> 1]; }
  const iMinR = Math.round(EST.minRange / binSize), iMaxR = Math.round(EST.maxRange / binSize);
  const fls = lf.map(f => median(f.profile, iMinR, iMaxR + 1) || 1e-9);
  const zEnd = lf[lf.length - 1].dz;
  let best = 0;
  for (let k = 0.7; k <= 1.31; k += 0.05) {
    let sc = 0;
    for (let q = 0; q < lf.length; q++) {
      const c = Math.round((h + sign * k * (lf[q].dz - zEnd)) / binSize);
      let m = 0;
      for (let j = c - 2; j <= c + 2; j++) if (j >= 0 && j < nb) { const v = Math.log10(lf[q].profile[j] / (B[j] + fls[q]) + 1e-9); if (v > m) m = v; }
      sc += Math.min(m, 2);
    }
    if (sc / lf.length > best) best = sc / lf.length;
  }
  return best;
}

export function horizontalPlanes(liftFrames, inv, opt = {}) {
  const binSize = opt.binSize ?? EST.binSize;
  const result = { floor: null, ceil: null, labeled: false, consistent: null, bands: [] };
  const lo = Math.round(0.6 / binSize), hi = Math.round(3.2 / binSize);
  const invP = prominence(inv, binSize);
  const refine = (h) => {
    const c = Math.round(h / binSize);
    let bi = c, bv = -1;
    for (let i = c - 4; i <= c + 4; i++) if (i > 0 && i < inv.length - 1 && invP[i] > bv && inv[i] >= inv[i - 1] && inv[i] >= inv[i + 1]) { bv = invP[i]; bi = i; }
    return bv > 1.3 ? bi * binSize : h;
  };
  const lf = liftFrames.filter(f => Number.isFinite(f.dz)).sort((a, b) => a.t - b.t);
  const dzSpan = lf.length ? Math.max(...lf.map(f => f.dz)) - Math.min(...lf.map(f => f.dz)) : 0;
  result.liftSpan = dzSpan;
  if (lf.length >= 8 && dzSpan > 0.12) {
    const iMinR = Math.round(EST.minRange / binSize), iMaxR = Math.round(EST.maxRange / binSize);
    // log amplitude above each frame's noise floor, capped
    // per-range median over the lift: whatever does not move with the phone's height (body, ringing)
    const nb = lf[0].profile.length;
    const B = new Float32Array(nb), col = new Float64Array(lf.length);
    for (let i = 0; i < nb; i++) { for (let q = 0; q < lf.length; q++) col[q] = lf[q].profile[i]; col.sort(); B[i] = col[lf.length >> 1]; }
    const L = lf.map(f => {
      const fl = median(f.profile, iMinR, iMaxR + 1) || 1e-9;
      const o = new Float32Array(nb);
      for (let i = 0; i < nb; i++) { const v = Math.log10(f.profile[i] / (B[i] + fl) + 1e-9); o[i] = v < 0 ? 0 : v > 2 ? 2 : v; }
      return o;
    });
    const zEnd = lf[lf.length - 1].dz;
    const search = (sign) => {
      let best = { s: -1 };
      const all = [];
      for (let i = lo; i <= hi; i++) {
        const h = i * binSize;
        let bh = { s: -1 };
        for (let k = 0.7; k <= 1.31; k += 0.1) {
          let sc = 0;
          for (let q = 0; q < lf.length; q++) {
            const c = Math.round((h + sign * k * (lf[q].dz - zEnd)) / binSize);
            let m = 0;
            for (let j = c - 2; j <= c + 2; j++) if (j >= 0 && j < L[q].length && L[q][j] > m) m = L[q][j];
            sc += m;
          }
          sc /= lf.length;
          if (sc > bh.s) bh = { s: sc, h, k };
        }
        all.push(bh.s);
        if (bh.s > best.s) best = bh;
      }
      all.sort((a, b) => a - b);
      best.gain = best.s - all[all.length >> 1];
      return best;
    };
    const fl = search(1), ce = search(-1);
    const mk = (b) => ({ h: refine(b.h), sigma: 0.012 + 0.009 * b.h, gain: b.gain, k: b.k });
    if (fl.gain > 0.25) result.floor = mk(fl);
    if (ce.gain > 0.25) result.ceil = mk(ce);
    result.labeled = !!(result.floor || result.ceil);
    if (result.floor && result.ceil) result.consistent = Math.abs(fl.k - ce.k) <= 0.2;
  }
  // turn-phase bands (azimuth-invariant, prominent) as candidates and as a fallback
  for (let i = lo; i <= hi; i++) if (inv[i] > inv[i - 1] && inv[i] >= inv[i + 1] && invP[i] > 1.6) result.bands.push({ h: i * binSize, amp: inv[i], prom: invP[i] });
  result.bands.sort((a, b) => b.amp * b.prom - a.amp * a.prom);
  result.bands = result.bands.filter((b, i, arr) => arr.findIndex(c => Math.abs(c.h - b.h) < 0.08) === i).slice(0, 6);
  if (!result.floor && result.bands[0]) result.floor = { h: result.bands[0].h, sigma: 0.03, assumed: true };
  if (!result.ceil) { const b = result.bands.find(b => !result.floor || Math.abs(b.h - result.floor.h) > 0.08); if (b) result.ceil = { h: b.h, sigma: 0.03, assumed: true }; }
  return result;
}

// ---------- Hough over (phi, d) for a given hold radius ----------

export const GRID = { nPhi: 360, d0: 0.2, dStep: 0.01 };
const JMAX = 180;
const COSJ = new Float64Array(2 * JMAX + 1), SINJ = new Float64Array(2 * JMAX + 1);
for (let j = -JMAX; j <= JMAX; j++) { COSJ[j + JMAX] = Math.cos(j * DEG); SINJ[j + JMAX] = Math.sin(j * DEG); }

export function houghAcc(cands, rho, kappa, nD) {
  const { nPhi, d0, dStep } = GRID;
  const acc = new Float32Array(nPhi * nD);
  // anchor the 1 deg grid to the data, not to absolute heading, so a constant IMU offset shifts the
  // accumulator exactly instead of re-rounding every vote (heading equivariance, E102)
  const anchor = cands.length ? (((cands[0].theta / DEG) % 1) + 1) % 1 : 0;
  for (const c of cands) {
    const w = c.w;
    if (!(w > 0)) continue;
    const base = c.theta / DEG;
    for (let j = -JMAX; j <= JMAX; j++) {
      const ca = COSJ[j + JMAX], sa = SINJ[j + JMAX];
      const d = c.r + rho * ca - kappa * rho * c.omega * sa;
      const x = (d - d0) / dStep;
      const i0 = Math.floor(x);
      if (i0 < 0 || i0 >= nD - 1) continue;
      const fr = x - i0;
      let pb = Math.round(base - anchor - j) % nPhi;
      if (pb < 0) pb += nPhi;
      // two microphones (core/stereo.js): weight the vote by agreement of the measured echo cosine with cos(j)
      const g = c.cos != null ? w * cosWeight(c.cos, ca) : w;
      const o = pb * nD + i0;
      acc[o] += g * (1 - fr); acc[o + 1] += g * fr;
    }
  }
  const out = deridge(blur(acc, nPhi, nD), nPhi, nD);
  out.anchor = anchor;
  return out;
}

// A wall is a localized peak in phi; anything constant across phi at one d (leftovers of floor,
// ceiling, body and ringing) is a ridge. Subtract each range row's median over phi.
export function deridge(acc, nPhi, nD) {
  const col = new Float32Array(nPhi);
  for (let i = 0; i < nD; i++) {
    for (let p = 0; p < nPhi; p++) col[p] = acc[p * nD + i];
    col.sort();
    const m = col[nPhi >> 1];
    for (let p = 0; p < nPhi; p++) { const v = acc[p * nD + i] - m; acc[p * nD + i] = v > 0 ? v : 0; }
  }
  return acc;
}

export function blur(acc, nPhi, nD) {
  const k = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16];
  const tmp = new Float32Array(acc.length), out = new Float32Array(acc.length);
  for (let p = 0; p < nPhi; p++) {
    for (let i = 0; i < nD; i++) {
      let s = 0;
      for (let q = -2; q <= 2; q++) { const ii = i + q; if (ii >= 0 && ii < nD) s += k[q + 2] * acc[p * nD + ii]; }
      tmp[p * nD + i] = s;
    }
  }
  for (let p = 0; p < nPhi; p++) {
    for (let i = 0; i < nD; i++) {
      let s = 0;
      for (let q = -2; q <= 2; q++) { const pp = (p + q + nPhi) % nPhi; s += k[q + 2] * tmp[pp * nD + i]; }
      out[p * nD + i] = s;
    }
  }
  return out;
}

export function houghPeaks(acc, nD, maxPeaks = 24, relThr = 0.04) {
  const { nPhi, d0, dStep } = GRID;
  let mx = 0;
  for (let i = 0; i < acc.length; i++) if (acc[i] > mx) mx = acc[i];
  const thr = mx * relThr;
  const loc = [];
  for (let p = 0; p < nPhi; p++) {
    for (let i = 1; i < nD - 1; i++) {
      const v = acc[p * nD + i];
      if (v < thr) continue;
      let ok = true;
      for (let dp = -1; dp <= 1 && ok; dp++) for (let di = -1; di <= 1; di++) {
        if (!dp && !di) continue;
        const pp = (p + dp + nPhi) % nPhi;
        if (acc[pp * nD + i + di] > v) { ok = false; break; }
      }
      if (ok) loc.push({ p, i, v });
    }
  }
  loc.sort((a, b) => b.v - a.v);
  const kept = [];
  for (const L of loc) {
    let sup = false;
    for (const K of kept) {
      const dp = Math.min(Math.abs(L.p - K.p), nPhi - Math.abs(L.p - K.p));
      if (dp <= 18 && Math.abs(L.i - K.i) <= 10) { sup = true; break; }
    }
    if (sup) continue;
    // sub-bin refinement
    const at = (p, i) => acc[((p + nPhi) % nPhi) * nD + i];
    const a = at(L.p - 1, L.i), b = L.v, c = at(L.p + 1, L.i);
    const fp = (a - 2 * b + c) !== 0 ? 0.5 * (a - c) / (a - 2 * b + c) : 0;
    const e = at(L.p, L.i - 1), f = at(L.p, L.i + 1);
    const fi = (e - 2 * b + f) !== 0 ? 0.5 * (e - f) / (e - 2 * b + f) : 0;
    kept.push({ phi: wrap((L.p + (acc.anchor || 0) + Math.max(-0.5, Math.min(0.5, fp))) * DEG), d: d0 + (L.i + Math.max(-0.5, Math.min(0.5, fi))) * dStep, s: b, p: L.p, i: L.i });
    if (kept.length >= maxPeaks) break;
  }
  return kept;
}

export function bestRho(cands, kappa, nD) {
  const score = (rho) => {
    const acc = houghAcc(cands, rho, kappa, nD);
    const pk = houghPeaks(acc, nD, 6, 0.05);
    return { rho, s: pk.reduce((a, p) => a + p.s, 0), acc, pk };
  };
  const curve = [];
  let best = null;
  for (let rho = EST.rhoMin; rho <= EST.rhoMax + 1e-9; rho += 0.04) {
    const r = score(rho); curve.push([rho, r.s]);
    if (!best || r.s > best.s) best = r;
  }
  const r0 = best.rho;
  for (let rho = Math.max(EST.rhoMin, r0 - 0.03); rho <= Math.min(EST.rhoMax, r0 + 0.03) + 1e-9; rho += 0.01) {
    if (Math.abs(rho - r0) < 1e-6) continue;
    const r = score(rho); curve.push([rho, r.s]);
    if (r.s > best.s) best = r;
  }
  curve.sort((a, b) => a[0] - b[0]);
  return { rho: best.rho, acc: best.acc, curve };
}

// ---------- per-track model test: flat reflector versus elevated (floor or ceiling) bounce ----------
// A first-order wall echoes at r = d - rho cos(a); a wall-plus-floor bounce at r = sqrt((d - rho cos a)^2 + h^2).
// Both are near-sinusoids; the elevated one has a smaller effective swing. Fit both and compare support.

function trackSupport(cands, m, rho, tMid, gate = 0.035) {
  let s = 0, n = 0;
  for (const c of cands) {
    const a = c.theta - m.phi;
    if (Math.abs(wrap(a)) > EST.gateDeg * DEG) continue;
    const u = m.d - rho * Math.cos(a);
    const e = c.r - Math.sqrt(u * u + m.h * m.h);
    if (Math.abs(e) < gate) { s += c.w * (1 - (e / gate) ** 2); n++; }
  }
  return { s, n };
}

function fitTrack(cands, init, rho, tMid, freeH) {
  let m = { ...init };
  for (let it = 0; it < 8; it++) {
    const gate = it < 2 ? 0.07 : 0.04;
    let A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = [0, 0, 0];
    let cnt = 0;
    for (const c of cands) {
      const a = c.theta - m.phi;
      if (Math.abs(wrap(a)) > EST.gateDeg * DEG) continue;
      const u = m.d - rho * Math.cos(a);
      const R = Math.sqrt(u * u + m.h * m.h) || 1e-9;
      const e = c.r - R;
      if (Math.abs(e) > gate) continue;
      const J = [-(u / R) * rho * Math.sin(a), u / R, freeH ? m.h / R : 0];
      const w = c.w;
      for (let i = 0; i < 3; i++) { g[i] += w * J[i] * e; for (let k = 0; k < 3; k++) A[i][k] += w * J[i] * J[k]; }
      cnt++;
    }
    if (cnt < 6) break;
    const n = freeH ? 3 : 2;
    for (let i = 0; i < n; i++) A[i][i] += 1e-6 + 1e-3 * A[i][i];
    const L = cholesky(A.slice(0, n).map(r => Float64Array.from(r.slice(0, n))));
    if (!L) break;
    const dx = cholSolve(L, g.slice(0, n));
    m.phi = wrap(m.phi + dx[0]); m.d += dx[1];
    if (freeH) m.h = Math.max(0, Math.min(3.5, m.h + dx[2]));
    if (Math.abs(dx[1]) < 1e-4 && Math.abs(dx[0]) < 1e-4) break;
  }
  return { ...m, ...trackSupport(cands, m, rho, tMid) };
}

export function classifyPeaks(cands, peaks, rho, tMid, hHints = []) {
  const out = [];
  for (const pk of peaks) {
    const flat = fitTrack(cands, { phi: pk.phi, d: pk.d, h: 0 }, rho, tMid, false);
    let best = null;
    const hs = [0.8, 1.2, 1.6, 2.0].concat(hHints);
    for (const h0 of hs) {
      if (pk.d * pk.d - h0 * h0 < 0.25) continue;
      const el = fitTrack(cands, { phi: pk.phi, d: Math.sqrt(pk.d * pk.d - h0 * h0), h: h0 }, rho, tMid, true);
      if (!best || el.s > best.s) best = el;
    }
    const elevated = best && best.h > 0.45 && best.s > 1.2 * flat.s && best.n >= 8;
    out.push({ ...pk, flat, elev: best, elevated, phi: elevated ? pk.phi : flat.phi, d: elevated ? pk.d : flat.d, s: pk.s });
  }
  return out;
}

// ---------- analysis by synthesis: choose walls by how much of the echo cloud their image sources explain ----------
// For a hypothesis (walls, horizontal distances hs, rho), every candidate echo is explained by the nearest
// predicted track among: first-order wall, wall + horizontal plane, adjacent corner, corner + horizontal plane.
// Score = sum of candidate weights times a Gaussian of the residual. Walls are added greedily by gain.

export function packCands(cands, maxN = 1100) {
  // wall evidence only beyond the body's near field (operating limit: stand about a metre from walls)
  const s = cands.filter(c => c.r >= 0.55).sort((a, b) => b.w - a.w).slice(0, maxN);
  return { th: s.map(c => c.theta), r: s.map(c => c.r), w: s.map(c => c.w), om: s.map(c => c.omega), n: s.length, total: s.reduce((a, c) => a + c.w, 0) };
}

function tracksFor(walls) {
  const pairs = adjacentPairs(walls);
  const corners = [];
  for (const [i, j] of pairs) { const X = intersectWalls(walls[i], walls[j]); if (X) corners.push(X); }
  return corners;
}

export function explainScore(C, walls, hs, rho, sig = 0.025) {
  const corners = tracksFor(walls);
  const gate = EST.gateDeg * DEG, inv2 = 1 / (2 * sig * sig);
  let score = 0;
  for (let q = 0; q < C.n; q++) {
    const th = C.th[q], r = C.r[q];
    let best = Infinity;
    for (const w of walls) {
      const a = wrap(th - w.phi);
      if (Math.abs(a) > gate) continue;
      const u = w.d - rho * Math.cos(a);
      let e = Math.abs(r - u); if (e < best) best = e;
      for (const h of hs) { e = Math.abs(r - Math.sqrt(u * u + h * h)); if (e < best) best = e; }
    }
    if (corners.length) {
      const px = rho * Math.cos(th), py = rho * Math.sin(th);
      for (const X of corners) {
        if (angDiff(th, Math.atan2(X[1], X[0])) > gate) continue;
        const R = Math.hypot(X[0] - px, X[1] - py);
        let e = Math.abs(r - R); if (e < best) best = e;
        for (const h of hs) { e = Math.abs(r - Math.sqrt(R * R + h * h)); if (e < best) best = e; }
      }
    }
    if (best < 3 * sig) score += C.w[q] * Math.exp(-best * best * inv2);
  }
  return score;
}

// Fraction of frames facing a wall (within 25 degrees) that show an echo on its first-order track.
// Real walls echo when faced; aperture sidelobes and bounce ghosts predict echoes that are not there.
export function apexCoverage(C, w, rho, frameHeadings) {
  let faced = 0, hit = 0;
  for (const th of frameHeadings) {
    if (angDiff(th, w.phi) > 25 * DEG) continue;
    faced++;
  }
  if (!faced) return 1;
  const seen = new Set();
  for (let q = 0; q < C.n; q++) {
    const a = wrap(C.th[q] - w.phi);
    if (Math.abs(a) > 25 * DEG) continue;
    if (Math.abs(C.r[q] - (w.d - rho * Math.cos(a))) < 0.03) seen.add(C.th[q]);
  }
  hit = seen.size;
  return hit / faced;
}

// Number of open sides of a plan, invariant to rotation: 0 when closed, else one per started quarter turn
// of the largest normal gap beyond 180 deg. Time O(n log n).
export function openSides(ws) {
  if (ws.length < 2) return 4;
  const a = ws.map(w => wrap(w.phi)).sort((x, y) => x - y);
  let gap = 0;
  for (let i = 0; i < a.length; i++) gap = Math.max(gap, i + 1 < a.length ? a[i + 1] - a[i] : a[0] + TAU - a[i]);
  return gap < Math.PI - 1e-9 ? 0 : 1 + Math.floor((gap - Math.PI) / (Math.PI / 2) + 1e-9);
}

// Local search over wall sets: add, swap (replace a wall within 25 degrees) and remove moves,
// first improvement by explained-echo score minus a per-wall cost, until no move helps.
export function greedyWalls(cands, peaks, hs, rho, opt = {}) {
  const C = opt.C || packCands(cands);
  const pool = peaks.filter(p => p.d >= EST.minWallDist).slice(0, opt.maxPeaks ?? 16);
  const wMean = C.total / Math.max(1, C.n);
  const cost = Math.max((opt.minGainFrac ?? 0.04) * C.total, 6 * wMean);
  // every wall must be a real edge (>= minEdge) of the convex plan: corners and bounce tracks posing
  // as walls end up tangent to the plan and are rejected
  const minEdge = opt.minEdge ?? 0.4;
  const openCost = opt.openCost ?? 1.0;
  let lastOpen = 0;
  const valid = (ws) => {
    lastOpen = 4;
    if (ws.length < 3) return true;
    const pg = roomPolygon(ws);
    // openness from the largest gap between wall normals (the plan is bounded iff every gap < 180 deg).
    // Counting edges of the clipping box instead made the cost depend on absolute heading (E102).
    lastOpen = openSides(ws);
    const L = new Float64Array(ws.length);
    for (let i = 0; i < pg.poly.length; i++) {
      const id = pg.edgeWall[i];
      if (id < 0) continue;
      const a = pg.poly[i], b = pg.poly[(i + 1) % pg.poly.length];
      L[id] += Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    return Array.prototype.every.call(L, v => v >= minEdge);
  };
  // you are inside a room: every open side of the plan costs as much as a wall
  const val = (ws) => valid(ws) ? explainScore(C, ws, hs, rho) - cost * ws.length - openCost * cost * lastOpen : -Infinity;
  let walls = opt.init ? opt.init.slice() : [];
  let cur = val(walls);
  const log = [];
  for (let pass = 0; pass < 24; pass++) {
    let best = null;
    for (const pk of pool) {
      if (walls.includes(pk)) continue;
      const near = walls.filter(w => angDiff(w.phi, pk.phi) < 25 * DEG);
      if (near.length === 0 && walls.length < 8) {
        const v = val(walls.concat([pk]));
        if (v > cur + 1e-9 && (!best || v > best.v)) best = { v, ws: walls.concat([pk]), move: 'add', pk };
      }
      for (const w of near) {
        const ws = walls.filter(x => x !== w).concat([pk]);
        const v = val(ws);
        if (v > cur + 1e-9 && (!best || v > best.v)) best = { v, ws, move: 'swap', pk };
      }
    }
    for (const w of walls) {
      const ws = walls.filter(x => x !== w);
      const v = val(ws);
      if (v > cur + 1e-9 && (!best || v > best.v)) best = { v, ws, move: 'remove', pk: w };
    }
    if (!best) break;
    walls = best.ws; cur = best.v;
    log.push({ move: best.move, phi: best.pk.phi, d: best.pk.d });
  }
  return { walls, score: explainScore(C, walls, hs, rho) / C.total, val: cur / C.total, log };
}

// ---------- wall selection with image-source explanations ----------

function explainPeak(pk, walls, hs) {
  for (let i = 0; i < walls.length; i++) {
    for (let j = i + 1; j < walls.length; j++) {
      const sep = angDiff(walls[i].phi, walls[j].phi);
      if (sep < 35 * DEG || sep > 145 * DEG) continue;
      const X = intersectWalls(walls[i], walls[j]);
      if (!X) continue;
      const R = Math.hypot(X[0], X[1]), psi = Math.atan2(X[1], X[0]);
      if (angDiff(pk.phi, psi) < 10 * DEG) {
        if (Math.abs(pk.d - R) < 0.15) return 'corner';
        for (const h of hs) if (Math.abs(pk.d - Math.hypot(R, h)) < 0.15) return 'tri';
      }
    }
  }
  for (const w of walls) {
    if (angDiff(pk.phi, w.phi) > 12 * DEG) continue;
    for (const h of hs) if (Math.abs(pk.d - Math.hypot(w.d, h)) < 0.12) return 'wallH';
    if (Math.abs(pk.d - (2 * w.d + 0.1)) < 0.2) return 'ghost';
  }
  return null;
}

export function selectWalls(peaks, hs, opt = {}) {
  const farRatio = opt.farRatio ?? EST.farRatio;
  const walls = [], explained = [], objects = [];
  if (!peaks.length) return { walls, explained, objects };
  const s0 = peaks[0].s;
  const pool = peaks.filter(p => p.s >= 0.06 * s0);
  for (const pk of pool) {
    const ex = explainPeak(pk, walls, hs);
    if (ex) { explained.push({ ...pk, type: ex }); continue; }
    const near = walls.findIndex(w => angDiff(w.phi, pk.phi) < 25 * DEG);
    if (near >= 0) {
      const w = walls[near];
      if (pk.d > w.d + 0.25 && pk.s >= farRatio * w.s) { objects.push(w); walls[near] = pk; }
      else objects.push(pk);
      continue;
    }
    if (walls.length < 8) walls.push(pk);
  }
  // a wall that is itself the corner of two other walls is a corner
  for (let pass = 0; pass < 2; pass++) {
    for (let i = walls.length - 1; i >= 0; i--) {
      const others = walls.filter((_, j) => j !== i);
      const ex = explainPeak(walls[i], others, hs);
      if (ex) { explained.push({ ...walls[i], type: ex }); walls.splice(i, 1); }
    }
  }
  // fill wide angular gaps with weaker unexplained peaks
  const sorted = () => walls.slice().sort((a, b) => wrap(a.phi) - wrap(b.phi));
  for (let pass = 0; pass < 3; pass++) {
    const ws = sorted();
    let added = false;
    for (let i = 0; i < ws.length && ws.length >= 2; i++) {
      const a = ws[i], b = ws[(i + 1) % ws.length];
      let gap = wrap(b.phi - a.phi); if (gap <= 0) gap += TAU;
      if (gap < 110 * DEG) continue;
      const cand = peaks.filter(p => p.s >= 0.02 * s0 && !walls.includes(p) &&
        angDiff(p.phi, a.phi) > 35 * DEG && angDiff(p.phi, b.phi) > 35 * DEG &&
        wrap(p.phi - a.phi + TAU) > 0 && ((wrap(p.phi - a.phi) + TAU) % TAU) < gap &&
        !explainPeak(p, walls, hs));
      if (cand.length) { walls.push(cand[0]); added = true; break; }
    }
    if (!added) break;
  }
  return { walls, explained, objects };
}

// ---------- smoother: robust Levenberg-Marquardt over [rho, drift, (phi_i, d_i)...] ----------

// State x = [rho, drift, h_1..h_m, phi_1, d_1, ..., phi_N, d_N] (walls start at offset o = 2 + m).
// Landmark types: wall, corner (adjacent walls), wallH and cornerH (via horizontal plane k).
// Centre of rotation offset at time t: piecewise-linear 2D knots stored after the walls (v0.4). The body
// sways and steps while turning; with three or more walls this path is identifiable (echo SLAM).
export function trajOffset(x, tr, t) {
  if (!tr || !tr.K) return [0, 0];
  const s = Math.max(0, Math.min(tr.K - 1 - 1e-9, (t - tr.t0) / tr.dt)), j = Math.floor(s), f = s - j;
  const a = tr.kb + 2 * j, b = a + 2;
  return [(1 - f) * x[a] + f * x[b], (1 - f) * x[a + 1] + f * x[b + 1]];
}

function predict(x, c, lm, kappa, tMid, o, tr) {
  const rho = x[0], th = c.theta + x[1] * (c.t - tMid);
  const cc = trajOffset(x, tr, c.t);
  if (lm.type === 'wall' || lm.type === 'wallH') {
    const a = th - x[o + 2 * lm.i];
    const pw = x[o + 2 * lm.i];
    const u = x[o + 1 + 2 * lm.i] - rho * Math.cos(a) - (Math.cos(pw) * cc[0] + Math.sin(pw) * cc[1]);
    const dop = kappa * rho * c.omega * Math.sin(a);
    if (lm.type === 'wall') return u + dop;
    const h = x[2 + lm.k], R = Math.sqrt(u * u + h * h);
    return R + dop * u / R;
  }
  const X = intersectWalls({ phi: x[o + 2 * lm.i], d: x[o + 1 + 2 * lm.i] }, { phi: x[o + 2 * lm.j], d: x[o + 1 + 2 * lm.j] });
  if (!X) return NaN;
  const px = rho * Math.cos(th) + cc[0], py = rho * Math.sin(th) + cc[1];
  const dx = X[0] - px, dy = X[1] - py, R = Math.hypot(dx, dy);
  const vx = -rho * c.omega * Math.sin(th), vy = rho * c.omega * Math.cos(th);
  const rdot = -(dx * vx + dy * vy) / R;
  if (lm.type === 'corner') return R + kappa * rdot;
  const h = x[2 + lm.k], RR = Math.sqrt(R * R + h * h);
  return RR + kappa * rdot * R / RR;
}

function facing(c, lm, x, tMid, o) {
  const th = c.theta + x[1] * (c.t - tMid);
  if (lm.type === 'wall' || lm.type === 'wallH') return angDiff(th, x[o + 2 * lm.i]) <= EST.gateDeg * DEG;
  const X = intersectWalls({ phi: x[o + 2 * lm.i], d: x[o + 1 + 2 * lm.i] }, { phi: x[o + 2 * lm.j], d: x[o + 1 + 2 * lm.j] });
  return X ? angDiff(th, Math.atan2(X[1], X[0])) <= EST.gateDeg * DEG : false;
}

function adjacentPairs(walls) {
  const idx = walls.map((w, i) => i).sort((a, b) => wrap(walls[a].phi) - wrap(walls[b].phi));
  const pairs = [];
  for (let k = 0; k < idx.length && idx.length >= 2; k++) {
    if (idx.length === 2 && k === 1) break;
    const i = idx[k], j = idx[(k + 1) % idx.length];
    let gap = wrap(walls[j].phi - walls[i].phi); if (gap <= 0) gap += TAU;
    if (gap > 25 * DEG && gap < 155 * DEG) pairs.push([i, j, gap]);
  }
  return pairs;
}

function numJacobian(resid, x, r0) {
  const n = x.length, J = [];
  for (let j = 0; j < n; j++) {
    const hstep = j === 1 ? 1e-6 : 1e-5;
    const xp = Float64Array.from(x); xp[j] += hstep;
    const rp = resid(xp);
    const col = new Float64Array(r0.length);
    for (let m = 0; m < r0.length; m++) col[m] = (rp[m] - r0[m]) / hstep;
    J.push(col);
  }
  return J;
}

function normal(J, r0) {
  const n = J.length, A = zeros(n), g = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let k = i; k < n; k++) { let s = 0; const a = J[i], b = J[k]; for (let m = 0; m < r0.length; m++) s += a[m] * b[m]; A[i][k] = A[k][i] = s; }
    let s = 0; for (let m = 0; m < r0.length; m++) s += J[i][m] * r0[m]; g[i] = s;
  }
  return { A, g };
}

export { predict as _predict };

export function smooth(cands, walls0, rho0, opt) {
  const kappa = opt.kappa, tMid = opt.tMid, nW = walls0.length;
  const hs0 = opt.hs || [];
  const m = hs0.length, o = 2 + m;
  // centre trajectory knots after the walls; gauge fixed by a zero-mean prior (the mean offset and the
  // wall distances are otherwise interchangeable)
  const K = opt.trajKnots ?? EST.trajKnots;
  let ct0 = Infinity, ct1 = -Infinity;
  for (const c of cands) { if (c.t < ct0) ct0 = c.t; if (c.t > ct1) ct1 = c.t; }
  const tr = K >= 2 && ct1 > ct0 ? { kb: o + 2 * nW, K, t0: ct0, dt: (ct1 - ct0) / (K - 1) } : null;
  const n = o + 2 * nW + (tr ? 2 * K : 0);
  let x = new Float64Array(n);
  x[0] = rho0; x[1] = 0;
  hs0.forEach((h, k) => { x[2 + k] = h; });
  walls0.forEach((w, i) => { x[o + 2 * i] = w.phi; x[o + 1 + 2 * i] = w.d; });
  const turnT = opt.turnDur || 20;
  const pairs = adjacentPairs(walls0);
  const lms = walls0.map((_, i) => ({ type: 'wall', i }))
    .concat(pairs.map(([i, j]) => ({ type: 'corner', i, j })));
  for (let k = 0; k < m; k++) {
    walls0.forEach((_, i) => lms.push({ type: 'wallH', i, k }));
    pairs.forEach(([i, j]) => lms.push({ type: 'cornerH', i, j, k }));
  }
  const useM = opt.manhattan ?? EST.manhattan;
  const mTol = (opt.manhattanTolDeg ?? EST.manhattanTolDeg) * DEG;
  const mSig = (opt.manhattanSigmaDeg ?? EST.manhattanSigmaDeg) * DEG;
  const mPairs = [];
  if (useM) {
    for (let i = 0; i < nW; i++) for (let j = i + 1; j < nW; j++) {
      const dd = wrap(walls0[j].phi - walls0[i].phi);
      for (const target of [Math.PI / 2, -Math.PI / 2, Math.PI]) if (angDiff(dd, target) < mTol) mPairs.push([i, j, target]);
    }
  }
  const priorRows = (xx) => {
    const rows = [(xx[0] - 0.3) / 0.3, xx[1] / ((10 * DEG) / turnT)];
    for (let k = 0; k < m; k++) rows.push((xx[2 + k] - hs0[k]) / 0.3);
    for (const [i, j, target] of mPairs) rows.push(wrap(xx[o + 2 * j] - xx[o + 2 * i] - target) / mSig);
    if (tr) {
      let mx = 0, my = 0;
      for (let j = 0; j < tr.K; j++) { const a = tr.kb + 2 * j; rows.push(xx[a] / 0.06, xx[a + 1] / 0.06); mx += xx[a]; my += xx[a + 1]; }
      rows.push(mx / tr.K / 0.002, my / tr.K / 0.002);
    }
    return rows;
  };
  let sigma = 0.03, assoc = [], weights = [];
  const diag = { rounds: [] };
  const makeResid = () => (xx) => {
    const pr = priorRows(xx);
    const r = new Float64Array(assoc.length + pr.length);
    for (let q = 0; q < assoc.length; q++) {
      const a = assoc[q], c = cands[a.ci];
      const p = predict(xx, c, lms[a.li], kappa, tMid, o, tr);
      r[q] = Number.isFinite(p) ? Math.sqrt(weights[q]) * (c.r - p) : 0;
    }
    for (let q = 0; q < pr.length; q++) r[assoc.length + q] = pr[q];
    return r;
  };
  for (let round = 0; round < 4; round++) {
    const gate = round === 0 ? 0.10 : Math.max(3.5 * sigma, 0.035);
    const best = new Map();
    for (let ci = 0; ci < cands.length; ci++) {
      const c = cands[ci];
      let bl = -1, be = Infinity;
      for (let li = 0; li < lms.length; li++) {
        if (!facing(c, lms[li], x, tMid, o)) continue;
        const e = c.r - predict(x, c, lms[li], kappa, tMid, o, tr);
        if (Math.abs(e) < Math.abs(be)) { be = e; bl = li; }
      }
      if (bl < 0 || Math.abs(be) > gate) continue;
      const key = c.k * 4096 + bl;
      const prev = best.get(key);
      if (!prev || Math.abs(be) < Math.abs(prev.e)) best.set(key, { ci, li: bl, e: be });
    }
    assoc = Array.from(best.values());
    if (assoc.length < n + 4) break;
    const absE = assoc.map(a => Math.abs(a.e)).sort((a, b) => a - b);
    sigma = Math.max(0.004, 1.4826 * absE[absE.length >> 1]);
    const h = 1.5 * sigma;
    weights = assoc.map(a => {
      const w = Math.max(0.05, cands[a.ci].score);
      const hub = Math.abs(a.e) <= h ? 1 : h / Math.abs(a.e);
      return w * hub / (sigma * sigma);
    });
    const resid = makeResid();
    const cost = (r) => r.reduce((s, v) => s + v * v, 0);
    let lambda = 1e-3, r0 = resid(x), c0 = cost(r0);
    for (let it = 0; it < 12; it++) {
      const J = numJacobian(resid, x, r0);
      const { A, g } = normal(J, r0);
      let improved = false;
      for (let tries = 0; tries < 6 && !improved; tries++) {
        const Al = A.map((row, i) => { const rr = Float64Array.from(row); rr[i] += lambda * (A[i][i] + 1e-9); return rr; });
        const L = cholesky(Al);
        if (!L) { lambda *= 10; continue; }
        const dx = cholSolve(L, g.map(v => -v));
        const xn = Float64Array.from(x);
        for (let i = 0; i < n; i++) xn[i] += dx[i];
        for (let i = 0; i < nW; i++) xn[o + 2 * i] = wrap(xn[o + 2 * i]);
        xn[0] = Math.max(EST.rhoMin * 0.8, Math.min(EST.rhoMax * 1.2, xn[0]));
        for (let k = 0; k < m; k++) xn[2 + k] = Math.max(EST.minPlaneDist * 0.8, xn[2 + k]);
        const rn = resid(xn), cn = cost(rn);
        if (cn < c0) { x = xn; r0 = rn; c0 = cn; lambda = Math.max(1e-7, lambda / 3); improved = true; }
        else lambda *= 5;
      }
      if (!improved) break;
    }
    diag.rounds.push({ round, n: assoc.length, sigma, cost: c0 });
  }
  let P = null, inflate = 1;
  if (assoc.length >= n + 4) {
    const resid = makeResid();
    const r0 = resid(x);
    const { A } = normal(numJacobian(resid, x, r0), r0);
    P = inverseSPD(A);
    // neighbouring frames share sway and multipath, so they carry less information than their count:
    // inflate by the AR(1) variance factor of the residuals (count each contribution once)
    const byLm = new Map();
    for (const a of assoc) { if (!byLm.has(a.li)) byLm.set(a.li, []); byLm.get(a.li).push([cands[a.ci].k, a.e]); }
    let num = 0, den = 0;
    for (const arr of byLm.values()) {
      arr.sort((p, q) => p[0] - q[0]);
      for (let q = 1; q < arr.length; q++) if (arr[q][0] - arr[q - 1][0] <= 2) { num += arr[q][1] * arr[q - 1][1]; den += 0.5 * (arr[q][1] ** 2 + arr[q - 1][1] ** 2); }
    }
    const rho1 = den > 0 ? Math.max(0, Math.min(0.95, num / den)) : 0;
    inflate = Math.min(40, (1 + rho1) / (1 - rho1));
    if (P) for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) P[i][k] *= inflate;
  }
  // consider covariance for unmodelled systematics: sound speed (+-5 C), speaker-mic offset, heading noise
  if (P) {
    for (let i = 0; i < nW; i++) {
      const d = x[o + 1 + 2 * i];
      P[o + 1 + 2 * i][o + 1 + 2 * i] += (0.009 * d) ** 2 + 0.012 ** 2;
      P[o + 2 * i][o + 2 * i] += (0.4 * DEG) ** 2;
    }
    for (let k = 0; k < m; k++) P[2 + k][2 + k] += (0.009 * x[2 + k]) ** 2 + 0.012 ** 2;
  }
  const walls = walls0.map((w, i) => ({
    phi: wrap(x[o + 2 * i]), d: x[o + 1 + 2 * i], s: w.s,
    sPhi: P ? Math.sqrt(Math.max(0, P[o + 2 * i][o + 2 * i])) : 5 * DEG,
    sD: P ? Math.sqrt(Math.max(0, P[o + 1 + 2 * i][o + 1 + 2 * i])) : 0.1,
    support: assoc.filter(a => lms[a.li].type === 'wall' && lms[a.li].i === i).length,
    inferred: !!w.inferred
  }));
  for (const w of walls) if (w.inferred) { w.sD = Math.max(w.sD, 0.15); w.sPhi = Math.max(w.sPhi, 5 * DEG); }
  const hsOut = hs0.map((_, k) => ({ h: x[2 + k], sigma: P ? Math.sqrt(Math.max(0, P[2 + k][2 + k])) : 0.05,
    support: assoc.filter(a => lms[a.li].k === k).length }));
  const traj = tr ? { t0: tr.t0, dt: tr.dt, K: tr.K, cx: Array.from({ length: tr.K }, (_, j) => x[tr.kb + 2 * j]), cy: Array.from({ length: tr.K }, (_, j) => x[tr.kb + 2 * j + 1]) } : null;
  return { x, P, o, walls, hs: hsOut, rho: x[0], drift: x[1], traj, sigma, inflate, nAssoc: assoc.length, lms, assoc, manhattanPairs: mPairs.length, diag };
}

// ---------- the whole scan ----------

export function solveScan(frames, opt = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
  // two distinct microphones: every stage reads the speaker-side channel; the other one only gives echo cosines
  const st = opt.stereo && opt.stereo.ok ? opt.stereo : null;
  if (st) frames = frames.map(f => { if (!f.profile2) return f; const nf = nearFar(f, st); return { ...f, profile: nf.near, far: nf.far }; });
  const kappa = opt.kappa ?? 0;
  const lift = frames.filter(f => f.phase === 'lift');
  const turn = frames.filter(f => f.phase === 'turn');
  if (turn.length < 20) return { ok: false, reason: 'Too few frames in the turn. Turn a full circle slowly.' };
  headingRates(turn);
  const span = Math.abs(turn[turn.length - 1].yaw - turn[0].yaw);
  const inv = invariantProfile(turn);
  const hp = horizontalPlanes(lift, inv, opt);
  const bg = localBackground(turn, opt.bgHalfDeg ?? 45);
  // candidate source: 'model' (residual peaks scored by the learned scorer or amplitude) or 'persistence'
  // (persistence over noise, mapped to (0,1) by s / (s + s0))
  const useP = (opt.candidates ?? EST.candidates) === 'persistence';
  let cands = useP ? persistenceCandidates(turn, bg, opt) : extractCandidates(turn, bg, opt);
  // audit only (tools/audit.mjs): lets an oracle replace or filter the candidates; unset in the app
  if (opt.candidateHook) cands = opt.candidateHook(cands, turn);
  const s0 = opt.persistS0 ?? EST.persistS0;
  for (const c of cands) c.w = c.wGiven != null ? c.wGiven : useP ? c.score / (c.score + s0) : c.score;
  if (st) attachCos(cands, turn, st, opt.binSize ?? EST.binSize);
  const maxRange = opt.maxRange ?? EST.maxRange;
  const nD = Math.ceil((maxRange + 0.8 - GRID.d0) / GRID.dStep);
  const hb = bestRho(cands, kappa, nD);
  const peaks = houghPeaks(hb.acc, nD, 24, 0.02);
  // Coordinate ascent on the explained-echo score (sound tracing in reverse):
  // walls by greedy selection, then the hold radius, then the floor and ceiling distances.
  const C = packCands(cands);
  const frameHeadings = turn.map(f => f.yaw);
  let hsCur = [];
  for (const v of [hp.floor && !hp.floor.assumed ? hp.floor.h : null, hp.ceil && !hp.ceil.assumed ? hp.ceil.h : null].concat(hp.bands.map(b => b.h)))
    if (v && hsCur.every(u => Math.abs(u - v) > 0.06) && hsCur.length < 2) hsCur.push(v);
  const shiftWalls = (ws, dr) => ws.map(w => ({ ...w, d: w.d + dr }));
  const hs0 = hsCur.slice();
  const starts = [hb.rho];
  for (const r of [0.2, 0.3, 0.42]) if (starts.every(u => Math.abs(u - r) > 0.05)) starts.push(r);
  let bestSel = null, peaksBest = peaks;
  const allSel = [];
  for (const rho0 of starts) {
    let rhoCur = rho0, hsC = hs0.slice();
    let peaksCur = Math.abs(rho0 - hb.rho) < 1e-9 ? peaks : houghPeaks(houghAcc(cands, rho0, kappa, nD), nD, 24, 0.02);
    let sel = null;
    // the v0.4 selector re-searches exactly afterwards, so the greedy pass only needs rho and plane heights
    const nIter = (opt.selector ?? EST.selector) === 'dp' ? 2 : 3;
    for (let iter = 0; iter < nIter; iter++) {
      const g = greedyWalls(cands, peaksCur, hsC, rhoCur, { ...opt, C });
      let walls = g.walls;
      let bestR = { rho: rhoCur, sc: explainScore(C, walls, hsC, rhoCur) };
      for (let r = EST.rhoMin; r <= EST.rhoMax + 1e-9; r += 0.01) {
        const sc = explainScore(C, shiftWalls(walls, r - rhoCur), hsC, r);
        if (sc > bestR.sc) bestR = { rho: r, sc };
      }
      walls = shiftWalls(walls, bestR.rho - rhoCur);
      rhoCur = bestR.rho;
      for (let slot = 0; slot < 2; slot++) {
        let bestH = { h: hsC[slot], sc: explainScore(C, walls, hsC.filter(v => v), rhoCur) };
        for (let h = EST.minPlaneDist; h <= 2.6; h += 0.03) {
          if (hsC.some((u, k) => k !== slot && u && Math.abs(u - h) < 0.06)) continue;
          const trial = hsC.slice(); trial[slot] = h;
          const sc = explainScore(C, walls, trial.filter(v => v), rhoCur);
          if (sc > bestH.sc * 1.002) bestH = { h, sc };
        }
        if (bestH.h) hsC[slot] = bestH.h;
      }
      hsC = hsC.filter(v => v);
      const wMean = C.total / Math.max(1, C.n);
      const costW = Math.max((opt.minGainFrac ?? 0.04) * C.total, 6 * wMean);
      const v = explainScore(C, walls, hsC, rhoCur) - costW * walls.length;
      sel = { walls, log: g.log, hs: hsC.slice(), rho: rhoCur, val: v / C.total, score: explainScore(C, walls, hsC, rhoCur) / C.total, rho0 };
      peaksCur = houghPeaks(houghAcc(cands, rhoCur, kappa, nD), nD, 24, 0.02);
    }
    allSel.push(sel);
    if (!bestSel || sel.val > bestSel.val) { bestSel = sel; peaksBest = peaksCur; }
  }
  // Self-consistency: do the independent starts land on the same room? A rival room that explains the
  // echoes almost as well is ambiguity the score alone cannot show.
  const sameRoom = (A, B) => A.length === B.length && A.every(a => B.some(b => angDiff(a.phi, b.phi) < 6 * DEG && Math.abs(a.d - b.d) < 0.06));
  const agreeFrac = allSel.filter(q => sameRoom(q.walls, bestSel.walls)).length / allSel.length;
  const rivals = allSel.filter(q => !sameRoom(q.walls, bestSel.walls));
  const rivalMargin = rivals.length ? bestSel.val - Math.max(...rivals.map(q => q.val)) : Infinity;
  // v0.4 selector (core/select.js): exact search over wall lines with a detection likelihood; the greedy
  // answers above are kept as contenders, so the joint score decides between them (SANDBOX.md)
  let selInfo = null;
  if ((opt.selector ?? EST.selector) === 'dp') {
    const r = dpSelect({ cands, rho: bestSel.rho, hs: bestSel.hs, kappa, tMid: 0.5 * (turn[0].t + turn[turn.length - 1].t), turnDur: turn[turn.length - 1].t - turn[0].t, nD, opt, peaks: peaksBest, contenders: allSel.map(q => q.walls) });
    if (r) {
      bestSel.walls = r.walls; bestSel.rho = r.rho; bestSel.hs = r.hs.slice();
      bestSel.score = explainScore(C, bestSel.walls, bestSel.hs, bestSel.rho) / C.total;
      selInfo = r;
    }
  }
  const peaksCur = peaksBest;
  // close open sides with the best weaker evidence, flagged as inferred
  {
    let ws = bestSel.walls.slice();
    for (let pass = 0; pass < 3; pass++) {
      const open0 = ws.length < 3 ? 4 : openSides(ws);
      if (open0 === 0) break;
      let best = null;
      for (const pk of peaksCur) {
        if (pk.d < EST.minWallDist || ws.includes(pk) || ws.some(w => angDiff(w.phi, pk.phi) < 25 * DEG)) continue;
        const trial = ws.concat([{ ...pk, inferred: true }]);
        const open = openSides(trial);
        if (open >= open0 && trial.length >= 3) continue;
        const sc = explainScore(C, trial, bestSel.hs, bestSel.rho);
        if (!best || sc > best.sc) best = { sc, trial };
      }
      if (!best) break;
      ws = best.trial;
    }
    bestSel.walls = ws;
  }
  const sel = { walls: bestSel.walls, explained: [], objects: peaksCur.slice(0, 16).filter(p => bestSel.walls.every(w => angDiff(w.phi, p.phi) > 15 * DEG || Math.abs(w.d - p.d) > 0.2)) };
  const hs = bestSel.hs;
  if (sel.walls.length < 2) return { ok: false, reason: 'Could not find walls. Try the high band off, or move away from furniture.', inv, cands, peaks, hp, span };
  const tMid = 0.5 * (turn[0].t + turn[turn.length - 1].t);
  const sm = smooth(cands, sel.walls, bestSel.rho, { ...opt, kappa, tMid, hs, turnDur: turn[turn.length - 1].t - turn[0].t });
  // drop walls that do not form an edge of the closed plan
  let walls = sm.walls;
  let pg = roomPolygon(walls);
  const keep = walls.map((_, i) => pg.used.has(i));
  const dropped = walls.filter((_, i) => !keep[i]);
  const wallIndex = walls.map((_, i) => i).filter(i => keep[i]);
  if (dropped.length) {
    walls = walls.filter((_, i) => keep[i]);
    pg = roomPolygon(walls);
  }
  const openEdges = pg.edgeWall.map(id => id < 0);
  // floor and ceiling: distances come from the bounce tracks (smoother); which is which comes from
  // the lift (the floor recedes as the phone rises, the ceiling approaches). Binary decision per pair.
  let floor = hp.floor ? { ...hp.floor } : null, ceil = hp.ceil ? { ...hp.ceil } : null;
  const Hs = sm.hs.filter(H => H.support >= 10);
  if (Hs.length >= 1) {
    const lab = (H, sign) => liftTrackScore(lift, H.h, sign) ?? 0;
    let assign;
    if (Hs.length >= 2) {
      const [a, b] = Hs;
      const s1 = lab(a, 1) + lab(b, -1), s2 = lab(a, -1) + lab(b, 1);
      assign = s1 >= s2 ? { f: a, c: b, margin: s1 - s2 } : { f: b, c: a, margin: s2 - s1 };
    } else {
      const a = Hs[0];
      assign = lab(a, 1) >= lab(a, -1) ? { f: a, c: null } : { f: null, c: a };
    }
    const mk = (H, src) => H ? { h: H.h, sigma: H.sigma, fromBounces: H.support, labelMargin: assign.margin ?? null, assumed: !lift.length } : src;
    floor = mk(assign.f, floor);
    ceil = mk(assign.c, ceil);
  }
  // reflectors that are not walls, corners or ghosts become objects with Gaussian extent
  const objects = sel.objects.concat(dropped).filter(o => o.s >= 0.12 * peaks[0].s)
    .map(o => ({ x: o.d * Math.cos(o.phi), y: o.d * Math.sin(o.phi), sr: 0.04, st: Math.max(0.05, o.d * 5 * DEG), phi: o.phi, d: o.d }))
    .filter(o => pointInConvex(pg.poly, o.x, o.y));
  const t1 = (typeof performance !== 'undefined' ? performance : Date).now();
  return {
    ok: true, walls, wallIndex, wallOffset: sm.o, poly: pg.poly, edgeWall: pg.edgeWall, openEdges, closed: !openEdges.some(Boolean),
    floor, ceil, horiz: hp,
    rho: sm.rho, drift: sm.drift, sigma: sm.sigma, inflate: sm.inflate, P: sm.P, x: sm.x, nAssoc: sm.nAssoc,
    smoother: sm, objects, explained: sel.explained, peaks, packed: C, rhoCurve: hb.curve, hs: sm.hs, greedy: bestSel.log,
    quality: { explainedFrac: bestSel.score, selector: selInfo ? 'dp' : 'greedy', odds: selInfo ? selInfo.margin : null, selMs: selInfo ? selInfo.ms : 0, agreeFrac, rivalMargin: Number.isFinite(rivalMargin) ? rivalMargin : null, starts: allSel.length, measured: walls.filter(w => !w.inferred).length, inferred: walls.filter(w => w.inferred).length, closed: !openEdges.some(Boolean), sigma: sm.sigma },
    // structural plausibility only; confidence also needs perturbation persistence (core/persist.js)
    plausible: walls.filter(w => !w.inferred).length >= 3 && !openEdges.some(Boolean) && walls.filter(w => w.inferred).length <= 1,
    inv, cands: opt.keepCands ? cands : undefined, nCands: cands.length,
    span, tMid, kappa, solveMs: t1 - t0
  };
}

// Baseline: rotating rangefinder. Strongest residual echo per 5 degree heading bin,
// corrected by an assumed 0.30 m hold radius, joined into a star polygon.
export function baselineScan(frames, opt = {}) {
  const turn = frames.filter(f => f.phase === 'turn');
  if (turn.length < 20) return { ok: false };
  const inv = invariantProfile(turn);
  const cands = extractCandidates(turn, inv, { ...opt, scorer: null });
  const bins = new Array(72).fill(null);
  for (const c of cands) {
    let b = Math.floor((((c.theta % TAU) + TAU) % TAU) / (5 * DEG)) % 72;
    if (!bins[b] || c.amp > bins[b].amp) bins[b] = c;
  }
  const pts = [];
  for (let b = 0; b < 72; b++) {
    const c = bins[b];
    if (!c) continue;
    const r = c.r + 0.3;
    pts.push([r * Math.cos(c.theta), r * Math.sin(c.theta), Math.atan2(Math.sin(c.theta), Math.cos(c.theta))]);
  }
  pts.sort((a, b) => a[2] - b[2]);
  return { ok: pts.length >= 3, poly: pts.map(p => [p[0], p[1]]), bins };
}
