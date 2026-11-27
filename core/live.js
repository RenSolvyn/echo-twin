// Live walls (v0.6, SANDBOX.md): every chirp's echo candidates, weighted by the echo model, vote into Hough
// maps over (wall angle, wall distance) for four hold radii. Each chirp the best map's peaks are described
// and a small model gives P(real wall); walls with P >= tau are shown. A wall echo follows
//   r(theta) = d - rho cos(theta - phi) + kappa rho omega sin(theta - phi),
// so the votes are exact physics; only the weights and the final yes or no are learned.
// Time per chirp: votes O(4 x 361 x candidates); decode O(360 x nD) for blur, ridge removal and peaks.


import { wrap, angDiff, DEG, roomPolygon } from './geom.js';
import { cosWeight } from './stereo.js';

export const RHOS = [0.15, 0.25, 0.35, 0.45];
// Live grid: 1 deg by 2 cm (half the final solver's range resolution; peaks are refined below a bin).
export const LGRID = { nPhi: 360, d0: 0.2, dStep: 0.02 };
export const N_WFEAT = 14;
// Echo cloud grid (frozen by the v0.6.1 pre-registration, SANDBOX.md).
export const CLOUD = { nB: 120, r0: 0.4, r1: 6.0, dr: 0.025, flat: 40, zero: 65 };
const JMAX = 180, COSJ = new Float64Array(2 * JMAX + 1), SINJ = new Float64Array(2 * JMAX + 1);
for (let j = -JMAX; j <= JMAX; j++) { COSJ[j + JMAX] = Math.cos(j * DEG); SINJ[j + JMAX] = Math.sin(j * DEG); }

export class LiveWalls {
  // opt: { kappa, maxRange, model (SmallMLP over N_WFEAT features) or null, tau }
  constructor(opt = {}) {
    this.kappa = opt.kappa ?? 0; this.model = opt.model || null; this.tau = opt.tau ?? 0.5;
    this.nD = Math.ceil(((opt.maxRange ?? 6) + 0.8 - LGRID.d0) / LGRID.dStep);
    this.acc = RHOS.map(() => new Float32Array(LGRID.nPhi * this.nD));
    this.tmp = new Float32Array(LGRID.nPhi * this.nD); this.sm = new Float32Array(LGRID.nPhi * this.nD);
    this.hist = []; // per chirp { theta, omega, list: [{ r, p }] }
    this.anchor = null; this.theta0 = null; this.swept = 0;
  }

  // Add one chirp's candidates ({ r, p, cos }). Time O(4 x 361 x candidates).
  add(theta, omega, cands) {
    if (this.anchor == null) { this.anchor = (((theta / DEG) % 1) + 1) % 1; this.theta0 = theta; }
    this.swept = Math.max(this.swept, Math.abs(theta - this.theta0));
    const { nPhi, d0, dStep } = LGRID, nD = this.nD;
    const list = [];
    for (const c of cands) {
      const w = c.p;
      if (!(w > 0.02)) continue;
      list.push({ r: c.r, p: w });
      const base = theta / DEG;
      RHOS.forEach((rho, ri) => {
        const acc = this.acc[ri];
        for (let j = -JMAX; j <= JMAX; j++) {
          const ca = COSJ[j + JMAX], sa = SINJ[j + JMAX];
          const d = c.r + rho * ca - this.kappa * rho * (omega || 0) * sa;
          const x = (d - d0) / dStep, i0 = Math.floor(x);
          if (i0 < 0 || i0 >= nD - 1) continue;
          const fr = x - i0;
          let pb = Math.round(base - this.anchor - j) % nPhi; if (pb < 0) pb += nPhi;
          const g = c.cos != null ? w * cosWeight(c.cos, ca) : w;
          const o = pb * nD + i0;
          acc[o] += g * (1 - fr); acc[o + 1] += g * fr;
        }
      });
    }
    this.hist.push({ theta, omega: omega || 0, list });
  }

  // Hold radius whose raw map has the strongest wall-like columns: the sum of the six largest per-angle
  // maxima. Time O(4 x 360 x nD), plain scans only.
  bestRho() {
    let best = -1, bi = 0;
    const col = new Float32Array(LGRID.nPhi);
    this.acc.forEach((a, ri) => {
      for (let p = 0; p < LGRID.nPhi; p++) { let m = 0; const o = p * this.nD; for (let i = 0; i < this.nD; i++) if (a[o + i] > m) m = a[o + i]; col[p] = m; }
      col.sort();
      let s = 0; for (let q = 1; q <= 6; q++) s += col[LGRID.nPhi - q];
      if (s > best) { best = s; bi = ri; }
    });
    return bi;
  }

  // Ridge removal: subtract each range row's median over angle, estimated from every third angle.
  // Time O(nD x 120 log 120).
  static deridgeFast(acc, nPhi, nD) {
    const col = new Float32Array(Math.ceil(nPhi / 3));
    for (let i = 0; i < nD; i++) {
      let q = 0; for (let p = 0; p < nPhi; p += 3) col[q++] = acc[p * nD + i];
      const c = col.subarray(0, q); c.sort();
      const m = c[q >> 1];
      for (let p = 0; p < nPhi; p++) { const v = acc[p * nD + i] - m; acc[p * nD + i] = v > 0 ? v : 0; }
    }
    return acc;
  }

  // 5-tap binomial blur in range then angle (angle wraps), into preallocated buffers. Time O(10 x 360 x nD).
  blurInto(acc) {
    const nPhi = LGRID.nPhi, nD = this.nD, t = this.tmp, o = this.sm;
    for (let p = 0; p < nPhi; p++) {
      const b = p * nD;
      for (let i = 0; i < nD; i++) {
        const a0 = i >= 2 ? acc[b + i - 2] : 0, a1 = i >= 1 ? acc[b + i - 1] : 0, a3 = i + 1 < nD ? acc[b + i + 1] : 0, a4 = i + 2 < nD ? acc[b + i + 2] : 0;
        t[b + i] = 0.0625 * (a0 + a4) + 0.25 * (a1 + a3) + 0.375 * acc[b + i];
      }
    }
    for (let p = 0; p < nPhi; p++) {
      const b0 = ((p - 2 + nPhi) % nPhi) * nD, b1 = ((p - 1 + nPhi) % nPhi) * nD, b2 = p * nD, b3 = ((p + 1) % nPhi) * nD, b4 = ((p + 2) % nPhi) * nD;
      for (let i = 0; i < nD; i++) o[b2 + i] = 0.0625 * (t[b0 + i] + t[b4 + i]) + 0.25 * (t[b1 + i] + t[b3 + i]) + 0.375 * t[b2 + i];
    }
    return o;
  }

  // Local maxima above 3 % of the largest, with non-maximum suppression (18 deg, 20 cm) and sub-bin refinement.
  // Time O(360 x nD).
  findPeaks(m, maxPeaks = 24, relThr = 0.03) {
    const { nPhi, d0, dStep } = LGRID, nD = this.nD;
    let mx = 0; for (let i = 0; i < m.length; i++) if (m[i] > mx) mx = m[i];
    const thr = mx * relThr, loc = [];
    for (let p = 0; p < nPhi; p++) {
      const pm = ((p - 1 + nPhi) % nPhi) * nD, pp = ((p + 1) % nPhi) * nD, pc = p * nD;
      for (let i = 1; i < nD - 1; i++) {
        const v = m[pc + i];
        if (v < thr || v < m[pc + i - 1] || v < m[pc + i + 1] || v < m[pm + i] || v < m[pp + i] || v < m[pm + i - 1] || v < m[pm + i + 1] || v < m[pp + i - 1] || v < m[pp + i + 1]) continue;
        loc.push({ p, i, v });
      }
    }
    loc.sort((a, b) => b.v - a.v);
    const kept = [];
    const at = (p, i) => m[((p + nPhi) % nPhi) * nD + i];
    for (const L of loc) {
      if (kept.some(K => Math.min(Math.abs(L.p - K.p), nPhi - Math.abs(L.p - K.p)) <= 18 && Math.abs(L.i - K.i) <= 10)) continue;
      const a = at(L.p - 1, L.i), b = L.v, c = at(L.p + 1, L.i);
      const fp = (a - 2 * b + c) !== 0 ? 0.5 * (a - c) / (a - 2 * b + c) : 0;
      const e = at(L.p, L.i - 1), f = at(L.p, L.i + 1);
      const fi = (e - 2 * b + f) !== 0 ? 0.5 * (e - f) / (e - 2 * b + f) : 0;
      kept.push({ phi: wrap((L.p + this.anchor + Math.max(-0.5, Math.min(0.5, fp))) * DEG), d: d0 + (L.i + Math.max(-0.5, Math.min(0.5, fi))) * dStep, s: b, p: L.p, i: L.i });
      if (kept.length >= maxPeaks) break;
    }
    return kept;
  }

  // Best hold radius's map, blurred and ridge-removed; cached until the next chirp (it lives in this.sm).
  // Time O(4 x 360 x nD) when stale, O(1) when cached.
  map() {
    if (this.mapK !== this.hist.length) {
      this.mapRi = this.bestRho();
      LiveWalls.deridgeFast(this.blurInto(this.acc[this.mapRi]), LGRID.nPhi, this.nD);
      this.mapK = this.hist.length;
    }
    return { rho: RHOS[this.mapRi], m: this.sm };
  }

  // Echo cloud (v0.6.1, SANDBOX.md): the wall evidence projected back to the floor, read along bearings every
  // 3 deg (heading frame) at ranges r0..r1. A point at bearing a, range r takes the largest positive evidence
  // over wall angles phi within CLOUD.zero deg of a at wall distance r cos(a - phi), times a window that is 1 up
  // to CLOUD.flat deg and falls to 0 at CLOUD.zero deg. Outline: per bearing, the range of the strongest value
  // (pre-registered rule; the farthest value lands on double-bounce ghosts). Returns { rho, nB, nR, r0, dr,
  // heat (Uint8Array nB x nR, scaled to the cloud's largest value), ridge (m, NaN when empty), strength (0..1) }.
  // Time O(nB x (2 zero + 1) x nR), about 3.5 M steps; space O(nB x nR).
  cloud() {
    const { nB, r0, r1, dr, flat, zero } = CLOUD, nR = Math.round((r1 - r0) / dr) + 1;
    const { rho, m } = this.map(), nD = this.nD, { d0, dStep, nPhi } = LGRID;
    const prof = new Float32Array(nB * nR), ridge = new Float32Array(nB), strength = new Float32Array(nB);
    const nq = 2 * zero + 1, cq = new Float64Array(nq), wq = new Float64Array(nq);
    for (let q = -zero; q <= zero; q++) { const a = Math.abs(q); cq[q + zero] = Math.cos(q * DEG); wq[q + zero] = a <= flat ? 1 : a >= zero ? 0 : (zero - a) / (zero - flat); }
    const anchor = this.anchor ?? 0;
    let gmax = 0;
    for (let b = 0; b < nB; b++) {
      const base = b * (360 / nB), o = b * nR;
      for (let q = -zero; q <= zero; q++) {
        const w = wq[q + zero]; if (w <= 0) continue;
        let pp = Math.round(base + q - anchor) % nPhi; if (pp < 0) pp += nPhi;
        const col = pp * nD, c = cq[q + zero];
        for (let i = 0; i < nR; i++) {
          const ii = Math.round(((r0 + i * dr) * c - d0) / dStep);
          if (ii < 0 || ii >= nD) continue;
          const v = m[col + ii] * w;
          if (v > prof[o + i]) prof[o + i] = v;
        }
      }
      let mx = 0, im = -1;
      for (let i = 0; i < nR; i++) if (prof[o + i] > mx) { mx = prof[o + i]; im = i; }
      ridge[b] = im >= 0 ? r0 + im * dr : NaN; strength[b] = mx;
      if (mx > gmax) gmax = mx;
    }
    const heat = new Uint8Array(nB * nR), s = gmax > 0 ? 255 / gmax : 0;
    for (let i = 0; i < heat.length; i++) heat[i] = Math.round(prof[i] * s);
    for (let b = 0; b < nB; b++) strength[b] = gmax > 0 ? strength[b] / gmax : 0;
    return { rho, nB, nR, r0, dr, heat, ridge, strength, prof };
  }

  // Peaks of the best hold-radius map, with features. Time O(360 x nD).
  peaks() {
    if (this.hist.length < 8) return { rho: null, peaks: [] };
    const { rho, m } = this.map();
    const pk = this.findPeaks(m);
    const K = this.hist.length, top = pk.length ? pk[0].s : 1;
    const at = (p, i) => m[((p + LGRID.nPhi) % LGRID.nPhi) * this.nD + Math.max(0, Math.min(this.nD - 1, i))];
    const peaks = pk.map((q, rank) => {
      // support: chirps whose candidates lie on this wall's range curve within 3 cm
      let sup = 0, supP = 0, nSup = 0, lo = Infinity, hi = -Infinity;
      for (const h of this.hist) {
        const rr = q.d - rho * Math.cos(h.theta - q.phi) + this.kappa * rho * h.omega * Math.sin(h.theta - q.phi);
        let bp = 0;
        for (const c of h.list) if (Math.abs(c.r - rr) < 0.03 && c.p > bp) bp = c.p;
        if (bp > 0) { sup++; supP += bp; nSup++; const t = h.theta - this.theta0; lo = Math.min(lo, t); hi = Math.max(hi, t); }
      }
      const v = q.s;
      const sharpP = (2 * v - at(q.p - 2, q.i) - at(q.p + 2, q.i)) / (v + 1e-9);
      const sharpD = (2 * v - at(q.p, q.i - 2) - at(q.p, q.i + 2)) / (v + 1e-9);
      let dup = 0, perp = 0, behind = 0;
      for (const o of pk) {
        if (o === q || o.s <= q.s) continue;
        const da = angDiff(o.phi, q.phi);
        if (da < 30 * DEG && Math.abs(o.d - q.d) < 0.3) dup = 1;
        if (Math.abs(da - Math.PI / 2) < 6 * DEG || da > Math.PI - 6 * DEG) perp = 1;
        if (da < 10 * DEG && o.d < q.d - 0.1) behind = 1;
      }
      const x = new Float32Array(N_WFEAT);
      x[0] = v / K; x[1] = v / (top + 1e-9); x[2] = supP / K; x[3] = nSup ? (hi - lo) / (2 * Math.PI) : 0; x[4] = nSup ? supP / nSup : 0;
      x[5] = q.d / 6; x[6] = Math.max(-1, Math.min(2, sharpP)); x[7] = Math.max(-1, Math.min(2, sharpD)); x[8] = Math.min(1.1, this.swept / (2 * Math.PI));
      x[9] = rank / 24; x[10] = dup; x[11] = perp; x[12] = behind; x[13] = sup / K;
      return { phi: wrap(q.phi), d: q.d, s: v, feat: x };
    });
    return { rho, peaks };
  }

  // Walls to show now: peaks with P >= tau, strongest first, one per 15 deg and 0.2 m. Returns { rho, walls, plan }.
  walls() {
    const { rho, peaks } = this.peaks();
    for (const q of peaks) q.p = this.model ? this.model.p(q.feat) : Math.min(1, q.feat[1]);
    const keep = [];
    for (const q of peaks.filter(q => q.p >= this.tau).sort((a, b) => b.p - a.p)) {
      if (keep.some(k => angDiff(k.phi, q.phi) < 15 * DEG && Math.abs(k.d - q.d) < 0.2)) continue;
      keep.push(q);
    }
    const walls = keep.map(q => ({ phi: q.phi, d: q.d, p: q.p, sD: 0.03, inferred: false }));
    const plan = walls.length >= 2 ? roomPolygon(walls) : null;
    return { rho, walls, plan, all: peaks };
  }
}

// Whether the app draws live walls during the turn; set only by the pre-registered sealed test (SANDBOX.md v0.6).
// Adopted 2026-11-27: displayed-wall precision 0.93 at 75 % and 0.87 at 100 % of the turn (lower bounds 0.85 and
// 0.77), worst per-chirp p95 9.2 ms. Sparse: about half a wall per room by the end of the turn (recall 0.09).
export const LIVE_ENABLED = true;

// Whether the plan view draws the echo cloud; set only by the pre-registered sealed test (SANDBOX.md v0.6.1).
// Adopted 2026-11-27 on 50 fresh rooms: strongest-ridge outline within 15 cm on 0.38 of bearings vs 0.10 with
// directions scrambled (difference +0.28, CI +0.23 to +0.33); strongest beats farthest by +0.31 (CI +0.26 to
// +0.36). The solver stays ahead (+0.10, CI +0.03 to +0.17), so the cloud is evidence on screen, never walls.
export const CLOUD_ENABLED = true;
