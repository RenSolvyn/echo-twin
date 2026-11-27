// Live twin after the scan:
//  1. Wall EKF on S1 x R per wall plus hold radius and heading bias, updated from each chirp's echoes,
//     with innovation-based adaptive measurement noise (Sage-Husa) and an NIS consistency monitor.
//  2. Divergence: how far each live profile departs from what the twin predicts for that heading,
//     with hysteresis so events fire on transitions only.
//  3. Doppler radar: alternating up and down sweeps shift a moving echo by +kappa rdot and -kappa rdot;
//     a constant-velocity Kalman filter per target uses that coupling to measure radial speed.

import { wrap, DEG, angDiff } from './geom.js';
import { median } from './dsp.js';

const TWO_PI = 2 * Math.PI;

export class LiveTwin {
  // room: solveScan result. scanFrames: turn frames with profile and yaw. opt: { kappa, binSize, minRange, maxRange }
  constructor(room, scanFrames, opt = {}) {
    this.kappa = opt.kappa ?? 0;
    this.binSize = opt.binSize ?? 0.01;
    this.iMin = Math.round((opt.minRange ?? 0.35) / this.binSize);
    this.iMax = Math.round((opt.maxRange ?? 6) / this.binSize);
    // EKF state: [rho, beta, phi_1, d_1, ...]
    const W = room.walls;
    const n = 2 + 2 * W.length;
    this.x = new Float64Array(n);
    this.x[0] = room.rho; this.x[1] = 0;
    W.forEach((w, i) => { this.x[2 + 2 * i] = w.phi; this.x[3 + 2 * i] = w.d; });
    this.P = Array.from({ length: n }, (_, i) => { const r = new Float64Array(n); r[i] = 1e-6; return r; });
    this.P[0][0] = 0.02 ** 2; this.P[1][1] = (2 * DEG) ** 2;
    W.forEach((w, i) => { this.P[2 + 2 * i][2 + 2 * i] = w.sPhi ** 2; this.P[3 + 2 * i][3 + 2 * i] = w.sD ** 2; });
    this.nW = W.length;
    this.R = (room.sigma || 0.02) ** 2;
    this.nis = []; this.lastT = null;
    // static prediction per 5 degree heading bin, per sweep direction (down learned online)
    this.bins = 72;
    this.stat = { 1: new Array(this.bins).fill(null), [-1]: new Array(this.bins).fill(null) };
    this.vari = { 1: new Array(this.bins).fill(null), [-1]: new Array(this.bins).fill(null) };
    const groups = Array.from({ length: this.bins }, () => []);
    for (const f of scanFrames) groups[this.bin(f.yaw)].push(f.profile);
    groups.forEach((g, b) => { if (g.length) this.stat[1][b] = medianProfile(g); });
    this.fillGaps(1);
    // divergence thresholds from the scan's own variation (neighbouring frames)
    const ds = [];
    for (let k = 1; k < scanFrames.length; k++) ds.push(divergence(scanFrames[k].profile, scanFrames[k - 1].profile, this.iMin, this.iMax));
    ds.sort((a, b) => a - b);
    const q = (p) => ds.length ? ds[Math.min(ds.length - 1, Math.floor(p * ds.length))] : 0.2;
    this.tHi = q(0.99) * 1.3; this.tLo = q(0.9);
    this.event = false; this.hiCount = 0; this.loCount = 0;
    this.tracks = []; this.nextId = 1;
  }

  bin(yaw) { let a = ((yaw % TWO_PI) + TWO_PI) % TWO_PI; return Math.floor(a / TWO_PI * this.bins) % this.bins; }

  fillGaps(dir) {
    const S = this.stat[dir];
    for (let b = 0; b < this.bins; b++) {
      if (S[b]) continue;
      for (let k = 1; k < this.bins / 2; k++) {
        const v = S[(b + k) % this.bins] || S[(b - k + this.bins) % this.bins];
        if (v) { S[b] = v; break; }
      }
    }
  }

  predictRange(i, theta) {
    const a = theta + this.x[1] - this.x[2 + 2 * i];
    return this.x[3 + 2 * i] - this.x[0] * Math.cos(a);
  }

  // frame: { t, yaw, dir, profile, cands: [{ r, score }] }
  update(frame) {
    const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
    const dt = this.lastT == null ? 0.1 : Math.max(0.01, Math.min(1, frame.t - this.lastT));
    this.lastT = frame.t;
    // ---- wall EKF: predict (heading bias random walk), then sequential scalar updates ----
    this.P[1][1] += (0.5 * DEG) ** 2 * dt;
    for (let i = 0; i < this.nW; i++) this.P[3 + 2 * i][3 + 2 * i] += 1e-6 * dt;
    const n = this.x.length;
    const used = new Set();
    for (let i = 0; i < this.nW; i++) {
      const rhat = this.predictRange(i, frame.yaw);
      let best = null;
      for (const c of frame.cands || []) {
        if (used.has(c) || c.score < 0.5) continue;
        const e = c.r - rhat;
        if (!best || Math.abs(e) < Math.abs(best.e)) best = { c, e };
      }
      if (!best) continue;
      const a = frame.yaw + this.x[1] - this.x[2 + 2 * i];
      const H = new Float64Array(n);
      H[0] = -Math.cos(a); H[1] = this.x[0] * Math.sin(a);
      H[2 + 2 * i] = -this.x[0] * Math.sin(a); H[3 + 2 * i] = 1;
      const PH = new Float64Array(n);
      for (let r = 0; r < n; r++) { let s = 0; for (let k = 0; k < n; k++) s += this.P[r][k] * H[k]; PH[r] = s; }
      let HPH = 0; for (let k = 0; k < n; k++) HPH += H[k] * PH[k];
      const S = HPH + this.R;
      const nis = best.e * best.e / S;
      if (nis > 9) continue; // 3-sigma gate
      used.add(best.c);
      this.nis.push(nis); if (this.nis.length > 200) this.nis.shift();
      // Sage-Husa adaptive measurement noise: R tracks innovation variance minus predicted spread
      const bR = 0.03;
      this.R = Math.min(0.05 ** 2, Math.max(0.004 ** 2, (1 - bR) * this.R + bR * (best.e * best.e - HPH)));
      const K = PH.map(v => v / S);
      for (let r = 0; r < n; r++) this.x[r] += K[r] * best.e;
      for (let r = 0; r < n; r++) for (let k = 0; k < n; k++) this.P[r][k] -= K[r] * PH[k];
      this.x[2 + 2 * i] = wrap(this.x[2 + 2 * i]);
    }
    // ---- divergence against the twin's prediction for this heading ----
    const dir = frame.dir || 1, b = this.bin(frame.yaw);
    let S = this.stat[dir][b];
    if (!S) { this.stat[dir][b] = Float32Array.from(frame.profile); this.fillGaps(dir); S = this.stat[dir][b]; }
    const D = divergence(frame.profile, S, this.iMin, this.iMax);
    let transition = null;
    if (!this.event) {
      this.hiCount = D > this.tHi ? this.hiCount + 1 : 0;
      if (this.hiCount >= 2) { this.event = true; this.loCount = 0; transition = 'start'; }
    } else {
      this.loCount = D < this.tLo ? this.loCount + 1 : 0;
      if (this.loCount >= 3) { this.event = false; this.hiCount = 0; transition = 'end'; }
    }
    // clutter map: per heading bin and sweep direction, mean and variance of each range bin (CFAR)
    let V = this.vari[dir][b];
    if (!V) { V = this.vari[dir][b] = new Float32Array(S.length); for (let i = 0; i < S.length; i++) V[i] = (0.25 * S[i]) ** 2; }
    const det = detections(frame.profile, S, V, this.iMin, Math.min(this.iMax, Math.round(4.5 / this.binSize)), this.binSize);
    if (!this.event || det.length === 0) {
      for (let i = 0; i < S.length; i++) {
        const e = frame.profile[i] - S[i];
        S[i] += 0.08 * e;
        V[i] = 0.95 * V[i] + 0.05 * e * e;
      }
    }
    this.trackStep(det, dir, dt, frame);
    const t1 = (typeof performance !== 'undefined' ? performance : Date).now();
    return {
      divergence: D, event: this.event, transition, thresholds: [this.tLo, this.tHi],
      tracks: this.tracks.filter(tr => tr.hits >= 3).map(tr => ({ id: tr.id, r: tr.x[0], rdot: tr.x[1], sr: Math.sqrt(tr.P[0][0]), sv: Math.sqrt(tr.P[1][1]), yaw: tr.yaw })),
      nisMean: this.nis.length ? this.nis.reduce((a, b) => a + b, 0) / this.nis.length : null,
      R: Math.sqrt(this.R), walls: this.walls(), ms: t1 - t0
    };
  }

  trackStep(det, dir, dt, frame) {
    const k = this.kappa, q = 1.0; // white-acceleration process noise (m/s^2)^2
    for (const tr of this.tracks) {
      const [r, v] = tr.x;
      tr.x = [r + v * dt, v];
      const P = tr.P;
      const p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1] + q * dt ** 4 / 4;
      const p01 = P[0][1] + dt * P[1][1] + q * dt ** 3 / 2;
      const p11 = P[1][1] + q * dt * dt;
      tr.P = [[p00, p01], [p01, p11]];
      tr.matched = false;
    }
    const Rm = 0.015 ** 2;
    for (const d of det) {
      let best = null;
      for (const tr of this.tracks) {
        if (tr.matched) continue;
        const H = [1, dir * k];
        const zhat = tr.x[0] + dir * k * tr.x[1];
        const S = tr.P[0][0] + 2 * H[1] * tr.P[0][1] + H[1] * H[1] * tr.P[1][1] + Rm;
        const e = d.r - zhat;
        if (e * e / S < 9 && (!best || Math.abs(e) < Math.abs(best.e))) best = { tr, e, S, H };
      }
      if (best) {
        const { tr, e, S, H } = best;
        const PH = [tr.P[0][0] + H[1] * tr.P[0][1], tr.P[1][0] + H[1] * tr.P[1][1]];
        const K = [PH[0] / S, PH[1] / S];
        tr.x = [tr.x[0] + K[0] * e, tr.x[1] + K[1] * e];
        tr.P = [[tr.P[0][0] - K[0] * PH[0], tr.P[0][1] - K[0] * PH[1]], [tr.P[1][0] - K[1] * PH[0], tr.P[1][1] - K[1] * PH[1]]];
        tr.hits++; tr.misses = 0; tr.matched = true; tr.yaw = frame.yaw;
      } else if (this.tracks.length < 8) {
        this.tracks.push({ id: this.nextId++, x: [d.r, 0], P: [[0.03 ** 2, 0], [0, 1.0]], hits: 1, misses: 0, matched: true, yaw: frame.yaw });
      }
    }
    for (const tr of this.tracks) if (!tr.matched) tr.misses++;
    this.tracks = this.tracks.filter(tr => tr.misses <= 5);
  }

  walls() {
    const out = [];
    for (let i = 0; i < this.nW; i++) out.push({ phi: wrap(this.x[2 + 2 * i] + 0), d: this.x[3 + 2 * i], sD: Math.sqrt(Math.max(0, this.P[3 + 2 * i][3 + 2 * i])), sPhi: Math.sqrt(Math.max(0, this.P[2 + 2 * i][2 + 2 * i])) });
    return out;
  }
}

function medianProfile(list) {
  const n = list[0].length, m = new Float32Array(n), col = new Float64Array(list.length);
  for (let i = 0; i < n; i++) { for (let k = 0; k < list.length; k++) col[k] = list[k][i]; col.sort(); m[i] = col[list.length >> 1]; }
  return m;
}

// Normalised residual energy with 5 cm max-pooling so small pose shifts do not count as change. In [0, 1].
export function divergence(p, s, iMin, iMax) {
  let num = 0, den = 0;
  for (let i = iMin; i <= iMax; i += 1) {
    let a = 0, b = 0;
    for (let j = i - 2; j <= i + 2; j++) { if (p[j] > a) a = p[j]; if (s[j] > b) b = s[j]; }
    num += (a - b) * (a - b); den += a * a + b * b;
  }
  return den > 0 ? num / den : 0;
}

// CFAR on the clutter map: a detection must exceed 5 standard deviations of that bin's own history
// and twice the frame's noise floor.
function detections(p, s, v, iMin, iMax, binSize) {
  const res = new Float32Array(p.length);
  for (let i = iMin; i <= iMax; i++) { const d = p[i] - s[i]; res[i] = d > 0 ? d : 0; }
  const fl = median(p, iMin, iMax + 1) || 1e-9;
  const out = [];
  for (let i = iMin + 1; i < iMax; i++) {
    const thr = Math.max(2 * fl, 5 * Math.sqrt(v[i]));
    if (res[i] > thr && res[i] > res[i - 1] && res[i] >= res[i + 1]) {
      const a = res[i - 1], b = res[i], c = res[i + 1], den = a - 2 * b + c;
      const fr = den !== 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) : 0;
      out.push({ r: (i + fr) * binSize, amp: b });
    }
  }
  out.sort((x, y) => y.amp - x.amp);
  return out.slice(0, 4);
}
