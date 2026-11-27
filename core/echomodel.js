// Live echo front end (v0.6, SANDBOX.md): one chirp at a time, using only that chirp and earlier ones.
// For each echo candidate it builds features and a small model gives P(first-order wall echo). The same code
// runs in the app while the user turns and in the tools that train and test it.
//
// Per chirp: heading-local background from up to 24 earlier chirps within 45 deg, persistence peaks of the residual,
// two sub-band matched filters (lower and upper half of the sweep) for a spectral cue, continuity with the
// candidates of the last three chirps (a wall's range changes by at most hold radius x heading change), and
// the echo cosine when a second microphone is present. Time per chirp: O(bins x 40) for the background plus
// two extra matched filters; about 1 to 2 ms in Node.

import { EchoProcessor } from './dsp.js';
import { persistencePeaks, EST } from './estimate.js';
import { echoCos } from './stereo.js';

export const WIN = 6;
export const N_EFEAT = 16 + 2 * WIN + 1;
const RHO_MAX = 0.45;

function robustSigma(arr) {
  const a = Array.from(arr).sort((x, y) => x - y), m = a[a.length >> 1];
  const d = a.map(v => Math.abs(v - m)).sort((x, y) => x - y);
  return 1.4826 * d[d.length >> 1] || 1e-12;
}

// Small MLP: inputs standardised, tanh hidden layer, sigmoid output. json: { inputs, hidden, W1, b1, W2, b2, mean, std }.
export class SmallMLP {
  constructor(json) {
    this.json = json; this.F = json.inputs; this.H = json.hidden;
    this.W1 = Float32Array.from(json.W1); this.b1 = Float32Array.from(json.b1); this.W2 = Float32Array.from(json.W2); this.b2 = json.b2;
    this.mean = Float32Array.from(json.mean); this.std = Float32Array.from(json.std); this.h = new Float32Array(this.H);
  }
  // Time O(F x H).
  p(x) {
    const { F, H, W1, b1, W2, mean, std, h } = this;
    let o = this.b2;
    for (let k = 0; k < H; k++) {
      let s = b1[k];
      for (let i = 0; i < F; i++) s += W1[k * F + i] * ((x[i] - mean[i]) / std[i]);
      h[k] = Math.tanh(s); o += W2[k] * h[k];
    }
    return 1 / (1 + Math.exp(-o));
  }
}

export class EchoFront {
  // proc: the main EchoProcessor of the scan. opt.model: SmallMLP or null (then p = persistence weight).
  constructor(proc, opt = {}) {
    this.proc = proc; this.model = opt.model || null; this.bin = proc.binSize;
    const fm = 0.5 * (proc.f0 + proc.f1), half = proc.chirpDur / 2;
    const base = { fs: proc.fs, c: proc.c, rangeOffset: proc.rangeOffset, maxRange: proc.maxRange, binSize: proc.binSize };
    this.lo = new EchoProcessor({ ...base, f0: proc.f0, f1: fm, chirpDur: half });
    this.hi = new EchoProcessor({ ...base, f0: fm, f1: proc.f1, chirpDur: half });
    this.iMin = Math.round(EST.minRange / this.bin); this.iMax = Math.round(EST.maxRange / this.bin);
    this.hist = []; // { yaw, profile } of earlier chirps (lift and turn)
    this.prev = []; // candidate lists of the last three turn chirps: { yaw, list: [{ r, w }] }
    this.bwKHz = (proc.f1 - proc.f0) / 1000;
  }

  reset() { this.hist = []; this.prev = []; }

  // Heading-local background from earlier chirps within 45 deg (up to 24). Time O(bins x 24).
  background(yaw) {
    const near = [];
    for (let i = this.hist.length - 1; i >= 0 && near.length < 24; i--) {
      const h = this.hist[i];
      if (Math.abs(Math.atan2(Math.sin(h.yaw - yaw), Math.cos(h.yaw - yaw))) <= Math.PI / 4) near.push(h.profile);
    }
    if (near.length < 3) return null;
    const n = near[0].length, bg = new Float32Array(n), col = new Float32Array(near.length);
    for (let i = 0; i < n; i++) { for (let q = 0; q < near.length; q++) col[q] = near[q][i]; col.sort(); bg[i] = col[near.length >> 1]; }
    return bg;
  }

  // One chirp. f: { yaw, omega, profile, seg, dir, phase, far?, stereo? } (far: the second microphone's profile,
  // aligned as in core/stereo.js; stereo: { gap }). Returns candidates [{ r, score, w, p, cos, feat }].
  step(f) {
    const prof = f.profile;
    const bg = f.phase === 'turn' ? this.background(f.yaw) : null;
    this.hist.push({ yaw: f.yaw, profile: prof });
    if (this.hist.length > 200) this.hist.shift();
    if (!bg) return [];
    const n = prof.length, res = new Float32Array(n), dif = [];
    for (let j = 0; j < n; j++) { res[j] = Math.max(0, prof[j] - bg[j]); if (j >= this.iMin && j <= this.iMax) dif.push(prof[j] - bg[j]); }
    const sg = robustSigma(dif);
    const pk = persistencePeaks(res, this.iMin, this.iMax, 14);
    const rawSorted = Array.from(prof.slice(this.iMin, this.iMax + 1)).sort((a, b) => a - b);
    const floor = rawSorted[rawSorted.length >> 1] || 1e-9;
    let lo = null, hi = null, ratioMed = 0;
    if (f.seg) {
      lo = this.lo.process(f.seg, f.dir || 1).profile; hi = this.hi.process(f.seg, f.dir || 1).profile;
      const rs = [];
      for (let j = this.iMin; j <= this.iMax; j += 4) rs.push(Math.log10((lo[j] + 1e-6) / (hi[j] + 1e-6)));
      rs.sort((a, b) => a - b); ratioMed = rs[rs.length >> 1];
    }
    const out = [];
    pk.forEach((p, rank) => {
      const i = p.i, r = i * this.bin, s = p.v / sg, w = s / (s + 2);
      const x = new Float32Array(N_EFEAT);
      let j = 0;
      x[j++] = w; x[j++] = Math.log10(s + 1); x[j++] = Math.log10(prof[i] / floor + 1e-9); x[j++] = r / 6; x[j++] = rank / 14;
      const a = res[i] || 1e-9;
      for (let o = -WIN; o <= WIN; o++) { const v = (res[i + o] ?? 0) / a; x[j++] = v > 1.5 ? 1.5 : v; }
      // continuity with the last three turn chirps
      let cont = 0;
      for (let q = 1; q <= 3; q++) {
        const P = this.prev[this.prev.length - q];
        if (!P) { x[j++] = 3; continue; }
        const dth = Math.abs(Math.atan2(Math.sin(f.yaw - P.yaw), Math.cos(f.yaw - P.yaw)));
        const bound = RHO_MAX * dth + 0.02;
        let m = 3;
        for (const c of P.list) { const e = Math.abs(c.r - r) / bound; if (e < m) m = e; if (q === 1) cont = Math.max(cont, c.w * Math.exp(-e * e)); }
        x[j++] = m;
      }
      x[j++] = cont;
      x[j++] = Math.min(2, Math.abs(f.omega || 0));
      const lr = lo && hi ? Math.log10((lo[i] + 1e-6) / (hi[i] + 1e-6)) : 0;
      x[j++] = lr; x[j++] = lr - ratioMed;
      const cs = f.far && f.stereo ? echoCos(prof, f.far, r, f.stereo.gap, this.bin) : null;
      x[j++] = cs == null ? 0 : 1; x[j++] = cs == null ? 0 : cs;
      x[j++] = prof[i] > 0 ? res[i] / prof[i] : 0;
      x[j++] = this.bwKHz / 10;
      const prob = this.model ? this.model.p(x) : w;
      out.push({ r, score: s, w, p: prob, cos: cs, feat: x, rank });
    });
    this.prev.push({ yaw: f.yaw, list: out.map(c => ({ r: c.r, w: c.w })) });
    if (this.prev.length > 3) this.prev.shift();
    return out;
  }
}
