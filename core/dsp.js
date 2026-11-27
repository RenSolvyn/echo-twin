// Chirp generation and the matched-filter echo processor shared by the app, the simulator and the tools.
// A frame is one recorded segment around one chirp; it becomes a range profile (echo envelope per range bin).

import { fft, nextPow2 } from './fft.js';
import { DEFAULTS, soundSpeed } from './config.js';

export function tukey(x, a) {
  if (x < 0 || x > 1) return 0;
  if (x < a / 2) return 0.5 * (1 - Math.cos(2 * Math.PI * x / a));
  if (x > 1 - a / 2) return 0.5 * (1 - Math.cos(2 * Math.PI * (1 - x) / a));
  return 1;
}

// Continuous chirp value at time t (s) after its start. dir = +1 up-sweep, -1 down-sweep.
export function chirpAt(t, f0, f1, dur, dir = 1, taper = 0.15) {
  if (t < 0 || t > dur) return 0;
  const a = dir > 0 ? f0 : f1, b = dir > 0 ? f1 : f0;
  const k = (b - a) / dur;
  return tukey(t / dur, taper) * Math.sin(2 * Math.PI * (a * t + 0.5 * k * t * t));
}

export function makeChirp(fs, f0, f1, dur, dir = 1, taper = 0.15) {
  const n = Math.round(fs * dur);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = chirpAt(i / fs, f0, f1, dur, dir, taper);
  return x;
}

// Range bias per unit range-rate for a linear FM sweep: an echo receding at rdot (m/s)
// appears at r + dir * kappa * rdot, kappa = f_center * T / B.
export function dopplerKappa(f0, f1, dur) {
  return 0.5 * (f0 + f1) * dur / Math.abs(f1 - f0);
}

function median(arr, lo, hi) {
  const n = hi - lo;
  if (n <= 0) return 0;
  const tmp = Array.prototype.slice.call(arr, lo, hi).sort((a, b) => a - b);
  return tmp[n >> 1];
}

export class EchoProcessor {
  // o: { fs, f0, f1, chirpDur, preSec, maxRange, binSize, tempC | c, rangeOffset }
  constructor(o) {
    const d = DEFAULTS;
    this.fs = o.fs;
    this.f0 = o.f0; this.f1 = o.f1;
    this.chirpDur = o.chirpDur ?? d.chirpDur;
    this.preSec = o.preSec ?? d.preSec;
    this.maxRange = o.maxRange ?? d.maxRange;
    this.binSize = o.binSize ?? d.binSize;
    this.c = o.c ?? soundSpeed(o.tempC ?? d.tempC);
    this.rangeOffset = o.rangeOffset ?? 0;
    // PHAT-beta weighting (0 = plain matched filter): flattens the cross-spectrum magnitude inside the band so
    // every frequency votes by phase, not by level. Off by default; measured in tools/frontend.mjs.
    this.phatBeta = o.phatBeta ?? 0;
    this.pre = Math.round(this.preSec * this.fs);
    this.post = Math.ceil((2 * (this.maxRange + 0.2) / this.c + this.chirpDur + 0.003) * this.fs);
    this.segLen = this.pre + this.post;
    this.chirps = { 1: makeChirp(this.fs, this.f0, this.f1, this.chirpDur, 1), [-1]: makeChirp(this.fs, this.f0, this.f1, this.chirpDur, -1) };
    this.N = nextPow2(this.segLen + this.chirps[1].length);
    this.nBins = Math.round(this.maxRange / this.binSize) + 1;
    this.kappa = dopplerKappa(this.f0, this.f1, this.chirpDur);
    // Receiver-side Hamming weighting of the reference sweep (a mismatched filter): range sidelobes
    // fall from about -13 dB to about -40 dB, so a strong near echo (the user's body) does not leak
    // over the floor and the walls. Costs about 1.5x mainlobe width; peak interpolation keeps precision.
    this.spec = {};
    for (const dir of [1, -1]) {
      const re = new Float64Array(this.N), im = new Float64Array(this.N);
      const c = this.chirps[dir];
      const L = c.length;
      for (let i = 0; i < L; i++) re[i] = c[i] * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (L - 1)));
      fft(re, im);
      this.spec[dir] = { re, im };
    }
    this.re = new Float64Array(this.N); this.im = new Float64Array(this.N);
    this.xr = new Float64Array(this.N); this.xi = new Float64Array(this.N);
    this.env = new Float64Array(this.N);
  }

  // Interference whitening: learn the noise spectrum from chirp-free audio (the gap before the next
  // chirp) and down-weight frequencies that carry interference (hum harmonics, other ultrasonic sources).
  learnNoise(seg) {
    const { N } = this;
    const re = new Float64Array(N), im = new Float64Array(N);
    const n = Math.min(seg.length, N);
    for (let i = 0; i < n; i++) re[i] = seg[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    fft(re, im);
    if (!this.noisePsd) this.noisePsd = new Float64Array(N);
    const a = this.noiseCount ? 0.1 : 1;
    for (let k = 0; k < N; k++) this.noisePsd[k] = (1 - a) * this.noisePsd[k] + a * (re[k] * re[k] + im[k] * im[k]);
    this.noiseCount = (this.noiseCount || 0) + 1;
    // weights inside the sweep band, relative to the band median, clipped to [0.1, 1]
    const k0 = Math.floor(this.f0 / this.fs * N), k1 = Math.ceil(this.f1 / this.fs * N);
    const band = Array.from(this.noisePsd.slice(k0, k1 + 1)).sort((x, y) => x - y);
    const med = band[band.length >> 1] || 1e-30;
    this.whiten = new Float64Array(N).fill(1);
    for (let k = 0; k < N; k++) {
      const kk = k <= N / 2 ? k : N - k;
      if (kk < k0 || kk > k1) continue;
      this.whiten[k] = Math.max(0.1, Math.min(1, Math.sqrt(med / (this.noisePsd[kk] || med))));
    }
  }

  // Envelope of the matched-filter output (analytic signal magnitude) for a segment.
  envelope(seg, dir = 1) {
    const { N, re, im, xr, xi, env } = this;
    xr.fill(0); xi.fill(0);
    const n = Math.min(seg.length, this.segLen);
    for (let i = 0; i < n; i++) xr[i] = seg[i];
    fft(xr, xi);
    const S = this.spec[dir];
    const half = N >> 1;
    for (let k = 0; k < N; k++) {
      let g = 0;
      if (k === 0 || k === half) g = 1; else if (k < half) g = 2;
      // correlation: X * conj(C)
      const wk = this.whiten ? g * this.whiten[k] : g;
      re[k] = wk * (xr[k] * S.re[k] + xi[k] * S.im[k]);
      im[k] = wk * (xi[k] * S.re[k] - xr[k] * S.im[k]);
    }
    if (this.phatBeta > 0) {
      const k0 = Math.floor(this.f0 / this.fs * N), k1 = Math.ceil(this.f1 / this.fs * N);
      const mags = [];
      for (let k = k0; k <= k1; k++) mags.push(Math.hypot(re[k], im[k]));
      mags.sort((a, b) => a - b);
      const eps = 1e-3 * Math.pow(mags[mags.length >> 1] || 1e-30, this.phatBeta);
      for (let k = 0; k < N; k++) {
        if (k < k0 || k > k1) { re[k] = 0; im[k] = 0; continue; }
        const w = 1 / (Math.pow(Math.hypot(re[k], im[k]), this.phatBeta) + eps);
        re[k] *= w; im[k] *= w;
      }
    }
    fft(re, im, true);
    for (let k = 0; k < N; k++) env[k] = Math.hypot(re[k], im[k]);
    return env;
  }

  // Returns { profile (Float32Array, envelope / direct amplitude per range bin), n0, direct, snrDb }.
  // Time: O(N log N) with N = next power of two of the segment. Space: O(N).
  process(seg, dir = 1) {
    const env = this.envelope(seg, dir);
    const fs = this.fs;
    const searchEnd = Math.min(this.segLen - 2, 2 * this.pre + Math.round(0.002 * fs));
    let n0 = 1, best = -1;
    for (let i = 1; i < searchEnd; i++) if (env[i] > best) { best = env[i]; n0 = i; }
    const a = env[n0 - 1], b = env[n0], c = env[n0 + 1];
    const den = a - 2 * b + c;
    const frac = den !== 0 ? 0.5 * (a - c) / den : 0;
    const n0f = n0 + Math.max(-0.5, Math.min(0.5, frac));
    const direct = b;
    const spm = 2 * fs / this.c; // samples per metre of range
    const profile = new Float32Array(this.nBins);
    const half = this.binSize / 2;
    for (let j = 0; j < this.nBins; j++) {
      const r = j * this.binSize - this.rangeOffset;
      const l0 = n0f + (r - half) * spm, l1 = n0f + (r + half) * spm;
      let i0 = Math.ceil(l0), i1 = Math.floor(l1);
      if (i1 < i0) { i0 = i1 = Math.round(0.5 * (l0 + l1)); }
      let m = 0;
      for (let i = i0; i <= i1; i++) if (i >= 0 && i < this.segLen && env[i] > m) m = env[i];
      profile[j] = direct > 0 ? m / direct : 0;
    }
    const lo = Math.round(DEFAULTS.minRange / this.binSize);
    const floor = median(profile, lo, this.nBins);
    const snrDb = floor > 0 ? -20 * Math.log10(floor) : 0;
    return { profile, n0: n0f, direct, snrDb, dir };
  }
}

// Local maxima above a multiple of the median floor, refined to sub-bin precision.
// Returns peaks sorted by amplitude, at most maxPeaks.
export function findPeaks(p, iMin, iMax, floorMul = 4, maxPeaks = 14) {
  iMin = Math.max(1, iMin); iMax = Math.min(p.length - 2, iMax);
  const floor = median(p, iMin, iMax + 1) || 1e-9;
  const thr = floor * floorMul;
  const out = [];
  for (let i = iMin; i <= iMax; i++) {
    const v = p[i];
    if (v > thr && v > p[i - 1] && v >= p[i + 1]) {
      const a = p[i - 1], c = p[i + 1];
      const den = a - 2 * v + c;
      const frac = den !== 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) : 0;
      out.push({ i: i + frac, bin: i, amp: v, floor });
    }
  }
  out.sort((x, y) => y.amp - x.amp);
  // keep peaks at least 2 bins apart (strongest first)
  const kept = [];
  for (const pk of out) {
    if (kept.every(k => Math.abs(k.i - pk.i) >= 2)) kept.push(pk);
    if (kept.length >= maxPeaks) break;
  }
  return kept;
}

export { median };
