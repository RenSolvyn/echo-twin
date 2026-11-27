// Image-source room acoustics simulator with automatic ground truth.
// Convex prism rooms (vertical walls, flat floor and ceiling), reflections up to third order with
// face-validity checks, true Doppler from phone and target motion, transducer and chassis ringing,
// directivity, body shadow, late reverberation, noise and clipping. Everything is domain-randomized
// from a seed so every run is reproducible.

import { makeRng } from './rng.js';
import { fft } from './fft.js';
import { chirpAt, EchoProcessor } from './dsp.js';
import { BANDS, DEFAULTS, DEVICE_PROFILES, soundSpeed } from './config.js';
import { area, pointInConvex } from './geom.js';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const unit = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

function isConvexCCW(poly) {
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length], c = poly[(i + 2) % poly.length];
    const cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (cr <= 1e-6) return false;
  }
  return true;
}

function minSide(poly) {
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    m = Math.min(m, Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  return m;
}

function interiorAnglesOk(poly, lo, hi) {
  for (let i = 0; i < poly.length; i++) {
    const a = poly[(i + poly.length - 1) % poly.length], b = poly[i], c = poly[(i + 1) % poly.length];
    const u = [a[0] - b[0], a[1] - b[1]], v = [c[0] - b[0], c[1] - b[1]];
    const ang = Math.acos((u[0] * v[0] + u[1] * v[1]) / (Math.hypot(...u) * Math.hypot(...v)));
    if (ang < lo || ang > hi) return false;
  }
  return true;
}

function makeRoom(rng, kind, hard) {
  const maxSide = hard ? 8 : 7;
  for (let attempt = 0; attempt < 500; attempt++) {
    let poly;
    if (kind === 'rect' || kind === 'quad') {
      const W = rng.uniform(2.5, maxSide), L = rng.uniform(2.5, maxSide);
      poly = [[-W / 2, -L / 2], [W / 2, -L / 2], [W / 2, L / 2], [-W / 2, L / 2]];
      if (kind === 'quad') poly = poly.map(([x, y]) => [x + rng.uniform(-0.6, 0.6), y + rng.uniform(-0.6, 0.6)]);
      const a = rng.uniform(0, Math.PI / 2);
      poly = poly.map(([x, y]) => [Math.cos(a) * x - Math.sin(a) * y, Math.sin(a) * x + Math.cos(a) * y]);
      if (kind === 'quad' && !interiorAnglesOk(poly, 55 * Math.PI / 180, 125 * Math.PI / 180)) continue;
    } else {
      const n = rng.int(5, 6);
      const ax = rng.uniform(1.8, hard ? 4 : 3.4), ay = rng.uniform(1.6, hard ? 3.6 : 3.0);
      const a0 = rng.uniform(0, 2 * Math.PI);
      poly = [];
      for (let i = 0; i < n; i++) {
        const t = a0 + (i + rng.uniform(-0.22, 0.22)) * 2 * Math.PI / n;
        poly.push([ax * Math.cos(t), ay * Math.sin(t)]);
      }
    }
    if (!isConvexCCW(poly)) continue;
    if (minSide(poly) < (kind === 'poly' ? 1.2 : 1.5)) continue;
    return poly;
  }
  return [[-2, -1.5], [2, -1.5], [2, 1.5], [-2, 1.5]];
}

function wallDistance(poly, x, y) {
  let m = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
    const s = (-(dy) * (x - a[0]) + dx * (y - a[1])) / L; // left normal, inside > 0
    m = Math.min(m, s);
  }
  return m;
}

export function randomScene(seed, opt = {}) {
  const rng = makeRng(seed * 7919 + 17);
  const hard = !!opt.hard;
  const r = rng.next();
  const kind = opt.kind || (r < 0.55 ? 'rect' : r < 0.8 ? 'quad' : 'poly');
  const poly = makeRoom(rng, kind, hard);
  const H = rng.uniform(2.3, 3.2);
  const rho = hard ? rng.uniform(0.10, 0.45) : rng.uniform(0.15, 0.45);
  // rotation centre: inside, far enough from every wall
  let c0 = null;
  const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
  for (let i = 0; i < 400 && !c0; i++) {
    const x = rng.uniform(Math.min(...xs), Math.max(...xs)), y = rng.uniform(Math.min(...ys), Math.max(...ys));
    if (pointInConvex(poly, x, y) && wallDistance(poly, x, y) >= Math.max(0.9, rho + 0.55)) c0 = [x, y];
  }
  if (!c0) c0 = [xs.reduce((a, b) => a + b) / xs.length, ys.reduce((a, b) => a + b) / ys.length];
  const z0 = Math.min(rng.uniform(1.0, 1.45), H - 0.75);

  const band = opt.band ? BANDS[opt.band] : (rng.chance(0.75) ? BANDS.high : BANDS.audible);
  const cls = opt.deviceClass || (rng.chance(0.6) ? 'android' : 'iphone');
  const nModes = rng.int(2, 4), modes = [];
  for (let i = 0; i < nModes; i++) {
    modes.push({ f: rng.uniform(band.f0 - 1500, band.f1 + 1500), tau: rng.uniform(0.0001, 0.0015), a: rng.uniform(0.1, 0.8) * (rng.chance(0.5) ? 1 : -1) });
  }
  const chassis = [];
  for (let i = 0, n = rng.int(1, 2); i < n; i++) {
    chassis.push({ f: rng.uniform(band.f0 - 1000, band.f1 + 1000), tau: rng.uniform(0.0005, 0.004), a: rng.uniform(0.05, 0.4) });
  }
  const device = {
    cls,
    b: cls === 'android' ? rng.uniform(0.008, 0.03) : rng.uniform(0.12, 0.16),
    txUp: cls === 'iphone' ? rng.uniform(0.5, 1) : 0,
    txIn: cls === 'iphone' ? rng.uniform(0, 0.7) : 0,
    betaTx: cls === 'android' ? rng.uniform(0.3, 0.9) : rng.uniform(0.1, 0.6),
    gammaTx: rng.uniform(1, 2.5),
    betaRx: rng.uniform(0, 0.5),
    shadow: rng.uniform(0.25, 0.8),
    modes, chassis,
    coupling: rng.uniform(0.4, 2.5),
    clip: rng.chance(0.3) ? rng.uniform(0.4, 0.9) : 0,
    rangeOffset: DEVICE_PROFILES[cls].rangeOffset
  };

  const nWalls = poly.length;
  const amax = hard ? 0.75 : 0.6;
  const alphaWalls = Array.from({ length: nWalls }, () => rng.uniform(0.05, amax));
  const nObj = rng.int(0, hard ? 12 : 8);
  const objects = [];
  for (let i = 0; i < nObj * 30 && objects.length < nObj; i++) {
    const x = rng.uniform(Math.min(...xs), Math.max(...xs)), y = rng.uniform(Math.min(...ys), Math.max(...ys));
    if (!pointInConvex(poly, x, y) || wallDistance(poly, x, y) < 0.15) continue;
    if (Math.hypot(x - c0[0], y - c0[1]) < rho + 0.5) continue;
    objects.push({ p: [x, y, rng.uniform(0.3, 1.2)], sigma: rng.uniform(0.01, hard ? 0.25 : 0.12) });
  }
  const tempC = rng.uniform(15, 28);
  const drift = (hard ? 12 : 6) * Math.PI / 180 * rng.uniform(-1, 1);
  const sway = [0, 1].map(() => ({ ax: rng.uniform(0.005, 0.02), ay: rng.uniform(0.005, 0.02), w: 2 * Math.PI / rng.uniform(2, 5), ph: rng.uniform(0, 6.3) }));
  return {
    seed, kind, hard, poly, H, c0, z0, rho, band, device,
    alphaWalls, alphaFloor: rng.uniform(0.05, 0.7), alphaCeil: rng.uniform(0.05, 0.5),
    objects,
    body: { sigma: rng.uniform(0.05, 0.25), head: rng.uniform(0.02, 0.1), ghost: rng.uniform(0.05, 0.25) },
    rt60: rng.uniform(0.2, 0.9), revK: rng.uniform(0.3, 1.5),
    echoSnrDb: hard ? rng.uniform(5, 25) : rng.uniform(10, 30),
    tempC, tempAssumed: DEFAULTS.tempC,
    imuOffset: rng.uniform(-Math.PI, Math.PI), drift, headingNoise: 0.5 * Math.PI / 180,
    liftScaleErr: rng.uniform(0.85, 1.15), liftNoise: rng.uniform(0.005, 0.02), liftNoisePh: rng.uniform(0, 6.3),
    liftDur: 2.5, liftDz: rng.uniform(0.25, 0.45),
    turnDur: rng.uniform(16, 26), turnTotal: 2 * Math.PI + rng.uniform(0.15, 0.6), turnSign: rng.chance(0.5) ? 1 : -1,
    turnWobble: rng.uniform(0, 0.6), turnWobbleN: rng.int(1, 3), psi0: rng.uniform(-Math.PI, Math.PI),
    sway, latencyJitter: 0.0005,
    period: DEFAULTS.period, chirpDur: DEFAULTS.chirpDur,
    movers: []
  };
}

// Planes in 3D: inward normal n and a point q. Walls carry their edge for face checks.
function buildPlanes(scene) {
  const planes = [];
  const P = scene.poly;
  for (let i = 0; i < P.length; i++) {
    const a = P[i], b = P[(i + 1) % P.length];
    const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
    planes.push({ type: 'wall', idx: i, n: [-dy / L, dx / L, 0], q: [a[0], a[1], 0], a, b, refl: Math.sqrt(1 - scene.alphaWalls[i]) });
  }
  planes.push({ type: 'floor', idx: P.length, n: [0, 0, 1], q: [0, 0, 0], refl: Math.sqrt(1 - scene.alphaFloor) });
  planes.push({ type: 'ceil', idx: P.length + 1, n: [0, 0, -1], q: [0, 0, scene.H], refl: Math.sqrt(1 - scene.alphaCeil) });
  return planes;
}

const sdist = (pl, x) => dot(pl.n, sub(x, pl.q));
const reflect = (pl, x) => sub(x, mul(pl.n, 2 * sdist(pl, x)));

function onFace(pl, X, scene) {
  const eps = 1e-6;
  if (pl.type === 'wall') {
    if (X[2] < -eps || X[2] > scene.H + eps) return false;
    const dx = pl.b[0] - pl.a[0], dy = pl.b[1] - pl.a[1];
    const t = ((X[0] - pl.a[0]) * dx + (X[1] - pl.a[1]) * dy) / (dx * dx + dy * dy);
    return t >= -eps && t <= 1 + eps;
  }
  return pointInConvex(scene.poly, X[0], X[1]);
}

// Valid specular paths from S to M up to maxOrder reflections.
function imagePaths(scene, planes, S, M, maxOrder = 3) {
  const out = [];
  const rec = (I, seq, imgs, refl) => {
    if (seq.length > 0) {
      // backtrack from M through the reflection points
      let prev = M, ok = true;
      const pts = [];
      for (let m = seq.length - 1; m >= 0; m--) {
        const pl = planes[seq[m]];
        const target = imgs[m];
        const s0 = sdist(pl, prev), s1 = sdist(pl, target);
        if (!(s0 > 0 && s1 < 0)) { ok = false; break; }
        const t = s0 / (s0 - s1);
        const X = add(prev, mul(sub(target, prev), t));
        if (!onFace(pl, X, scene)) { ok = false; break; }
        pts.push(X);
        prev = X;
      }
      if (ok) {
        const L = len(sub(M, I));
        out.push({ seq: seq.slice(), L, refl, dIn: unit(sub(pts[0], M)), dOut: unit(sub(pts[pts.length - 1], S)) });
      }
    }
    if (seq.length >= maxOrder || refl < 0.08) return;
    for (let k = 0; k < planes.length; k++) {
      if (seq.length && seq[seq.length - 1] === k) continue;
      const I2 = reflect(planes[k], I);
      rec(I2, seq.concat(k), imgs.concat([I2]), refl * planes[k].refl);
    }
  };
  rec(S, [], [], 1);
  return out;
}

function classify(seq, planes) {
  const walls = seq.filter(k => planes[k].type === 'wall');
  const nh = seq.length - walls.length;
  const nw = walls.length;
  if (nw === 0) return 'H';
  if (nw === 1 && nh === 0) return 'wall1';
  if (nw === 1) return 'wallH';
  if (nw === 2 && nh === 0) {
    const n = planes.filter(p => p.type === 'wall').length;
    const [a, b] = walls;
    return (Math.abs(a - b) === 1 || Math.abs(a - b) === n - 1) ? 'corner' : 'wall2';
  }
  if (nw === 2 && nh === 1) return 'tri';
  return 'high';
}

export function poseAt(scene, t) {
  const { liftDur, liftDz, turnDur, turnTotal, turnSign, psi0, c0, z0, rho } = scene;
  let psi = psi0, z = z0, phase = 'turn';
  if (t < liftDur) {
    phase = 'lift';
    const x = t / liftDur, s = x * x * (3 - 2 * x);
    z = z0 - liftDz * (1 - s);
  } else {
    const x = Math.min(1, (t - liftDur) / turnDur);
    const w = scene.turnWobble, m = scene.turnWobbleN;
    const s = x + w * Math.sin(2 * Math.PI * m * x) / (2 * Math.PI * m) * 0.9;
    psi = psi0 + turnSign * turnTotal * s;
  }
  const sw = scene.sway;
  // sway in the frame of the starting heading, so rotating the whole scene rotates the sway too
  const sx = sw[0].ax * Math.sin(sw[0].w * t + sw[0].ph) + sw[1].ax * Math.sin(sw[1].w * t + sw[1].ph);
  const sy = sw[0].ay * Math.cos(sw[0].w * t + sw[0].ph) + sw[1].ay * Math.cos(sw[1].w * t + sw[1].ph);
  const c0s = Math.cos(psi0), s0s = Math.sin(psi0);
  const cx = c0[0] + c0s * sx - s0s * sy;
  const cy = c0[1] + s0s * sx + c0s * sy;
  const u = [Math.cos(psi), Math.sin(psi), 0];
  const pos = [cx + rho * u[0], cy + rho * u[1], z];
  return { t, psi, pos, u, phase, center: [cx, cy] };
}

function directivity(cosA, beta, gamma) {
  return Math.pow((1 - beta) + beta * 0.5 * (1 + cosA), gamma);
}

export class Simulator {
  constructor(scene, opt = {}) {
    this.scene = scene;
    this.fs = opt.fs || 48000;
    this.cTrue = soundSpeed(scene.tempC);
    this.cAssumed = soundSpeed(scene.tempAssumed);
    this.proc = new EchoProcessor({
      fs: this.fs, f0: scene.band.f0, f1: scene.band.f1, chirpDur: scene.chirpDur,
      tempC: scene.tempAssumed, rangeOffset: scene.device.rangeOffset
    });
    this.planes = buildPlanes(scene);
    this.rng = makeRng(scene.seed * 31 + 5);
    const N = this.proc.N, fs = this.fs;
    this.N = N;
    // transducer ringing impulse response
    const ring = new Float64Array(N), ringI = new Float64Array(N);
    ring[0] = 1;
    for (let m = 1; m < Math.round(0.012 * fs); m++) {
      let v = 0;
      for (const md of scene.device.modes) v += md.a * Math.exp(-m / fs / md.tau) * Math.sin(2 * Math.PI * md.f * m / fs) * 0.1;
      ring[m] = v;
    }
    fft(ring, ringI);
    this.ring = { re: ring, im: ringI };
    this.chassis = new Float64Array(Math.round(0.015 * fs));
    for (let m = 1; m < this.chassis.length; m++) {
      let v = 0;
      for (const md of scene.device.chassis) v += md.a * Math.exp(-m / fs / md.tau) * Math.sin(2 * Math.PI * md.f * m / fs);
      this.chassis[m] = v;
    }
    // chirp spectra per sweep direction
    this.chirpSpec = {};
    for (const dir of [1, -1]) {
      const re = new Float64Array(N), im = new Float64Array(N);
      for (let i = 0; i < Math.round(scene.chirpDur * fs); i++) re[i] = chirpAt(i / fs, scene.band.f0, scene.band.f1, scene.chirpDur, dir);
      fft(re, im);
      this.chirpSpec[dir] = { re, im };
    }
    this.airDbPerM = 0.5 * (scene.band.f0 + scene.band.f1) > 12000 ? 0.3 : 0.06;
    // reference first-order echo amplitude of a wall at 3 m, for noise and reverb levels
    const meanRefl = scene.alphaWalls.reduce((a, b) => a + Math.sqrt(1 - b), 0) / scene.alphaWalls.length;
    this.aRef = 0.5 * meanRefl / 6;
    const chirpEnergy = 0.5 * scene.chirpDur * fs * 0.85;
    this.noiseSigma = this.aRef * Math.sqrt(chirpEnergy / Math.pow(10, scene.echoSnrDb / 10));
  }

  phoneGeom(pose) {
    const d = this.scene.device, u = pose.u;
    const lat = [-u[1], u[0], 0];
    const M = add(pose.pos, mul(u, 0.07));
    const S = d.cls === 'iphone' ? sub(M, mul(u, d.b)) : add(M, mul(lat, d.b));
    const txAxis = d.cls === 'iphone' ? unit(add(mul([0, 0, 1], d.txUp), mul(u, -d.txIn))) : u;
    return { S, M, u, txAxis };
  }

  gains(g, dOut, dIn) {
    const d = this.scene.device;
    let a = directivity(dot(dOut, g.txAxis), d.betaTx, d.gammaTx) * directivity(dot(dIn, g.u), d.betaRx, 1);
    const back = Math.cos(50 * Math.PI / 180);
    if (dot(dOut, mul(g.u, -1)) > back) a *= d.shadow;
    if (dot(dIn, mul(g.u, -1)) > back) a *= d.shadow;
    return a;
  }

  // Synthesizes one recorded segment for a chirp emitted at time t with sweep direction dir.
  // Returns { seg, arrivals, pose }. Arrival ranges use the estimator's convention (assumed sound
  // speed, assumed range offset, Doppler coupling), so labels sit where the estimator should see peaks.
  frame(t, dir = 1, extra = {}) {
    const sc = this.scene, fs = this.fs, N = this.N, proc = this.proc;
    const pose = extra.pose || poseAt(sc, t);
    const pose2 = extra.pose ? extra.pose : poseAt(sc, t + 0.01);
    const vPhone = extra.pose ? [0, 0, 0] : mul(sub(pose2.pos, pose.pos), 100);
    const g = this.phoneGeom(pose);
    const Ld = len(sub(g.M, g.S));
    const jitter = this.rng.uniform(-1, 1) * sc.latencyJitter * fs;
    const nDirect = proc.pre + jitter;
    const tEmit = nDirect - Ld / this.cTrue * fs; // emission time in segment samples
    const kappa = proc.kappa;
    const toR = (L, Ldot) => (L - Ld) / 2 * this.cAssumed / this.cTrue + sc.device.rangeOffset + dir * kappa * (Ldot / 2);
    const air = (L) => Math.pow(10, -this.airDbPerM * L / 20);

    const arr = [];
    const paths = imagePaths(sc, this.planes, g.S, g.M, 3);
    for (const p of paths) {
      const a = p.refl * this.gains(g, p.dOut, p.dIn) / p.L * air(p.L);
      const Ldot = -dot(add(p.dOut, p.dIn), vPhone);
      arr.push({ L: p.L, Ldot, amp: a, kind: classify(p.seq, this.planes), seq: p.seq });
    }
    // first-order wall ghosts through the user's body
    for (const p of arr.slice()) {
      if (p.kind !== 'wall1') continue;
      const refl = this.planes[p.seq[0]].refl;
      arr.push({ L: 2 * p.L + 0.2, Ldot: 2 * p.Ldot, amp: p.amp * refl * sc.body.ghost / 2, kind: 'ghost', seq: p.seq });
    }
    const scatter = (o, sigma, vo, kind) => {
      const d1 = sub(o, g.S), d2 = sub(o, g.M);
      const L1 = len(d1), L2 = len(d2);
      const dOut = unit(d1), dIn = unit(d2);
      const a = sigma * this.gains(g, dOut, dIn) / (L1 * L2) * air(L1 + L2);
      const Ldot = dot(add(dOut, dIn), sub(vo, vPhone));
      arr.push({ L: L1 + L2, Ldot, amp: a, kind });
    };
    for (const o of sc.objects) scatter(o.p, o.sigma, [0, 0, 0], 'obj');
    const torso = [pose.center[0] + 0.12 * pose.u[0], pose.center[1] + 0.12 * pose.u[1], pose.pos[2] - 0.05];
    scatter(torso, sc.body.sigma, mul(vPhone, 0.4), 'body');
    scatter([pose.center[0], pose.center[1], 1.62], sc.body.head, [0, 0, 0], 'head');
    for (const mv of (extra.movers || sc.movers)) {
      if (t < mv.t0 || t > mv.t1) continue;
      const pos = add(mv.p0, mul(mv.v, t - mv.t0));
      scatter(pos, mv.sigma, mv.v, 'mover');
    }

    // time-domain synthesis of discrete arrivals with exact Doppler time scaling
    const D = new Float64Array(N), Di = new Float64Array(N);
    const T = sc.chirpDur, f0 = sc.band.f0, f1 = sc.band.f1;
    const ampFloor = this.aRef * 2e-3;
    const Ad = sc.device.coupling / Math.max(Ld, 0.02);
    const synth = (L, Ldot, A) => {
      const start = tEmit + L / this.cTrue * fs;
      const scale = 1 - Ldot / this.cTrue;
      const i0 = Math.max(0, Math.ceil(start)), i1 = Math.min(proc.segLen - 1, Math.floor(start + T * fs / scale) + 1);
      for (let i = i0; i <= i1; i++) D[i] += A * chirpAt((i - start) / fs * scale, f0, f1, T, dir);
    };
    synth(Ld, 0, Ad);
    for (const a of arr) if (a.amp > ampFloor) synth(a.L, a.Ldot, a.amp);

    // impulse-train part: chassis ringing of the direct path and diffuse late reverberation
    const Hr = new Float64Array(N), Hi = new Float64Array(N);
    const nd = Math.round(nDirect);
    for (let m = 1; m < this.chassis.length && nd + m < proc.segLen; m++) Hr[nd + m] += Ad * this.chassis[m] * 0.05;
    let minL = Infinity;
    for (const a of arr) if (a.kind === 'wall1' || a.kind === 'H') minL = Math.min(minL, a.L);
    if (!isFinite(minL)) minL = 2;
    const nMix = Math.round(nDirect + (minL - Ld) / this.cTrue * fs);
    const sRev = sc.revK * this.aRef / Math.sqrt(fs * 0.01);
    for (let i = Math.max(0, nMix); i < proc.segLen; i++) {
      Hr[i] += sRev * this.rng.normal() * Math.exp(-6.91 * (i - nDirect) / fs / sc.rt60);
    }
    fft(D, Di); fft(Hr, Hi);
    const C = this.chirpSpec[dir], R = this.ring;
    for (let k = 0; k < N; k++) {
      const xr = D[k] + (Hr[k] * C.re[k] - Hi[k] * C.im[k]);
      const xi = Di[k] + (Hr[k] * C.im[k] + Hi[k] * C.re[k]);
      D[k] = xr * R.re[k] - xi * R.im[k];
      Di[k] = xr * R.im[k] + xi * R.re[k];
    }
    fft(D, Di, true);
    const seg = new Float32Array(proc.segLen);
    const clip = sc.device.clip ? sc.device.clip * Ad : Infinity;
    for (let i = 0; i < proc.segLen; i++) {
      let v = D[i] + this.noiseSigma * this.rng.normal();
      if (v > clip) v = clip; else if (v < -clip) v = -clip;
      seg[i] = v;
    }
    // ground-truth arrivals in estimator units
    const arrivals = arr.map(a => ({ r: toR(a.L, a.Ldot), rdot: a.Ldot / 2, amp: a.amp, kind: a.kind, seq: a.seq }));
    return { seg, arrivals, pose };
  }

  imuYaw(t, psi) {
    const sc = this.scene;
    const x = Math.max(0, Math.min(1, (t - sc.liftDur) / sc.turnDur));
    return psi + sc.imuOffset + sc.drift * x + sc.headingNoise * this.rng.normal();
  }

  // Full guided scan: lift then turn. Each frame carries the IMU heading the app would see.
  scan() {
    const sc = this.scene;
    const total = sc.liftDur + sc.turnDur + 0.3;
    const frames = [];
    for (let k = 0; k * sc.period < total; k++) {
      const t = k * sc.period;
      const f = this.frame(t, 1);
      const yaw = this.imuYaw(t, f.pose.psi);
      const fr = { t, yaw, phase: f.pose.phase, dir: 1, seg: f.seg, arrivals: f.arrivals, pose: f.pose, source: 'sim' };
      if (fr.phase === 'lift') fr.dz = (f.pose.pos[2] - sc.z0) * sc.liftScaleErr + sc.liftNoise * Math.sin(2.1 * t + sc.liftNoisePh);
      frames.push(fr);
    }
    return frames;
  }

  // Live radar mode: the phone held still (small sway) at heading psi, chirps every period with
  // alternating sweep direction, people walking through. Optional narrowband interferer (Hz, level).
  live({ psi = this.scene.psi0, duration = 10, period = 0.06, movers = [], t0 = 100, interferer = null } = {}) {
    const sc = this.scene;
    const frames = [];
    const rng = makeRng(sc.seed * 13 + 7);
    for (let k = 0; k * period < duration; k++) {
      const t = t0 + k * period;
      const dir = k % 2 === 0 ? 1 : -1;
      const sw = 0.004 * Math.sin(1.3 * t);
      const u = [Math.cos(psi), Math.sin(psi), 0];
      const pose = { t, psi, u, pos: [sc.c0[0] + sc.rho * u[0] + sw, sc.c0[1] + sc.rho * u[1], sc.z0], phase: 'live', center: sc.c0 };
      const f = this.frame(t, dir, { pose, movers });
      if (interferer) {
        for (let i = 0; i < f.seg.length; i++) f.seg[i] += interferer.level * this.aRef * Math.sin(2 * Math.PI * interferer.f * (t + i / this.fs));
      }
      const yaw = psi + sc.imuOffset + sc.drift + sc.headingNoise * rng.normal();
      frames.push({ t, yaw, dir, phase: 'live', seg: f.seg, arrivals: f.arrivals, pose, source: 'sim' });
    }
    return frames;
  }

  // Ground truth in the rotation-centre frame (world axes; the estimator frame differs by a rotation).
  truth() {
    const sc = this.scene;
    const walls = [];
    const P = sc.poly;
    for (let i = 0; i < P.length; i++) {
      const a = P[i], b = P[(i + 1) % P.length];
      const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
      const nOut = [dy / L, -dx / L];
      const d = nOut[0] * (a[0] - sc.c0[0]) + nOut[1] * (a[1] - sc.c0[1]);
      const foot = [sc.c0[0] + d * nOut[0], sc.c0[1] + d * nOut[1]];
      const t = ((foot[0] - a[0]) * dx + (foot[1] - a[1]) * dy) / (L * L);
      walls.push({ phi: Math.atan2(nOut[1], nOut[0]), d, observable: t > 0.03 && t < 0.97 });
    }
    return {
      walls,
      poly: P.map(([x, y]) => [x - sc.c0[0], y - sc.c0[1]]),
      floor: sc.z0, ceil: sc.H - sc.z0, H: sc.H,
      area: Math.abs(area(P)),
      imuOffset: sc.imuOffset
    };
  }
}
