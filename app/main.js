// Echo Twin app controller: guided scan, progressive solving in a worker, live twin and Doppler radar,
// simulated-room demo, recording export and replay. v0.5: floor plan view, live echo strip, a second
// standing spot, tape calibration, wideband sweep and two-microphone echo direction.
import { View3D } from './view3d.js';
import { PlanView } from './plan2d.js';
import { EchoStrip } from './radargram.js';
import { registerSpots, mergeSpots, framesBySpot, MERGE } from '../core/merge.js';
import { fitCalibration, applyCalibration, CALIB_SAVE, CALIB_ENABLED } from '../core/calib.js';
import { stereoSetup, STEREO_DEFAULT, nearFar } from '../core/stereo.js';
import { LIVE_ENABLED, CLOUD_ENABLED } from '../core/live.js';
import { roomPolygon } from '../core/geom.js';
import { parseTruth, scoreField, fieldText } from '../core/fieldscore.js';
import { persistenceCandidates } from '../core/estimate.js';
import { SonarAudio } from './audio.js';
import { Motion } from './sensors.js';
import { BANDS, DEFAULTS, DEVICE_PROFILES, guessDeviceClass, VERSION } from '../core/config.js';
import { EchoProcessor } from '../core/dsp.js';
import { extractCandidates } from '../core/estimate.js';
import { LiveTwin } from '../core/twin.js';
import { EchoMLP } from '../core/model.js';
import { randomScene, Simulator } from '../core/sim.js';
import { evaluate } from '../core/metrics.js';
import { rotate } from '../core/geom.js';

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('et.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('et.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
};
const S = {
  settings: Object.assign({ band: 'auto', device: 'auto', tempC: 20, model: true, stereo: 'auto' }, store.get('settings', {})),
  spotRooms: [], spotRegs: null, calPairs: [], cal: null, viewMode: 'plan', spot: 0,
  frames: [], room: null, twin: null, mlp: null, modelJson: null, audio: null, motion: new Motion(),
  diag: {}, source: null, radar: false, sim: null, solving: false, lastSolve: 0, jobs: new Map(), jobId: 0
};
const view = new View3D($('gl'));
const plan = new PlanView($('plan'));
const strip = new EchoStrip($('strip'));
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
worker.onmessage = (e) => { const j = S.jobs.get(e.data.id); if (j) { S.jobs.delete(e.data.id); j(e.data.res); } };

// ---------- small UI helpers ----------
let toastTimer = null;
function toast(msg, ms = 1600) {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), ms);
  if (navigator.vibrate) { try { navigator.vibrate(12); } catch (e) { /* ignore */ } }
}
function show(id) { $(id).hidden = false; }
function hide(id) { $(id).hidden = true; }
function sheet(id) { for (const s of ['primer', 'more', 'details', 'settings', 'about', 'field', 'calib']) $(s).hidden = s !== id; }
document.querySelectorAll('.sheet .close').forEach(b => b.addEventListener('click', () => sheet(null)));
const fmt = (v, d = 2) => (v == null || !Number.isFinite(v)) ? '-' : v.toFixed(d);

function scorer() { return S.settings.model && S.mlp ? (x) => S.mlp.score(x) : null; }

async function loadLiveModels() {
  try {
    const [e, w] = await Promise.all(['echo-e.json', 'live-w.json'].map(n => fetch(new URL('../model/' + n, import.meta.url)).then(r => r.ok ? r.json() : null)));
    S.liveModels = e && w ? { echo: e, live: w } : null;
  } catch (err) { S.liveModels = null; }
}

// Live walls during the turn (v0.6): a separate worker runs the echo model and the live decoder per chirp.
function startLiveWalls(procIn) {
  const proc = procIn || (S.audio && S.audio.proc);
  if (!LIVE_ENABLED || !S.liveModels || !proc) return;
  if (!S.liveWorker) {
    S.liveWorker = new Worker(new URL('./liveworker.js', import.meta.url), { type: 'module' });
    S.liveWorker.onmessage = (e) => (e.data.type === 'cloud' ? onLiveCloud(e.data) : onLiveWalls(e.data));
  }
  const p = proc;
  S.liveProc = proc;
  S.liveWorker.postMessage({ type: 'start', proc: { fs: p.fs, f0: p.f0, f1: p.f1, chirpDur: p.chirpDur, c: p.c, rangeOffset: p.rangeOffset, maxRange: p.maxRange, binSize: p.binSize, kappa: p.kappa }, echo: S.liveModels.echo, live: S.liveModels.live, stereo: null });
  S.liveOn = true; S.liveStereo = null; S.liveMs = [];
  S.cloudMs = []; S.cloudUpdates = 0; S.cloudFinal = false; S.cloudWanted = false;
  plan.clearCloud();
}

// Echo cloud (v0.6.1): the live wall evidence as a glow on the plan, with an outline along its strongest ridge.
// Live updates while turning; one final cloud when the turn ends (asked for, so it covers every chirp).
function onLiveCloud(d) {
  if (!(S.liveOn || (d.final && S.cloudWanted))) return;
  if (d.empty) { if (d.final) { S.cloudWanted = false; S.cloudFinal = true; } return; }
  S.cloudMs.push(d.ms); S.cloudUpdates++;
  plan.setCloud(d, S.spot);
  if (d.final) { S.cloudWanted = false; S.cloudFinal = true; }
}
function requestFinalCloud() {
  if (!CLOUD_ENABLED || !S.liveWorker) return;
  S.cloudWanted = true; S.liveWorker.postMessage({ type: 'cloud' });
}
function noteCloudTiming() {
  if (S.cloudMs && S.cloudMs.length) { const s = S.cloudMs.slice().sort((a, b) => a - b); S.diag.cloudMsP95 = +s[Math.floor(0.95 * (s.length - 1))].toFixed(1); }
}

// Cloud for an opened recording: the same worker runs over the first spot's turn, quietly (no live walls drawn).
function cloudForReplay(frames, proc) {
  if (!CLOUD_ENABLED || !LIVE_ENABLED || !S.liveModels || !proc) return;
  if (!S.liveWorker) { S.liveWorker = new Worker(new URL('./liveworker.js', import.meta.url), { type: 'module' }); S.liveWorker.onmessage = (e) => (e.data.type === 'cloud' ? onLiveCloud(e.data) : onLiveWalls(e.data)); }
  const p = proc;
  S.liveOn = false; S.cloudMs = []; S.cloudUpdates = 0; S.cloudFinal = false; plan.clearCloud();
  S.liveWorker.postMessage({ type: 'start', proc: { fs: p.fs, f0: p.f0, f1: p.f1, chirpDur: p.chirpDur, c: p.c, rangeOffset: p.rangeOffset, maxRange: p.maxRange, binSize: p.binSize, kappa: p.kappa }, echo: S.liveModels.echo, live: S.liveModels.live, stereo: null });
  const F = frames.filter(f => (f.spot || 0) === 0);
  const st = stereoFor(F, proc);
  if (st) S.liveWorker.postMessage({ type: 'stereo', stereo: st });
  for (let i = 0; i < F.length; i++) {
    const f = F[i], a = F[i - 2];
    const omega = a ? (f.yaw - a.yaw) / Math.max(0.05, f.t - a.t) : 0;
    const nf = st && f.profile2 ? nearFar(f, st) : null;
    S.liveWorker.postMessage({ type: 'chirp', f: { yaw: f.yaw, omega, phase: f.phase, dir: f.dir, profile: nf ? nf.near : f.profile, far: nf ? nf.far : null, seg: f.seg } });
  }
  requestFinalCloud();
}

function sendLive(f) {
  if (!S.liveOn) return;
  if (!S.liveStereo && f.phase === 'turn' && f.profile2 && S.settings.stereo !== 'off' && (S.settings.stereo === 'on' || STEREO_DEFAULT)) {
    const st = stereoSetup(S.frames.filter(x => (x.spot || 0) === S.spot), S.liveProc);
    if (st.ok) { S.liveStereo = st; S.liveWorker.postMessage({ type: 'stereo', stereo: st }); }
  }
  const nf = S.liveStereo && f.profile2 ? nearFar(f, S.liveStereo) : null;
  S.liveWorker.postMessage({ type: 'chirp', f: { yaw: f.yaw, omega: f.omega || 0, phase: f.phase, dir: f.dir, profile: nf ? nf.near : f.profile, far: nf ? nf.far : null, seg: f.seg } });
}

function onLiveWalls(d) {
  if (!S.liveOn || S.finished) return;
  S.liveMs.push(d.ms);
  const walls = d.walls || [];
  if (walls.length) strip.setFit(walls, d.rho);
  const pg = walls.length >= 2 ? roomPolygon(walls) : null;
  const last = S.frames[S.frames.length - 1];
  plan.setLive({ ok: true, walls: walls.map(w => ({ ...w, sD: 0.03 })), poly: pg ? pg.poly : [], edgeWall: pg ? pg.edgeWall : [] }, last ? last.yaw : null);
  S.liveWallsShown = walls.length;
}

async function loadModel() {
  try {
    const r = await fetch(new URL('../model/echo-mlp.json', import.meta.url));
    S.modelJson = await r.json();
    S.mlp = new EchoMLP(S.modelJson);
  } catch (e) { S.mlp = null; }
}

function stereoFor(frames, proc) {
  if (S.settings.stereo === 'off' || (S.settings.stereo !== 'on' && !STEREO_DEFAULT) || !proc) return null;
  const st = stereoSetup(frames, proc);
  return st.ok ? st : null;
}

function solve(frames, opt, proc) {
  const id = ++S.jobId;
  const slim = frames.map(f => ({ t: f.t, yaw: f.yaw, phase: f.phase, dir: f.dir, dz: f.dz, profile: f.profile, profile2: f.profile2, n0: f.n0, n02: f.n02, spot: f.spot }));
  const st = stereoFor(slim, proc);
  return new Promise(res => {
    S.jobs.set(id, (r) => { if (r) r.stereoUsed = !!st; res(r); });
    worker.postMessage({ id, frames: slim, opt: { ...opt, stereo: st, useModel: S.settings.model }, model: S.modelJson });
  });
}

// Solves every standing spot of a recording, registers spot B to spot A and keeps the raw rooms, so a
// later calibration can be applied per spot and merged again. Returns the displayed room.
async function solveAllSpots(frames, opt, proc) {
  const parts = framesBySpot(frames);
  const rooms = [];
  for (const fr of parts) rooms.push(await solve(fr, opt, proc));
  S.spotRooms = rooms.filter(r => r.ok);
  S.spotRegs = null;
  if (rooms.length > 1 && rooms[0].ok && rooms[1].ok) S.spotRegs = registerSpots(rooms[0], rooms[1]);
  return composeRoom(rooms[0]);
}

// Displayed room: each spot's raw room with the calibration applied, merged when the registration held.
function composeRoom(fallback) {
  const R = S.spotRooms.map(r => S.cal ? applyCalibration(r, S.cal) : r);
  if (!R.length) return fallback;
  if (MERGE.enabled && R.length > 1 && S.spotRegs && S.spotRegs.ok) {
    const sc = S.cal ? S.cal.scale : 1;
    return mergeSpots(R[0], R[1], { ...S.spotRegs, t: [S.spotRegs.t[0] * sc, S.spotRegs.t[1] * sc] });
  }
  // without merging, show the room around the spot the user stands at now
  return R[Math.min(S.spot, R.length - 1)] || R[0];
}

function setStep(i, title, text) {
  document.querySelectorAll('.steps i').forEach((el, k) => el.classList.toggle('on', k <= i));
  $('stepTitle').textContent = title; $('stepText').textContent = text;
}
function setRing(p, label) { $('ring').style.setProperty('--p', Math.max(0, Math.min(1, p))); $('ringText').textContent = label; }

function roomSummary(room) {
  if (!room || !room.ok) return '';
  const pts = room.cells ? room.cells.flat() : room.poly;
  const xs = pts.map(p => p[0]);
  // width and depth along the dominant wall direction
  const w0 = room.walls.slice().sort((a, b) => (b.support || 0) - (a.support || 0))[0];
  const R = pts.map(p => rotate(p, -(w0 ? w0.phi : 0)));
  const W = Math.max(...R.map(p => p[0])) - Math.min(...R.map(p => p[0]));
  const L = Math.max(...R.map(p => p[1])) - Math.min(...R.map(p => p[1]));
  const H = room.floor && room.ceil ? room.floor.h + room.ceil.h : null;
  const meas = room.walls.filter(w => !w.inferred).length;
  return `${fmt(W, 2)} x ${fmt(L, 2)} m${H ? `, h ${fmt(H, 2)} m` : ''}  |  ${meas} measured wall${meas === 1 ? '' : 's'}${room.closed ? '' : ', open side'}` + (xs.length ? '' : '');
}

// Live echo points: persistence peaks of the profile minus a running background built from recent chirps
// that faced within 45 deg (the solver's own candidate rule, done one chirp at a time). Also returns the
// residual for the echo strip. Time O(bins x 40).
function liveEchoes(f) {
  const buf = S.liveBuf = S.liveBuf || [];
  buf.push({ yaw: f.yaw, p: f.profile });
  if (buf.length > 40) buf.shift();
  const near = buf.filter(b => Math.abs(Math.atan2(Math.sin(b.yaw - f.yaw), Math.cos(b.yaw - f.yaw))) <= Math.PI / 4);
  if (near.length < 5) return { echoes: [], residual: null };
  const n = f.profile.length, bg = new Float32Array(n), col = new Float32Array(near.length), res = new Float32Array(n);
  for (let i = 0; i < n; i++) { for (let q = 0; q < near.length; q++) col[q] = near[q].p[i]; col.sort(); bg[i] = col[near.length >> 1]; res[i] = Math.max(0, f.profile[i] - bg[i]); }
  return { echoes: persistenceCandidates([f], [bg], { maxPeaks: 6 }).map(c => ({ r: c.r, score: c.score / (c.score + 2) })), residual: res };
}

function setView(m) {
  S.viewMode = m;
  plan.show(m === 'plan');
  view.paused = m === 'plan';
  if (m !== 'plan') view.setMode(m);
  $('btnView').textContent = m === 'plan' ? 'Plan' : m === 'follow' ? '3D' : 'Orbit';
}

// Fewer than 2 measured walls is not a room (field error E126): no walls, no outline polygon, no 3D room;
// the plan shows the echo cloud alone and says what to change.
function measuredCount(room) { return room && room.ok ? room.walls.filter(w => !w.inferred).length : 0; }
const NO_WALLS = 'No walls measured from here. The glow shows where echoes came from. Hold the phone at arm\'s length and turn slowly, about a metre from walls.';

function showRoom(room) {
  S.room = room;
  S.cloudOnly = !room.merged && measuredCount(room) < 2;
  view.setRoom(S.cloudOnly ? null : room.merged ? { ...room, poly: room.cells[0] } : room);
  plan.setRoom(room, { cloudOnly: S.cloudOnly });
  plan.setPhone(null);
  show('hud'); show('bar');
  $('dims').textContent = S.cloudOnly ? NO_WALLS : roomSummary(room) + (room.merged ? '  |  2 spots' : '') + (room.calibration ? '  |  tape calibrated' : '');
  $('btnSpot').hidden = !(S.source === 'live' && S.audio && S.spotRooms.length === 1);
  $('btnCalib').hidden = !CALIB_ENABLED;
  setView(S.userView || 'plan');
}

// ---------- real scan ----------
async function startScan() {
  S.liveBuf = [];
  const dismiss = store.get('primerDismiss', []).filter(t => Date.now() - t < 7 * 864e5);
  $('primerNote').textContent = dismiss.length >= 3 ? 'You declined this a few times. Echo Twin cannot map without the microphone.' : 'Audio stays on this phone. You can stop at any time.';
  sheet('primer');
}

function bandFor(name) { return name === 'audible' ? BANDS.audible : name === 'wide' ? BANDS.wide : BANDS.high; }

async function allowAndGo() {
  sheet(null);
  try {
    const perm = await Motion.requestPermission();
    S.diag.motionPermission = perm;
    S.motion.start();
    S.audio = new SonarAudio();
    const info = await S.audio.open();
    S.diag.audio = info;
  } catch (e) {
    toast('Microphone not available: ' + (e.message || e.name), 4000);
    return;
  }
  hide('home'); hide('bar'); hide('hud'); show('scan'); setView('follow');
  setStep(0, 'Sound check', 'Listening to one chirp.');
  setRing(0, '');
  const dev = S.settings.device === 'auto' ? guessDeviceClass(navigator.userAgent) : S.settings.device;
  S.diag.deviceClass = dev; S.deviceClass = dev;
  let band = bandFor(S.settings.band);
  const cfg = () => ({ band, period: DEFAULTS.period, rangeOffset: DEVICE_PROFILES[dev].rangeOffset, tempC: S.settings.tempC });
  S.audio.configure(cfg());
  let chk = await S.audio.soundCheck();
  if (chk.snrDb < 18 && S.settings.band === 'auto') {
    band = BANDS.audible; S.audio.configure(cfg());
    chk = await S.audio.soundCheck();
    toast('High band too quiet on this phone; using audible chirps', 2600);
  }
  S.diag.soundCheck = { ...chk, band: band.name };
  S.band = band;
  if (chk.snrDb < 12) {
    setStep(0, 'Echo too weak', `Signal ${fmt(chk.snrDb, 0)} dB. Turn the volume up, remove a thick case, and try again.`);
    $('btnStep').hidden = false; $('btnStep').textContent = 'Try again';
    $('btnStep').onclick = () => { hide('scan'); S.audio.close(); allowAndGo(); };
    return;
  }
  toast(`Sound check ok, ${fmt(chk.snrDb, 0)} dB`);
  S.frames = []; S.source = 'live'; S.spot = 0; S.spotRooms = []; S.spotRegs = null; S.calPairs = [];
  S.cal = CALIB_SAVE ? store.get('calib.' + dev, null) : null;
  S.tStart = performance.now();
  S.audio.onFrame = (q) => onScanFrame(q);
  startLiveWalls();
  setStep(1, 'Raise the phone', 'Hold it flat, screen up, top edge toward you, at waist height. Tap Start, then raise it slowly to your chest.');
  setRing(0, '');
  $('btnStep').hidden = false; $('btnStep').textContent = 'Start';
  $('btnStep').onclick = () => runLift();
}

function processFrame(q) {
  const r = S.audio.proc.process(q.seg, q.dir);
  const f = { t: (q.perf - S.tStart) / 1000, perf: q.perf, yaw: S.motion.yawAt(q.perf + 15), phase: q.phase, dir: q.dir, profile: r.profile, n0: r.n0, snr: r.snrDb, seg: q.seg, spot: S.spot };
  // a silent or missing second channel is not a second microphone
  if (q.seg2) {
    let e = 0; for (let i = 0; i < q.seg2.length; i += 8) e += q.seg2[i] * q.seg2[i];
    if (e > 1e-10) { const r2 = S.audio.proc.process(q.seg2, q.dir); f.profile2 = r2.profile; f.n02 = r2.n0; f.seg2 = q.seg2; }
  }
  return f;
}

function onScanFrame(q) {
  const f = processFrame(q);
  if (q.phase === 'live') return onLiveFrame(f);
  S.frames.push(f);
  if (S.frames.length > 2) { const a = S.frames[S.frames.length - 3]; f.omega = (f.yaw - a.yaw) / Math.max(0.05, f.t - a.t); }
  sendLive(f);
  view.setHeading(f.yaw);
  const le = liveEchoes(f);
  view.addEchoes(le.echoes);
  if (S.liveOn && q.phase === 'turn') plan.addEchoes(le.echoes, f.yaw, 0, 0.3);
  if (q.phase === 'turn') {
    if (le.residual) strip.add(f.yaw, le.residual, S.audio.proc.binSize);
    turnProgress(f);
  }
}

function beginTurn(title, text) {
  S.audio.phase = 'turn';
  S.turnYaw0 = S.motion.yawAt(performance.now());
  S.liveBuf = [];
  strip.reset(S.turnYaw0); show('strip');
  setStep(2, title, text + (S.liveOn ? ' Walls appear on the plan as they are heard.' : ''));
  setRing(0, '0%');
  if (S.liveOn) { plan.setLive({ ok: true, walls: [], poly: [], edgeWall: [] }, S.turnYaw0); setView('plan'); }
}

function runLift() {
  $('btnStep').hidden = true;
  S.liftT0 = performance.now();
  S.audio.startTrain({ phase: 'lift' });
  let k = 0;
  const iv = setInterval(() => { k++; setRing(k / 28, ''); }, 100);
  setTimeout(() => {
    clearInterval(iv);
    S.liftT1 = performance.now();
    const fz = S.motion.liftProfile(S.liftT0, S.liftT1);
    if (fz) for (const f of S.frames) if (f.phase === 'lift') f.dz = fz(f.perf);
    const span = fz ? Math.abs(fz(S.liftT1) - fz(S.liftT0)) : null;
    S.diag.liftSpan = span;
    if (span != null && span < 0.12) toast('Raised only a little; floor and ceiling may be less certain', 2600);
    beginTurn('Turn around', 'Keep it flat and turn slowly on the spot, a full circle. Stay about a metre from walls. Facing a wall squarely at the start helps if you calibrate with a tape later.');
  }, 2800);
}

function turnFrames() { return S.frames.filter(f => f.spot === S.spot); }

function turnProgress(f) {
  const need = 2 * Math.PI + 0.35;
  const done = Math.abs(f.yaw - S.turnYaw0) / need;
  setRing(done, Math.min(100, Math.round(done * 100)) + '%');
  const F = turnFrames(), n = F.length;
  if (n > 3) {
    const a = F[n - 3];
    const w = Math.abs(f.yaw - a.yaw) / Math.max(0.05, f.t - a.t);
    $('scanHint').textContent = w > 0.9 ? 'Slower, please' : '';
  }
  const now = performance.now();
  if (done > 0.35 && !S.solving && now - S.lastSolve > 3000) {
    S.solving = true; S.lastSolve = now;
    solve(F, { kappa: S.audio.proc.kappa, bwKHz: (S.band.f1 - S.band.f0) / 1000, fast: true }, S.audio.proc).then(res => {
      S.solving = false;
      if (res.ok && !S.finished) { if (S.spot === 0) view.setRoom(res); strip.setFit(res.walls, res.rho); }
    });
  }
  const t0 = F.find(x => x.phase === 'turn');
  if (done >= 1 && !S.finished) finishScan();
  if (t0 && f.t - t0.t > 75 && !S.finished) finishScan();
}

async function finishScan() {
  S.finished = true;
  if (S.liveOn) { requestFinalCloud(); S.liveOn = false; plan.endLive(); noteCloudTiming(); if (S.liveMs.length) { const s = S.liveMs.slice().sort((a, b) => a - b); S.diag.liveMsP95 = s[Math.floor(0.95 * s.length)]; } }
  S.audio.stopTrain();
  setStep(2, 'Building the room', S.spot ? 'Solving the second spot, then fitting it to the first.' : 'Solving the walls, floor and ceiling.');
  const opt = { kappa: S.audio.proc.kappa, bwKHz: (S.band.f1 - S.band.f0) / 1000, persist: true };
  const res = await solve(turnFrames(), opt, S.audio.proc);
  hide('scan'); hide('strip');
  if (S.spot === 0) {
    S.diag.solve = res.ok ? { rho: res.rho, drift: res.drift, sigma: res.sigma, ms: res.solveMs, assoc: res.nAssoc, cands: res.nCands, stereo: res.stereoUsed } : { reason: res.reason };
    if (!res.ok) { show('home'); toast(res.reason, 4000); return; }
    S.spotRooms = [res];
    showRoom(composeRoom(res));
    if (S.cloudOnly) { toast('No walls measured. Showing where echoes came from.', 3600); return; }
    toast(res.confident ? 'Room ready' : 'Room ready. Walls it could not hear are dashed; add a second spot to fill them in.', 3200);
    startRadar();
    return;
  }
  // second spot
  S.diag.solveB = res.ok ? { rho: res.rho, ms: res.solveMs, cands: res.nCands, stereo: res.stereoUsed } : { reason: res.reason };
  if (!res.ok) { showRoom(composeRoom()); toast('Second spot: ' + res.reason, 4000); return; }
  S.spotRooms = [S.spotRooms[0], res];
  S.spotRegs = registerSpots(S.spotRooms[0], res);
  S.diag.registration = S.spotRegs.ok ? { t: S.spotRegs.t, dhDeg: S.spotRegs.dh * 180 / Math.PI, score: S.spotRegs.score, second: S.spotRegs.second } : { failed: S.spotRegs.reason };
  showRoom(composeRoom());
  toast(MERGE.enabled ? (S.spotRegs.ok ? 'Second spot added' : `Could not fit the second spot to the first: ${S.spotRegs.reason}. Showing the first spot.`) : 'Second spot recorded for the field test. Merging spots is not reliable yet, so the plan shows the room from where you stand now. Save the recording from More.', 5200);
}

// Second standing spot: the gyro keeps tracking heading while the user walks, then one more turn (no lift).
function startSecondSpot() {
  if (!(S.source === 'live' && S.audio && S.spotRooms.length === 1)) { toast('Scan a room first'); return; }
  stopRadar();
  hide('bar'); hide('hud'); show('scan'); setView('follow');
  setStep(1, 'Walk to a new spot', 'Walk to the part of the room the first scan could not hear, at least a metre from walls. Keep the phone flat and pointing ahead while you walk. Tap Start when you are there.');
  setRing(0, '');
  $('btnStep').hidden = false; $('btnStep').textContent = 'Start';
  $('btnStep').onclick = () => {
    $('btnStep').hidden = true;
    S.spot = 1; S.finished = false;
    startLiveWalls();
    S.audio.period = DEFAULTS.period;
    S.audio.onFrame = (q) => onScanFrame(q);
    S.audio.startTrain({ phase: 'turn' });
    beginTurn('Turn around', 'Keep it flat and turn slowly on this spot, a full circle.');
  };
}

// ---------- live twin and radar ----------
function radarRoom() { const r = S.spotRooms[S.spot] || S.room; return S.cal && r ? applyCalibration(r, S.cal) : r; }

function startRadar() {
  if (!S.room || !S.room.ok || S.cloudOnly) return;
  const turn = S.frames.filter(f => f.phase === 'turn' && (f.spot || 0) === S.spot);
  if (S.source === 'live' && S.audio) {
    const kappa = S.audio.proc.kappa;
    S.audio.period = 0.06;
    S.twin = new LiveTwin(radarRoom(), turn, { kappa });
    S.audio.onFrame = (q) => onScanFrame(q);
    S.audio.startTrain({ alt: true, phase: 'live' });
  }
  S.radar = true; $('btnRadar').setAttribute('aria-pressed', 'true'); show('meter');
  clearTimeout(S.radarTimer);
  S.radarTimer = setTimeout(() => { if (S.radar && S.source === 'live') { stopRadar(); toast('Radar paused to save battery'); } }, 120000);
}

function stopRadar() {
  S.radar = false; $('btnRadar').setAttribute('aria-pressed', 'false'); hide('meter');
  if (S.audio) S.audio.stopTrain();
  if (S.simLive) { clearInterval(S.simLive); S.simLive = null; }
  view.setTracks([]);
}

function onLiveFrame(f) {
  if (!S.twin) return;
  const rr = radarRoom();
  f.cands = extractCandidates([f], rr.inv, { scorer: scorer() });
  const out = S.twin.update(f);
  view.setHeading(f.yaw);
  view.addEchoes(f.cands);
  if (S.viewMode === 'plan') { plan.setPhone(f.yaw, S.room.merged ? S.spot : 0, rr.rho || 0.3); plan.addEchoes(f.cands, f.yaw, S.room.merged ? S.spot : 0, rr.rho || 0.3); }
  view.setTracks(out.tracks);
  view.setTraces(out.walls);
  const frac = Math.min(1, out.divergence / Math.max(1e-6, out.thresholds[1] * 1.5));
  $('meterFill').style.width = (100 * frac) + '%';
  $('meter').classList.toggle('alert', out.event);
  if (out.transition === 'start') {
    const tr = out.tracks[0];
    toast(tr ? `Movement at ${fmt(tr.r, 1)} m, ${tr.rdot < 0 ? 'coming closer' : 'moving away'} ${fmt(Math.abs(tr.rdot), 1)} m/s` : 'Something changed in the room');
  }
  if (S.source !== 'sim') $('status').textContent = `filter NIS ${fmt(out.nisMean, 2)}, echo noise ${fmt(out.R * 100, 1)} cm, ${fmt(out.ms, 1)} ms per chirp`;
  S.diag.live = { nisMean: out.nisMean, R: out.R, thresholds: out.thresholds, ms: out.ms };
}

// ---------- simulated room demo ----------
async function simDemo(seed) {
  stopRadar();
  hide('home'); sheet(null);
  seed = seed ?? Math.floor(Math.random() * 100000) + 20000;
  const sc = randomScene(seed);
  const sim = new Simulator(sc);
  S.sim = { sc, sim, seed }; S.source = 'sim'; S.finished = false;
  const truth = sim.truth();
  view.setRoom(null);
  view.setTruth(truth.poly.map(p => rotate(p, sc.imuOffset + sc.drift * 0.5)), truth.H);
  show('scan'); setView('follow'); setStep(2, 'Simulated room', `Room ${seed}: ${sc.kind}, ${sc.device.cls} speaker, ${sc.band.name} band. Turning...`);
  $('btnStep').hidden = true;
  S.frames = []; S.spot = 0; S.spotRooms = []; S.spotRegs = null; S.calPairs = []; S.cal = null; S.liveBuf = [];
  strip.reset(null); show('strip');
  S.turnYaw0 = null; S.liveOn = false;
  startLiveWalls(sim.proc);
  const total = sc.liftDur + sc.turnDur + 0.3;
  let k = 0;
  await new Promise(resolve => {
    const step = () => {
      for (let j = 0; j < 4 && k * sc.period < total; j++, k++) {
        const t = k * sc.period;
        const fr = sim.frame(t, 1);
        const yaw = sim.imuYaw(t, fr.pose.psi);
        const f = { t, yaw, phase: fr.pose.phase, dir: 1, profile: sim.proc.process(fr.seg, 1).profile, seg: fr.seg };
        if (f.phase === 'lift') f.dz = (fr.pose.pos[2] - sc.z0) * sc.liftScaleErr + sc.liftNoise * Math.sin(2.1 * t + sc.liftNoisePh);
        S.frames.push(f);
        if (S.frames.length > 2) { const a = S.frames[S.frames.length - 3]; f.omega = (f.yaw - a.yaw) / Math.max(0.05, f.t - a.t); }
        if (S.liveOn) { if (f.phase === 'turn' && S.turnYaw0 == null) { S.turnYaw0 = yaw; plan.setLive({ ok: true, walls: [], poly: [], edgeWall: [] }, yaw); setView('plan'); } sendLive(f); }
        view.setHeading(yaw);
        const le = liveEchoes(f);
        view.addEchoes(le.echoes);
        if (f.phase === 'turn' && le.residual) strip.add(yaw, le.residual, sim.proc.binSize);
        if (S.liveOn && f.phase === 'turn') plan.addEchoes(le.echoes, yaw, 0, 0.3);
      }
      setRing(Math.min(1, k * sc.period / total), Math.round(100 * Math.min(1, k * sc.period / total)) + '%');
      if (k * sc.period < total) setTimeout(step, 40); else resolve();
    };
    step();
  });
  setStep(2, 'Building the room', 'Solving the walls, floor and ceiling.');
  if (S.liveOn) { await new Promise(r => setTimeout(r, 400)); requestFinalCloud(); S.liveOn = false; plan.endLive(); noteCloudTiming(); if (S.liveMs.length) { const s = S.liveMs.slice().sort((a, b) => a - b); S.diag.liveMsP95 = s[Math.floor(0.95 * s.length)]; } }
  const res = await solve(S.frames, { kappa: sim.proc.kappa, bwKHz: (sc.band.f1 - sc.band.f0) / 1000, persist: true }, sim.proc);
  hide('scan'); hide('strip');
  if (!res.ok) { toast(res.reason, 4000); show('home'); return; }
  const ev = evaluate({ ...res, P: null, smoother: null }, truth);
  const tpoly = truth.poly.map(p => rotate(p, ev.rot));
  view.setTruth(tpoly, truth.H);
  plan.setTruth(tpoly);
  S.spotRooms = [res];
  showRoom(res);
  const errs = (ev.wallErr || []).slice().sort((a, b) => a - b);
  S.diag.sim = { seed, kind: sc.kind, device: sc.device.cls, band: sc.band.name, iou: ev.iou, recall: ev.recall, precision: ev.precision, wallErrMedianCm: errs.length ? errs[errs.length >> 1] * 100 : null, floorErrCm: ev.floorErr != null ? ev.floorErr * 100 : null, ceilErrCm: ev.ceilErr != null ? ev.ceilErr * 100 : null };
  $('status').textContent = `${res.confident ? 'stable' : 'not verified'} | vs truth: IoU ${fmt(ev.iou, 2)}, walls ${Math.round(ev.recall * ev.nObservable)}/${ev.nObservable}, median error ${fmt(S.diag.sim.wallErrMedianCm, 1)} cm`;
  toast('Simulated room ready. Green dashes are the truth.', 2400);
  document.body.dataset.ready = '1';
  simRadar();
}

function simRadar() {
  if (S.cloudOnly) return;
  const { sc, sim } = S.sim;
  const turn = S.frames.filter(f => f.phase === 'turn');
  S.twin = new LiveTwin(S.room, turn, { kappa: sim.proc.kappa });
  const u = [Math.cos(sc.psi0), Math.sin(sc.psi0)];
  const t0 = 300;
  const walker = { p0: [sc.c0[0] + u[0] * 3.0, sc.c0[1] + u[1] * 3.0, 1.1], v: [-u[0] * 0.8, -u[1] * 0.8, 0], t0: t0 + 2, t1: t0 + 5, sigma: 0.6 };
  let t = t0, k = 0;
  S.radar = true; $('btnRadar').setAttribute('aria-pressed', 'true'); show('meter');
  S.simLive = setInterval(() => {
    const dir = k % 2 ? -1 : 1;
    const psi = sc.psi0;
    const pose = { t, psi, u: [u[0], u[1], 0], pos: [sc.c0[0] + sc.rho * u[0], sc.c0[1] + sc.rho * u[1], sc.z0], phase: 'live', center: sc.c0 };
    const fr = sim.frame(t, dir, { pose, movers: [walker] });
    const yaw = psi + sc.imuOffset + sc.drift;
    onLiveFrame({ t, yaw, dir, profile: sim.proc.process(fr.seg, dir).profile });
    t += 0.06; k++;
    if (t > t0 + 9) { t = t0; k = 0; }
  }, 60);
}

// ---------- recordings ----------
function b64(i16) { let s = ''; const u8 = new Uint8Array(i16.buffer); for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
function unb64(str) { const bin = atob(str); const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Int16Array(u8.buffer); }

function exportRecording() {
  if (!S.frames.length) { toast('Nothing recorded yet'); return; }
  const p = S.source === 'sim' ? S.sim.sim.proc : S.audio && S.audio.proc;
  const pack = (seg) => { let m = 0; for (const v of seg) m = Math.max(m, Math.abs(v)); const sc = m > 0 ? 32767 / m : 1; return [sc, b64(Int16Array.from(seg, v => Math.round(v * sc)))]; };
  const frames = S.frames.map(f => {
    const [sc, seg] = pack(f.seg);
    const o = { t: f.t, yaw: f.yaw, phase: f.phase, dir: f.dir, dz: f.dz, scale: sc, seg };
    if (f.spot) o.spot = f.spot;
    if (f.seg2) { const [sc2, seg2] = pack(f.seg2); o.scale2 = sc2; o.seg2 = seg2; }
    return o;
  });
  const meta = { app: 'echo-twin', version: VERSION, source: S.source, fs: p.fs, f0: p.f0, f1: p.f1, chirpDur: p.chirpDur, tempC: S.settings.tempC, rangeOffset: p.rangeOffset, ua: navigator.userAgent, date: new Date().toISOString(), diag: S.diag, simSeed: S.sim ? S.sim.seed : null };
  if (S.fieldTruth) meta.truth = S.fieldTruth;
  if (S.calPairs.length) meta.tape = S.calPairs;
  const blob = new Blob([JSON.stringify({ meta, frames })], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `echo-twin-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  toast('Recording saved');
}

async function openRecording(file) {
  try {
    const J = JSON.parse(await file.text());
    const m = J.meta;
    const proc = new EchoProcessor({ fs: m.fs, f0: m.f0, f1: m.f1, chirpDur: m.chirpDur, tempC: m.tempC, rangeOffset: m.rangeOffset });
    stopRadar(); hide('home'); sheet(null);
    S.frames = J.frames.map(f => {
      const i16 = unb64(f.seg);
      const seg = Float32Array.from(i16, v => v / f.scale);
      const p0 = proc.process(seg, f.dir);
      const fr = { t: f.t, yaw: f.yaw, phase: f.phase, dir: f.dir, dz: f.dz, spot: f.spot || 0, seg, profile: p0.profile, n0: p0.n0 };
      if (f.seg2) { const seg2 = Float32Array.from(unb64(f.seg2), v => v / f.scale2); const p1 = proc.process(seg2, f.dir); fr.seg2 = seg2; fr.profile2 = p1.profile; fr.n02 = p1.n0; }
      return fr;
    });
    S.source = 'replay'; S.spot = 0; S.calPairs = []; S.cal = null; S.replayProc = proc;
    S.fieldTruth = m.truth || null;
    view.setTruth(null); plan.setTruth(null);
    const res = await solveAllSpots(S.frames, { kappa: proc.kappa, bwKHz: (m.f1 - m.f0) / 1000, persist: true }, proc);
    if (!res || !res.ok) { toast(res ? res.reason : 'No room', 4000); show('home'); return; }
    S.diag.registration = S.spotRegs ? (S.spotRegs.ok ? { t: S.spotRegs.t, dhDeg: S.spotRegs.dh * 180 / Math.PI, score: S.spotRegs.score, second: S.spotRegs.second } : { failed: S.spotRegs.reason }) : null;
    if (m.tape && m.tape.length) { S.calPairs = m.tape; S.cal = fitCalibration(S.calPairs); }
    const shown = composeRoom(res);
    cloudForReplay(S.frames, proc);
    showRoom(shown);
    const fs0 = m.truth ? scoreField(shown, m.truth) : null;
    const nSpots = framesBySpot(S.frames).length;
    $('status').textContent = `replay of ${m.date ? m.date.slice(0, 16) : 'recording'}` + (nSpots > 1 ? (!MERGE.enabled ? ', second spot not merged (not reliable yet)' : S.spotRegs && S.spotRegs.ok ? ', 2 spots fitted' : ', second spot not fitted') : '') + (fs0 ? ` | tape: ${Math.round(fs0.recall * fs0.measured)}/${fs0.measured} walls within 15 cm, median error ${fs0.errMedCm == null ? '-' : fs0.errMedCm.toFixed(1)} cm` : '');
    toast(nSpots > 1 && MERGE.enabled && !(S.spotRegs && S.spotRegs.ok) ? 'Recording opened. The second spot could not be fitted: ' + (S.spotRegs ? S.spotRegs.reason : 'it did not solve') : 'Recording opened', 3600);
    document.body.dataset.ready = '1';
  } catch (e) { toast('Could not open that file: ' + e.message, 4000); }
}

// ---------- tape calibration ----------
function calibText() {
  const lines = [];
  if (S.calPairs.length) lines.push(...S.calPairs.map((p, i) => `wall ${i + 1}: tape ${p.tape.toFixed(2)} m, sound ${p.est.toFixed(2)} m`));
  if (S.cal) lines.push(`correction: ${S.cal.kind === 'offset' ? '' : `scale ${S.cal.scale.toFixed(4)}, `}offset ${(100 * S.cal.offset >= 0 ? '+' : '') + (100 * S.cal.offset).toFixed(1)} cm, applied to every wall, the floor and the ceiling`);
  lines.push(CALIB_SAVE ? 'Saved for later scans on this phone.' : 'Applies to this scan only (in simulation, a calibration did not carry over to other rooms).');
  return lines.join('\n');
}
function openCalib() { $('calibOut').textContent = calibText(); $('calibWallRow').hidden = S.calWall == null; sheet('calib'); }
function pickWallForCalib() { if (!S.room || !S.room.ok) { toast('Scan a room first'); return; } sheet(null); setView('plan'); S.pickWall = true; toast('Tap the wall you measured', 2400); }
$('plan').addEventListener('click', (e) => {
  if (!S.room) return;
  const wi = plan.hitWall(e.clientX, e.clientY);
  plan.setSelected(wi);
  if (wi < 0) return;
  const lab = PlanView.wallLabel(S.room, wi);
  if (S.pickWall) {
    S.pickWall = false; S.calWall = { wi, shown: lab.d };
    $('calibWallLabel').textContent = `Selected wall: ${lab.text} by sound. Tape from the spot you turned on (between your feet) straight to this wall:`;
    $('calibTape').value = '';
    openCalib();
  } else toast(`Wall at ${lab.text}${S.room.walls[wi].sD != null ? `, +/- ${Math.max(1, Math.round(100 * S.room.walls[wi].sD))} cm` : ''}`);
});
function saveCalib() {
  const tape = +$('calibTape').value;
  if (!S.calWall || !(tape > 0.2 && tape < 15)) { $('calibOut').textContent = 'Enter the taped distance in metres, for example 1.42'; return; }
  const s0 = S.cal ? S.cal.scale : 1, o0 = S.cal ? S.cal.offset : 0;
  S.calPairs.push({ est: (S.calWall.shown - o0) / s0, tape });
  S.cal = fitCalibration(S.calPairs);
  S.calWall = null;
  if (CALIB_SAVE && S.deviceClass) store.set('calib.' + S.deviceClass, S.cal);
  showRoom(composeRoom());
  openCalib();
  toast(`Calibrated: offset ${(100 * S.cal.offset >= 0 ? '+' : '') + (100 * S.cal.offset).toFixed(1)} cm`);
}
function clearCalib() { S.calPairs = []; S.cal = null; S.calWall = null; if (S.deviceClass) store.set('calib.' + S.deviceClass, null); if (S.room) showRoom(composeRoom()); openCalib(); toast('Calibration cleared'); }

// ---------- details, settings ----------
function confLine(r) {
  const odds = r.quality && r.quality.odds;
  if (r.oddsMin != null) return r.confident ? `verified (odds margin ${fmt(odds, 0)} nats, needs ${r.oddsMin})` : `not verified: ${r.plausible ? `a different room explains the echoes almost as well (margin ${fmt(odds, 0)} nats, needs ${r.oddsMin})` : 'too few measured walls or the plan is open'}`;
  if (!r.persist) return odds != null ? `not calibrated yet (odds margin ${fmt(odds, 0)} nats)` : 'not checked';
  const m = fmt(r.persist.minIoU, 2);
  if (r.tau == null) return `not calibrated yet (plan overlap under 4 perturbations: worst ${m}); treat the room as unverified`;
  if (r.confident) return `stable (worst overlap ${m} under 4 perturbations, needs ${fmt(r.tau, 2)})`;
  return `not verified: ${r.plausible ? `the plan changes under small perturbations (worst overlap ${m}, needs ${fmt(r.tau, 2)})` : 'too few measured walls or the plan is open'}`;
}

function details() {
  const r = S.room;
  const lines = [
    `Echo Twin ${VERSION}`,
    `source: ${S.source || '-'}`,
    `frames: ${S.frames.length}`,
    r && r.ok ? `walls: ${r.walls.map(w => `${fmt(w.d, 2)} m +/- ${fmt(w.sD * 100, 1)} cm${w.inferred ? ' (inferred)' : ''}`).join('; ')}` : '',
    r && r.ok ? `floor: ${r.floor ? fmt(r.floor.h, 2) + ' m' : '-'}  ceiling: ${r.ceil ? fmt(r.ceil.h, 2) + ' m' : '-'}  hold radius: ${fmt(r.rho, 2)} m  heading drift: ${fmt(r.drift * 180 / Math.PI * 20, 1)} deg/20 s` : '',
    r && r.ok ? `confidence: ${confLine(r)}` : '',
    S.spotRooms.length > 1 && !MERGE.enabled ? `second spot: recorded, not merged (wall-only fitting failed its sealed test; ${S.spotRegs && S.spotRegs.ok ? 'it would have fitted this one' : 'it would not have fitted this one'})` : '',
    r && r.merged ? `second spot: fitted, walked ${fmt(Math.hypot(...r.spots[1]), 2)} m, heading offset ${fmt(r.registration.dh * 180 / Math.PI, 1)} deg, evidence ${fmt(r.registration.score, 1)} vs ${fmt(r.registration.second, 1)} nats` : (S.spotRegs && !S.spotRegs.ok ? `second spot: not fitted (${S.spotRegs.reason})` : ''),
    r && r.calibration ? `tape calibration: offset ${fmt(100 * r.calibration.offset, 1)} cm, scale ${fmt(r.calibration.scale, 4)} from ${r.calibration.n} wall${r.calibration.n === 1 ? '' : 's'}` : '',
    `microphones: ${micLine()}`,
    `sweep: ${S.band ? S.band.label : (S.source === 'sim' ? S.sim.sc.band.name : '-')}`,
    `model: ${S.mlp ? `on (${S.modelJson.hidden * S.modelJson.inputs + 2 * S.modelJson.hidden + 1} weights, val AP ${fmt(S.modelJson.validation.ap, 3)} vs ${fmt(S.modelJson.validation.amplitudeBaselineAp, 3)})` : 'not loaded'}${S.settings.model ? '' : ', disabled'}`,
    `diagnostics: ${JSON.stringify(S.diag, (k, v) => typeof v === 'number' ? Math.round(v * 1000) / 1000 : v, 1)}`
  ];
  $('diag').textContent = lines.filter(Boolean).join('\n');
  sheet('details');
}

function micLine() {
  const F = S.frames.filter(f => f.profile2);
  if (!F.length) return `one channel${S.diag.audio && S.diag.audio.channels ? ` (browser reports ${S.diag.audio.channels})` : ''}`;
  const p = S.source === 'live' && S.audio ? S.audio.proc : S.replayProc || null;
  const st = p ? stereoSetup(S.frames, p) : null;
  if (!st) return 'two channels recorded';
  if (!st.ok) return `two channels, not used: ${st.reason}`;
  const use = S.settings.stereo === 'on' || (S.settings.stereo === 'auto' && STEREO_DEFAULT);
  return `two distinct microphones ${fmt(100 * st.gap, 1)} cm apart${use ? ', used for echo direction' : ' (echo direction off: ' + (S.settings.stereo === 'off' ? 'turned off in settings' : 'not adopted by the sealed test yet') + ')'}`;
}

function openSettings() {
  $('setBand').value = S.settings.band; $('setDevice').value = S.settings.device;
  $('setTemp').value = S.settings.tempC; $('setModel').checked = !!S.settings.model;
  $('setStereo').value = S.settings.stereo || 'auto';
  sheet('settings');
}

function saveSettings() {
  S.settings = { band: $('setBand').value, device: $('setDevice').value, tempC: Math.max(-20, Math.min(45, +$('setTemp').value || 20)), model: $('setModel').checked, stereo: $('setStereo').value };
  store.set('settings', S.settings);
  sheet(null); toast('Settings saved');
}

// ---------- wiring ----------
$('btnScan').onclick = startScan;
$('btnAllow').onclick = allowAndGo;
$('btnNotNow').onclick = () => { const d = store.get('primerDismiss', []); d.push(Date.now()); store.set('primerDismiss', d.slice(-5)); sheet(null); };
$('btnSim').onclick = () => simDemo();
$('btnNewSim').onclick = () => simDemo();
$('btnOpen').onclick = $('btnOpen2').onclick = () => $('fileIn').click();
$('fileIn').onchange = (e) => { if (e.target.files[0]) openRecording(e.target.files[0]); e.target.value = ''; };
$('btnCancel').onclick = async () => {
  S.finished = true; hide('strip');
  if (S.spot > 0 && S.spotRooms.length) { S.audio.stopTrain(); S.frames = S.frames.filter(f => f.spot === 0); S.spot = 0; hide('scan'); showRoom(composeRoom()); toast('Second spot cancelled'); return; }
  if (S.audio) await S.audio.close(); S.audio = null; hide('scan'); show('home'); toast('Scan cancelled');
};
$('btnRescan').onclick = async () => { stopRadar(); S.finished = false; if (S.audio) { await S.audio.close(); S.audio = null; } hide('bar'); hide('hud'); view.setTruth(null); view.setRoom(null); plan.setTruth(null); plan.setRoom(null); plan.clearCloud(); S.spotRooms = []; S.spotRegs = null; startScan(); };
$('btnSpot').onclick = startSecondSpot;
$('btnCalib').onclick = () => { S.calWall = null; openCalib(); };
$('btnCalibPick').onclick = pickWallForCalib;
$('btnCalibSave').onclick = saveCalib;
$('btnCalibClear').onclick = clearCalib;
$('btnRadar').onclick = () => { if (S.radar) { stopRadar(); toast('Radar off'); } else if (S.cloudOnly) toast('No measured walls to watch yet'); else if (S.source === 'sim') { simRadar(); toast('Radar on'); } else if (S.audio) { startRadar(); toast('Radar on'); } else toast('Scan a room first'); };
$('btnMore').onclick = () => sheet('more');
$('btnExport').onclick = exportRecording;
const VIEWS = ['plan', 'follow', 'orbit'];
$('btnView').onclick = () => { S.userView = VIEWS[(VIEWS.indexOf(S.viewMode || 'plan') + 1) % VIEWS.length]; setView(S.userView); };
$('btnField').onclick = () => {
  const t = S.fieldTruth || store.get('fieldTruth', null);
  if (t) { $('fieldWalls').value = t.walls.join(', '); $('fieldRoomH').value = t.roomHeight || ''; $('fieldPhoneH').value = t.phoneHeight || ''; }
  $('fieldOut').textContent = t && S.room ? fieldText(scoreField(S.room, t)) : '';
  sheet('field');
};
$('btnFieldScore').onclick = () => {
  const t = parseTruth($('fieldWalls').value, $('fieldRoomH').value, $('fieldPhoneH').value);
  if (!t) { $('fieldOut').textContent = 'Enter at least two wall distances in metres, for example 1.42, 2.10, 2.65'; return; }
  S.fieldTruth = t; store.set('fieldTruth', t);
  $('fieldOut').textContent = S.room && S.room.ok ? fieldText(scoreField(S.room, t)) : 'Saved. Scan or open a recording to compare.';
  toast('Measurements saved with this scan');
};
$('btnDetails').onclick = details;
$('btnSettings').onclick = openSettings;
$('btnSaveSettings').onclick = saveSettings;
$('btnAbout').onclick = () => sheet('about');
document.addEventListener('visibilitychange', () => { if (document.hidden && S.radar && S.source === 'live') stopRadar(); });

// test hook (tools/e2e.mjs): screen point of a measured wall on the plan
window.__et = { cloud: () => ({ updates: S.cloudUpdates || 0, final: !!S.cloudFinal, only: !!S.cloudOnly, ridges: plan.cloudRidgeCount(), dims: $('dims').textContent }), live: () => ({ on: !!S.liveOn, models: !!S.liveModels, chirps: (S.liveMs || []).length, shown: S.liveWallsShown || 0 }), wallPoint: (i) => plan.wallPoint(i), wallLabel: (i) => PlanView.wallLabel(S.room, i).text, measuredWalls: () => S.room ? S.room.walls.map((w, i) => w.inferred ? -1 : i).filter(i => i >= 0) : [] };
if ('serviceWorker' in navigator && location.protocol !== 'file:') navigator.serviceWorker.register('sw.js').catch(() => {});
loadLiveModels();
loadModel().then(() => {
  const q = new URLSearchParams(location.search);
  if (q.has('sim')) simDemo(+q.get('sim') || 5007);
});
