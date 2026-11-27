// Live walls during the turn (v0.6): the echo front end and the live decoder run here, one chirp per message,
// so the plan view can draw walls as they appear without slowing the screen. Messages:
//   { type: 'start', proc: { fs, f0, f1, chirpDur, c, rangeOffset, maxRange, binSize, kappa }, echo, live, stereo }
//   { type: 'chirp', f: { yaw, omega, phase, dir, profile, seg, far } }  ->  { type: 'walls', walls, rho, ms }
//   every 400 ms of chirps, and on { type: 'cloud' }  ->  { type: 'cloud', final, rho, nB, nR, r0, dr, heat, ridge, strength, ms }
import { EchoProcessor } from '../core/dsp.js';
import { EchoFront, SmallMLP } from '../core/echomodel.js';
import { LiveWalls, CLOUD_ENABLED } from '../core/live.js';

let front = null, live = null, stereo = null, lastCloud = 0;

// Echo cloud (v0.6.1): about 30 ms, so at most every 400 ms; buffers are transferred, not copied.
function postCloud(final) {
  if (!CLOUD_ENABLED || !live || live.hist.length < 8) { if (final) self.postMessage({ type: 'cloud', final, empty: true }); return; }
  const t0 = performance.now();
  const c = live.cloud();
  lastCloud = performance.now();
  self.postMessage({ type: 'cloud', final, rho: c.rho, nB: c.nB, nR: c.nR, r0: c.r0, dr: c.dr, heat: c.heat, ridge: c.ridge, strength: c.strength, ms: lastCloud - t0 }, [c.heat.buffer, c.ridge.buffer, c.strength.buffer]);
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'start') {
    const proc = new EchoProcessor(m.proc);
    front = new EchoFront(proc, { model: m.echo ? new SmallMLP(m.echo) : null });
    live = new LiveWalls({ kappa: m.proc.kappa, model: m.live ? new SmallMLP(m.live) : null, tau: m.live ? m.live.tau : 0.75 });
    stereo = m.stereo || null; lastCloud = 0;
    return;
  }
  if (m.type === 'stereo') { stereo = m.stereo; return; }
  if (m.type === 'cloud') { postCloud(true); return; }
  if (m.type === 'chirp' && front) {
    const t0 = performance.now();
    const f = m.f;
    const cands = front.step({ ...f, stereo: f.far && stereo ? stereo : null });
    if (f.phase !== 'turn') return;
    live.add(f.yaw, f.omega || 0, cands);
    const out = live.walls();
    self.postMessage({ type: 'walls', walls: out.walls, rho: out.rho, ms: performance.now() - t0 });
    if (performance.now() - lastCloud > 400) postCloud(false);
  }
};
