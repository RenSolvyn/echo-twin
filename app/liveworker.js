// Live walls during the turn (v0.6): the echo front end and the live decoder run here, one chirp per message,
// so the plan view can draw walls as they appear without slowing the screen. Messages:
//   { type: 'start', proc: { fs, f0, f1, chirpDur, c, rangeOffset, maxRange, binSize, kappa }, echo, live, stereo }
//   { type: 'chirp', f: { yaw, omega, phase, dir, profile, seg, far } }  ->  { type: 'walls', walls, rho, ms }
import { EchoProcessor } from '../core/dsp.js';
import { EchoFront, SmallMLP } from '../core/echomodel.js';
import { LiveWalls } from '../core/live.js';

let front = null, live = null, stereo = null;

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'start') {
    const proc = new EchoProcessor(m.proc);
    front = new EchoFront(proc, { model: m.echo ? new SmallMLP(m.echo) : null });
    live = new LiveWalls({ kappa: m.proc.kappa, model: m.live ? new SmallMLP(m.live) : null, tau: m.live ? m.live.tau : 0.75 });
    stereo = m.stereo || null;
    return;
  }
  if (m.type === 'stereo') { stereo = m.stereo; return; }
  if (m.type === 'chirp' && front) {
    const t0 = performance.now();
    const f = m.f;
    const cands = front.step({ ...f, stereo: f.far && stereo ? stereo : null });
    if (f.phase !== 'turn') return;
    live.add(f.yaw, f.omega || 0, cands);
    const out = live.walls();
    self.postMessage({ type: 'walls', walls: out.walls, rho: out.rho, ms: performance.now() - t0 });
  }
};
