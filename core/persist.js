// Perturbation persistence (after Lewowicz's persistent points and the stability theorem of
// persistent homology): an answer is trusted only if it stays within eps of itself when the input is
// perturbed by about the size of the known sensor errors. This catches ambiguous or search-dependent
// rooms. It cannot catch a room that is wrong in the same way every time (a model error), so it is a
// necessary condition for confidence, never a sufficient one.
//
// Cost: K extra solves. Time O(K * solve). Space O(frames) per perturbed copy.
import { solveScan } from './estimate.js';
import { iou, DEG } from './geom.js';
import { makeRng } from './rng.js';

export const PERSIST = {
  kinds: ['heading', 'drop', 'noise', 'heading+drop'],
  headingSigmaDeg: 0.6, // per-frame gyro noise
  driftDeg: 2.0, // accumulated drift across the turn, uniform in +-driftDeg
  keepFrac: 0.75, // fraction of turn frames kept in a drop perturbation
  noiseRel: 1.0, // added envelope noise, in units of each frame's median level
  // Minimum plan IoU under every perturbation, set by the pre-registered rule in SANDBOX.md. On the
  // v0.2.0 solver (validation seeds 5000 to 5039) no grid value reaches 90 % precision: at 0.70 two rooms
  // qualify and one is right, above that none qualify. So confidence is unavailable and nothing is
  // reported as confident. (On the v0.1 solver the rule gave 0.75 from a single room.)
  tau: null
};

function median(a) {
  const s = Array.from(a).sort((x, y) => x - y);
  return s.length ? s[s.length >> 1] : 0;
}

export function perturbFrames(frames, kind, rng) {
  const turn = frames.filter(f => f.phase === 'turn');
  const t0 = turn.length ? turn[0].t : 0, t1 = turn.length ? turn[turn.length - 1].t : 1;
  const drift = rng.uniform(-1, 1) * PERSIST.driftDeg * DEG;
  const out = [];
  for (const f of frames) {
    if (f.phase !== 'turn') { out.push(f); continue; }
    if (kind.includes('drop') && rng.next() > PERSIST.keepFrac) continue;
    let g = f;
    if (kind.includes('heading')) {
      const x = (f.t - t0) / Math.max(1e-6, t1 - t0);
      g = { ...g, yaw: f.yaw + drift * x + PERSIST.headingSigmaDeg * DEG * rng.normal() };
    }
    if (kind.includes('noise')) {
      const m = median(f.profile) * PERSIST.noiseRel;
      const p = Float32Array.from(f.profile, v => v + m * Math.abs(rng.normal()));
      g = { ...g, profile: p };
    }
    out.push(g);
  }
  return out;
}

// Returns { minIoU, medianIoU, ious, ok } comparing the base answer with K perturbed answers.
export function persistence(frames, opt, base, seed = 1) {
  if (!base || !base.ok || !base.poly) return { minIoU: 0, medianIoU: 0, ious: [], ok: false };
  const rng = makeRng(seed + frames.length);
  const ious = [];
  for (const kind of PERSIST.kinds) {
    const res = solveScan(perturbFrames(frames, kind, rng), opt);
    ious.push(res.ok && res.poly ? iou(base.poly, res.poly) : 0);
  }
  const s = ious.slice().sort((a, b) => a - b);
  return { minIoU: s[0], medianIoU: s[s.length >> 1], ious, ok: true };
}

// Confidence = structurally plausible AND persistent. With tau unset, nothing is confident.
export function isConfident(res, p, tau = PERSIST.tau) {
  return !!(tau != null && res && res.ok && res.plausible && p && p.ok && p.minIoU >= tau);
}
