// Two microphones (SANDBOX.md E4). Phones have a microphone at each end; browsers expose them as two channels
// only on some hardware. With the phone held flat and pointing ahead, the speaker-side (near) microphone is
// ahead and the other one is a spacing L behind it, so an echo arriving at angle a from the pointing direction
// reaches the far microphone L cos(a) later. Each channel's profile is aligned to its own direct path, which
// itself arrives about L later at the far microphone, so for one echo
//   r_far - r_near = L (cos a - 1) / 2,   cos a = 1 + 2 (r_far - r_near) / L.
// For a wall, a = heading - wall normal angle: the same angle as in the range model, so the Hough transform
// can weight every vote by how well the measured cosine agrees with the voted wall.

export const STEREO = { gapMin: 0.06, gapMax: 0.25, sigma: 0.15, floorW: 0.25 };

// Decides whether a recording has two distinct microphones. frames carry profile, n0 (direct-path sample
// index) and, for the second channel, profile2 and n02. Returns { ok, gap, swapped, reason }. Time O(frames x bins).
export function stereoSetup(frames, proc) {
  const F = frames.filter(f => f.profile2 && Number.isFinite(f.n0) && Number.isFinite(f.n02));
  if (F.length < 10) return { ok: false, reason: 'one channel' };
  let same = 0;
  for (const f of F.slice(0, 20)) {
    let m = 0;
    for (let i = 0; i < f.profile.length; i++) m = Math.max(m, Math.abs(f.profile[i] - f.profile2[i]));
    if (m < 1e-6) same++;
  }
  if (same >= Math.min(20, F.length) - 1) return { ok: false, reason: 'the two channels are identical' };
  // median by the mean of the two middle values, so swapping the channels negates it exactly
  const dn = F.map(f => f.n02 - f.n0).sort((a, b) => a - b);
  const med = 0.5 * (dn[(dn.length - 1) >> 1] + dn[dn.length >> 1]);
  const gap = Math.abs(med) * proc.c / proc.fs;
  if (!(gap >= STEREO.gapMin && gap <= STEREO.gapMax)) return { ok: false, gap, reason: `direct-path delay between channels is ${(100 * gap).toFixed(1)} cm of path, outside 6 to 25 cm` };
  return { ok: true, gap, swapped: med < 0 };
}

// Near and far profiles of one frame under a setup.
export function nearFar(f, st) {
  return st.swapped ? { near: f.profile2, far: f.profile, n0: f.n02 } : { near: f.profile, far: f.profile2, n0: f.n0 };
}

// cos a for an echo at range r (metres) on the near channel: the strongest far-channel peak within one
// spacing below r, refined to sub-bin precision. Returns null when the peak sits on the search edge.
// Time O(gap / binSize).
export function echoCos(near, far, r, gap, binSize) {
  const lo = Math.max(1, Math.floor((r - gap - 0.02) / binSize)), hi = Math.min(far.length - 2, Math.ceil((r + 0.02) / binSize));
  if (hi - lo < 3) return null;
  let bi = -1, bv = -Infinity;
  for (let i = lo; i <= hi; i++) if (far[i] > bv) { bv = far[i]; bi = i; }
  if (bi <= lo || bi >= hi) return null;
  const a = far[bi - 1], b = far[bi], c = far[bi + 1], den = a - 2 * b + c;
  const fr = den !== 0 ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) : 0;
  const rf = (bi + fr) * binSize;
  return Math.max(-1, Math.min(1, 1 + 2 * (rf - r) / gap));
}

// Attaches cos to candidates (c.k indexes the frames they came from). Time O(candidates x gap / binSize).
export function attachCos(cands, frames, st, binSize) {
  for (const c of cands) {
    const f = frames[c.k];
    if (!f || !f.far) continue;
    c.cos = echoCos(f.profile, f.far, c.r, st.gap, binSize);
  }
  return cands;
}

// Vote weight factor for a candidate with measured cosine cm voting for a wall at heading offset with cosine cv.
export function cosWeight(cm, cv) {
  if (cm == null) return 1;
  const e = cm - cv;
  return STEREO.floorW + (1 - STEREO.floorW) * Math.exp(-e * e / (2 * STEREO.sigma * STEREO.sigma));
}

// Default for phones that give two distinct channels; set only by the pre-registered sealed test (SANDBOX.md E4).
// Adopted 2026-11-27: sealed recall 0.57 -> 0.72 (paired +0.16, 95 % interval +0.08 to +0.24), precision 0.53 -> 0.70.
export const STEREO_DEFAULT = true;
