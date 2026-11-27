// Sound path check (v0.6.2): does a wall echo come back, and which audio setting brings it back best?
//
// Why: both field scans on an iPhone (Safari 27.0.1) left the phone 20 to 30 dB too quiet (direct path -31 and
// -48 dBFS, nothing above 19 kHz). With the microphone open and echo cancelling off, that iOS version plays
// through the earpiece at low volume (WebKit bug 326286), and the earpiece faces up and back in the scan pose.
// The old check scored only the direct path, so it passed while no wall was audible. This check scores what the
// solver needs: the strongest echo 0.6 to 3 m out, over the profile's median floor, averaged over chirps, with
// the phone held as in the scan and a wall about a metre away. Each audio setting is measured, never assumed.

export const SOUND_PATH = {
  // Provisional pass level: scan 1 (echoes about 6 dB over the floor) showed faint walls, scan 2 (0 dB) none.
  // Recorded with every scan so field-test truth can set it properly.
  passDb: 15,
  // The direct path is the clock every echo is timed from; below this it cannot be found reliably
  // (echo cancelling can remove it).
  directMinDb: 20,
  r0: 0.6, r1: 3.0,   // m: past the hand and body, within a room
  bgM: 0.15,          // m: each side of a peak searched for its surroundings
  chirps: 14,
};

// Audio settings to try. session: navigator.audioSession type set before the microphone opens ('auto' leaves
// it to Safari); aec: echo cancelling (Safari keeps the loudspeaker with it on, but it may also cancel echoes).
export const AUDIO_MODES = {
  'pr-off': { session: 'play-and-record', aec: false, label: 'quiet path' },
  'auto-off': { session: 'auto', aec: false, label: 'default path' },
  'aec-on': { session: 'play-and-record', aec: true, label: 'loudspeaker path' },
};

function median(a, lo, hi) {
  const s = Array.from(a.slice(lo, hi)).sort((x, y) => x - y);
  return s.length ? s[s.length >> 1] : 0;
}

// frames: [{ profile (relative to the direct path, as EchoProcessor.process returns), snrDb }].
// Returns { echoDb, rangeM, directDb, n }. On the chirp-averaged profile, each local peak between r0 and r1 is
// scored against the larger of the median floor (beyond minRange) and its own surroundings (the higher of the
// lowest values within o.bgM on either side), so the decaying tail of the direct sound, which is high but has
// no peak, does not count as a wall (it did, at the window edge, before this rule). The best peak's score and
// range are returned, with the median direct-path SNR. Time O(n x nBins + nBins x bg). Space O(nBins).
export function echoScore(frames, binSize, minRange, o = SOUND_PATH) {
  const n = frames ? frames.length : 0;
  if (!n) return { echoDb: -Infinity, rangeM: null, directDb: -Infinity, n: 0 };
  const nb = frames[0].profile.length, avg = new Float64Array(nb);
  for (const f of frames) for (let j = 0; j < nb; j++) avg[j] += f.profile[j] / n;
  const floor = median(avg, Math.round(minRange / binSize), nb);
  const j0 = Math.max(1, Math.round(o.r0 / binSize)), j1 = Math.min(nb - 2, Math.round(o.r1 / binSize));
  const w = Math.max(1, Math.round((o.bgM ?? 0.15) / binSize));
  let best = 0, bj = -1;
  for (let j = j0; j <= j1; j++) {
    const v = avg[j];
    if (!(v > avg[j - 1] && v >= avg[j + 1])) continue;
    let lo = Infinity, hi = Infinity;
    for (let k = Math.max(0, j - w); k < j; k++) if (avg[k] < lo) lo = avg[k];
    for (let k = j + 1; k <= Math.min(nb - 1, j + w); k++) if (avg[k] < hi) hi = avg[k];
    const ref = Math.max(floor, Math.max(lo, hi));
    const sc = ref > 0 ? v / ref : 0;
    if (sc > best) { best = sc; bj = j; }
  }
  const d = frames.map(f => f.snrDb).sort((a, b) => a - b);
  return {
    echoDb: best > 0 ? 20 * Math.log10(best) : -Infinity,
    rangeM: bj >= 0 ? bj * binSize : null,
    directDb: d[d.length >> 1],
    n
  };
}

// results: [{ mode, band, echoDb, directDb, ... }] in the order tried. Returns { best, ok }:
// only settings whose direct path is usable count; among those that pass, the near-ultrasonic band wins
// (inaudible), then the stronger echo; if none pass, the strongest echo (so "scan anyway" uses the best
// available); if none are usable, the first tried. Time O(n).
export function pickConfig(results, o = SOUND_PATH) {
  if (!results || !results.length) return { best: null, ok: false };
  const usable = results.filter(r => Number.isFinite(r.directDb) && r.directDb >= o.directMinDb && Number.isFinite(r.echoDb));
  if (!usable.length) return { best: results[0], ok: false };
  const pass = usable.filter(r => r.echoDb >= o.passDb);
  const byEcho = (a, b) => b.echoDb - a.echoDb;
  if (pass.length) {
    const high = pass.filter(r => r.band === 'high').sort(byEcho);
    return { best: high.length ? high[0] : pass.sort(byEcho)[0], ok: true };
  }
  return { best: usable.sort(byEcho)[0], ok: false };
}

// The order to try: the remembered setting first, then the rest; each mode with the near-ultrasonic band, then
// (when the band is automatic) the audible band. Time O(1).
export function candidates(remembered, bandAuto, fixedBand = 'high') {
  const modes = Object.keys(AUDIO_MODES);
  const first = remembered && AUDIO_MODES[remembered.mode] ? remembered : { mode: 'pr-off', band: fixedBand };
  const bands = bandAuto ? ['high', 'audible'] : [fixedBand];
  const out = [{ mode: first.mode, band: bandAuto ? first.band : fixedBand }];
  for (const m of [first.mode, ...modes.filter(m => m !== first.mode)]) for (const b of bands) {
    if (!out.some(c => c.mode === m && c.band === b)) out.push({ mode: m, band: b });
  }
  return out;
}
