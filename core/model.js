// Echo candidate scoring: a tiny MLP (40 -> 16 -> 1) trained on simulated echoes, and the amplitude
// baseline it has to beat. Inference is plain JavaScript: about 700 multiply-adds per candidate.

export const WIN_LO = -24, WIN_HI = 8;
export const N_FEAT = (WIN_HI - WIN_LO + 1) + 7;

// Features for one peak in a residual profile.
// ctx: { floor, frameMax, rank, bwKHz, nBins }
export function features(raw, res, pk, ctx, out = new Float32Array(N_FEAT)) {
  const b = pk.bin, a = pk.amp || 1e-9;
  let j = 0;
  for (let o = WIN_LO; o <= WIN_HI; o++) {
    const i = b + o;
    const v = i >= 0 && i < res.length ? res[i] / a : 0;
    out[j++] = v > 1.5 ? 1.5 : v;
  }
  out[j++] = Math.log10(a / ctx.floor + 1e-9) / 3;
  out[j++] = Math.log10(a / ctx.frameMax + 1e-9) / 3;
  out[j++] = pk.i / ctx.nBins;
  out[j++] = raw[b] > 0 ? res[b] / raw[b] : 0;
  let l = b, r = b;
  while (l > 0 && res[l - 1] > a / 2) l--;
  while (r < res.length - 1 && res[r + 1] > a / 2) r++;
  out[j++] = (r - l + 1) / 10;
  out[j++] = ctx.rank / 10;
  out[j++] = ctx.bwKHz / 10;
  return out;
}

// Baseline: CFAR-style amplitude over the noise floor, squashed to [0, 1].
export function amplitudeScore(pk, ctx) {
  const s = Math.log10(pk.amp / ctx.floor + 1e-9) / 2;
  return s < 0 ? 0 : s > 1 ? 1 : s;
}

export class EchoMLP {
  constructor(json) {
    this.json = json;
    this.H = json.hidden;
    this.F = json.inputs;
    this.W1 = Float32Array.from(json.W1);
    this.b1 = Float32Array.from(json.b1);
    this.W2 = Float32Array.from(json.W2);
    this.b2 = json.b2;
    this.mean = Float32Array.from(json.mean);
    this.std = Float32Array.from(json.std);
    this.h = new Float32Array(this.H);
  }
  score(x) {
    const { H, F, W1, b1, W2, mean, std, h } = this;
    let o = this.b2;
    for (let k = 0; k < H; k++) {
      let s = b1[k];
      const row = k * F;
      for (let i = 0; i < F; i++) s += W1[row + i] * ((x[i] - mean[i]) / std[i]);
      h[k] = Math.tanh(s);
      o += W2[k] * h[k];
    }
    return 1 / (1 + Math.exp(-o));
  }
}

// Average precision of a scored, labelled list (labels 0/1).
export function averagePrecision(scores, labels) {
  const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
  let tp = 0, ap = 0;
  const P = labels.reduce((a, b) => a + b, 0);
  if (!P) return 0;
  idx.forEach((i, rank) => {
    if (labels[i]) { tp++; ap += tp / (rank + 1); }
  });
  return ap / P;
}
