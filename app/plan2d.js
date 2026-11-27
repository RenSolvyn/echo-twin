// Floor plan view: a flat, measured drawing of the room. Walls are lines coloured by certainty with a band
// as wide as their uncertainty; every measured wall has a dimension line from the spot that heard it and its
// distance in metres (what a tape from that spot would read). Walls the solver could not hear are dashed.
// Tap a wall to select it (calibration). Plan coordinates (x, y) are metres, y up; the screen is rotated so
// the strongest wall runs horizontally.

const COL = { bg: '#0b0d10', grid: '#151c24', grid1: '#1c2631', cell: 'rgba(64,196,255,0.06)', truth: '#69f0ae', text: '#e8eef4', muted: '#8a97a6', open: '#ffb74d', sel: '#ffffff', echo: '64,196,255' };
const certainty = (sd) => sd < 0.03 ? '#4dd0e1' : sd < 0.08 ? '#81c784' : '#ffd54f';

export class PlanView {
  constructor(canvas) {
    this.c = canvas; this.g = canvas.getContext('2d');
    this.room = null; this.truth = null; this.sel = -1; this.echoes = []; this.heading = null; this.spot = 0; this.rho = 0.3;
    this.on = false; this.insetTop = 70; this.insetBottom = 92;
    this.resize();
    window.addEventListener('resize', () => { this.resize(); this.draw(); });
    const loop = () => { if (this.on && this.dirty) { this.dirty = false; this.draw(); } requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    setInterval(() => { if (this.on && this.echoes.length) this.dirty = true; }, 100);
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = this.c.clientWidth || window.innerWidth; this.h = this.c.clientHeight || window.innerHeight;
    this.c.width = Math.round(this.w * dpr); this.c.height = Math.round(this.h * dpr);
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }

  show(on) { this.on = on; this.c.hidden = !on; if (on) { this.resize(); this.draw(); } }
  setRoom(room) { this.room = room && room.ok ? room : null; this.sel = -1; this.fit(); this.dirty = true; }
  setTruth(poly) { this.truth = poly; this.fit(); this.dirty = true; }
  setSelected(i) { this.sel = i; this.dirty = true; }
  setPhone(heading, spot = 0, rho = 0.3) { this.heading = heading; this.spot = spot; this.rho = rho; this.dirty = true; }

  // live echoes: drawn along the heading they were heard at (or along a measured bearing), fading in 4 s
  addEchoes(list, heading, spot = 0, rho = 0.3) {
    const now = performance.now();
    const s = this.spots()[spot] || [0, 0];
    for (const e of list) {
      if (e.score < 0.35) continue;
      const a = e.bearing ?? heading;
      const px = s[0] + rho * Math.cos(heading), py = s[1] + rho * Math.sin(heading);
      this.echoes.push({ x: px + e.r * Math.cos(a), y: py + e.r * Math.sin(a), t: now, k: Math.min(1, e.score) });
    }
    if (this.echoes.length > 1500) this.echoes.splice(0, this.echoes.length - 1500);
    this.dirty = true;
  }
  clearEchoes() { this.echoes = []; this.dirty = true; }

  spots() { return this.room && this.room.spots ? this.room.spots : [[0, 0]]; }

  // Wall segments to draw: { a, b, wall index, spot that heard it, open } from each convex cell.
  segments() {
    const R = this.room;
    if (!R) return [];
    const cells = R.cells || [R.poly];
    const maps = R.cellWallIndex || [R.edgeWall];
    const out = [];
    cells.forEach((P, ci) => {
      if (!P) return;
      const ew = maps[ci] || [];
      for (let i = 0; i < P.length; i++) {
        const a = P[i], b = P[(i + 1) % P.length];
        if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 0.05) continue;
        const wi = ew[i] ?? -1;
        const w = wi >= 0 ? R.walls[wi] : null;
        // clip the bounding box of an open cell to 7 m around its spot so the plan stays readable
        const s = (R.spots || [[0, 0]])[ci] || [0, 0];
        if (!w && (Math.hypot(a[0] - s[0], a[1] - s[1]) > 7 || Math.hypot(b[0] - s[0], b[1] - s[1]) > 7)) continue;
        out.push({ a, b, wi, w, cell: ci, open: !w || !!w.inferred });
      }
    });
    return out;
  }

  fit() {
    const pts = [];
    for (const s of this.segments()) pts.push(s.a, s.b);
    if (this.truth) pts.push(...this.truth);
    for (const s of this.spots()) pts.push(s);
    if (pts.length < 2) pts.push([-3, -3], [3, 3]);
    const R = this.room;
    let rot = 0;
    if (R && R.walls && R.walls.length) {
      const w0 = R.walls.filter(w => !w.inferred).sort((a, b) => (b.support || 0) - (a.support || 0))[0] || R.walls[0];
      rot = -w0.phi + Math.PI / 2; // strongest wall's normal points up, so the wall runs horizontally
    }
    this.rot = rot;
    const c = Math.cos(rot), s = Math.sin(rot);
    const q = pts.map(([x, y]) => [c * x - s * y, s * x + c * y]);
    const xs = q.map(p => p[0]), ys = q.map(p => p[1]);
    this.bx = [Math.min(...xs), Math.max(...xs)]; this.by = [Math.min(...ys), Math.max(...ys)];
  }

  // metres -> screen pixels
  view() {
    const pad = 28, W = this.w - 2 * pad, H = this.h - this.insetTop - this.insetBottom - 2 * pad;
    const sx = W / Math.max(1, this.bx[1] - this.bx[0]), sy = H / Math.max(1, this.by[1] - this.by[0]);
    const k = Math.min(sx, sy, 140);
    const cx = (this.bx[0] + this.bx[1]) / 2, cy = (this.by[0] + this.by[1]) / 2;
    const ox = this.w / 2, oy = this.insetTop + pad + H / 2;
    const c = Math.cos(this.rot || 0), s = Math.sin(this.rot || 0);
    return { k, P: ([x, y]) => { const u = c * x - s * y, v = s * x + c * y; return [ox + (u - cx) * k, oy - (v - cy) * k]; } };
  }

  // Index of the wall nearest a tap (within 26 px), or -1. Time O(edges).
  hitWall(clientX, clientY) {
    const r = this.c.getBoundingClientRect();
    const x = clientX - r.left, y = clientY - r.top;
    const { P } = this.view();
    let best = -1, bd = 26;
    for (const s of this.segments()) {
      if (s.wi < 0) continue;
      const A = P(s.a), B = P(s.b);
      const ex = B[0] - A[0], ey = B[1] - A[1], L2 = ex * ex + ey * ey || 1;
      const t = Math.max(0, Math.min(1, ((x - A[0]) * ex + (y - A[1]) * ey) / L2));
      const d = Math.hypot(A[0] + t * ex - x, A[1] + t * ey - y);
      if (d < bd) { bd = d; best = s.wi; }
    }
    return best;
  }

  // Screen point on the drawn wall (test hook for tools/e2e.mjs and for accessibility tooling).
  wallPoint(wi) {
    const r = this.c.getBoundingClientRect(), { P } = this.view();
    const sg = this.segments().filter(s => s.wi === wi).sort((a, b) => Math.hypot(b.b[0] - b.a[0], b.b[1] - b.a[1]) - Math.hypot(a.b[0] - a.a[0], a.b[1] - a.a[1]))[0];
    if (!sg) return null;
    const q = P([(sg.a[0] + sg.b[0]) / 2, (sg.a[1] + sg.b[1]) / 2]);
    return [r.left + q[0], r.top + q[1]];
  }

  // Distance label of a wall from the spot that heard it (what a tape from that spot reads).
  static wallLabel(room, wi) {
    const w = room.walls[wi];
    const sp = (room.spots || [[0, 0]])[(w.spots && w.spots.length === 1 ? w.spots[0] : 0)] || [0, 0];
    const d = w.d - (Math.cos(w.phi) * sp[0] + Math.sin(w.phi) * sp[1]);
    return { d, text: `${d.toFixed(2)} m`, spot: sp };
  }

  draw() {
    const g = this.g, { k, P } = this.view();
    g.fillStyle = COL.bg; g.fillRect(0, 0, this.w, this.h);
    // grid, 0.5 m and 1 m, aligned to the plan
    const span = Math.max(this.bx[1] - this.bx[0], this.by[1] - this.by[0]) + 4;
    const cx = (this.bx[0] + this.bx[1]) / 2, cy = (this.by[0] + this.by[1]) / 2;
    const c = Math.cos(-(this.rot || 0)), s = Math.sin(-(this.rot || 0));
    const back = (u, v) => [c * u - s * v, s * u + c * v];
    for (let m = Math.floor(-span); m <= span; m += 0.5) {
      const major = Math.abs(m % 1) < 1e-9;
      g.strokeStyle = major ? COL.grid1 : COL.grid; g.lineWidth = 1;
      const a1 = P(back(cx + m, cy - span)), b1 = P(back(cx + m, cy + span));
      const a2 = P(back(cx - span, cy + m)), b2 = P(back(cx + span, cy + m));
      g.beginPath(); g.moveTo(a1[0], a1[1]); g.lineTo(b1[0], b1[1]); g.moveTo(a2[0], a2[1]); g.lineTo(b2[0], b2[1]); g.stroke();
    }
    const R = this.room;
    // cells
    if (R) for (const cell of (R.cells || [R.poly])) {
      if (!cell || cell.length < 3) continue;
      g.fillStyle = COL.cell; g.beginPath();
      cell.forEach((p, i) => { const q = P(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
      g.closePath(); g.fill();
    }
    // truth (simulated rooms only)
    if (this.truth) {
      g.setLineDash([6, 5]); g.strokeStyle = COL.truth; g.lineWidth = 1.5; g.beginPath();
      this.truth.forEach((p, i) => { const q = P(p); i ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
      g.closePath(); g.stroke(); g.setLineDash([]);
    }
    // echoes
    const now = performance.now();
    this.echoes = this.echoes.filter(e => now - e.t < 4000);
    for (const e of this.echoes) {
      const a = (1 - (now - e.t) / 4000) * e.k;
      const q = P([e.x, e.y]);
      g.fillStyle = `rgba(${COL.echo},${a.toFixed(3)})`; g.fillRect(q[0] - 1.5, q[1] - 1.5, 3, 3);
    }
    // walls: uncertainty band, then the line
    const segs = this.segments();
    for (const sg of segs) {
      const A = P(sg.a), B = P(sg.b);
      if (sg.open) {
        g.setLineDash([7, 6]); g.strokeStyle = COL.open; g.globalAlpha = 0.7; g.lineWidth = 2;
        g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
        continue;
      }
      const col = certainty(sg.w.sD ?? 0.2);
      g.strokeStyle = col; g.globalAlpha = 0.22; g.lineWidth = Math.max(4, 4 * (sg.w.sD ?? 0.02) * k);
      g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
      g.globalAlpha = 1; g.lineWidth = sg.wi === this.sel ? 4 : 2.5; g.strokeStyle = sg.wi === this.sel ? COL.sel : col;
      g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
    }
    // dimension lines and labels, once per wall (the longest drawn segment of it)
    if (R) {
      const longest = new Map();
      for (const sg of segs) if (!sg.open) { const L = Math.hypot(sg.b[0] - sg.a[0], sg.b[1] - sg.a[1]); if (!longest.has(sg.wi) || L > longest.get(sg.wi).L) longest.set(sg.wi, { ...sg, L }); }
      g.font = '600 13px -apple-system, Roboto, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
      for (const [wi, sg] of longest) {
        const lab = PlanView.wallLabel(R, wi);
        const w = R.walls[wi], n = [Math.cos(w.phi), Math.sin(w.phi)];
        const sp = lab.spot;
        // foot of the perpendicular from the spot, clamped onto the drawn segment
        let foot = [sp[0] + lab.d * n[0], sp[1] + lab.d * n[1]];
        const ex = sg.b[0] - sg.a[0], ey = sg.b[1] - sg.a[1], L2 = ex * ex + ey * ey;
        const tt = Math.max(0.12, Math.min(0.88, ((foot[0] - sg.a[0]) * ex + (foot[1] - sg.a[1]) * ey) / L2));
        const onSeg = [sg.a[0] + tt * ex, sg.a[1] + tt * ey];
        const S0 = P(sp), F = P(foot);
        g.setLineDash([2, 4]); g.strokeStyle = 'rgba(232,238,244,0.35)'; g.lineWidth = 1;
        g.beginPath(); g.moveTo(S0[0], S0[1]); g.lineTo(F[0], F[1]); g.stroke(); g.setLineDash([]);
        const Q = P([onSeg[0] - 0.28 * n[0], onSeg[1] - 0.28 * n[1]]);
        const txt = lab.text + (w.sD != null ? ` \u00b1${Math.max(1, Math.round(100 * w.sD))}` : '');
        const tw = g.measureText(txt).width + 12;
        g.fillStyle = wi === this.sel ? '#ffffff' : 'rgba(14,19,24,0.88)'; g.beginPath();
        if (g.roundRect) g.roundRect(Q[0] - tw / 2, Q[1] - 11, tw, 22, 7); else g.rect(Q[0] - tw / 2, Q[1] - 11, tw, 22);
        g.fill();
        g.fillStyle = wi === this.sel ? '#0b0d10' : COL.text; g.fillText(txt, Q[0], Q[1] + 0.5);
      }
    }
    // spots and the phone
    this.spots().forEach((sp, i) => {
      const q = P(sp);
      g.fillStyle = '#e8eef4'; g.beginPath(); g.arc(q[0], q[1], 6, 0, 2 * Math.PI); g.fill();
      g.fillStyle = COL.muted; g.font = '12px -apple-system, Roboto, sans-serif'; g.textAlign = 'left';
      g.fillText(this.spots().length > 1 ? `spot ${'AB'[i] || i + 1}` : 'you', q[0] + 9, q[1] - 9);
    });
    if (this.heading != null) {
      const sp = this.spots()[this.spot] || [0, 0];
      const h = this.heading, a = [sp[0] + this.rho * Math.cos(h), sp[1] + this.rho * Math.sin(h)];
      const b = [a[0] + 0.45 * Math.cos(h), a[1] + 0.45 * Math.sin(h)];
      const A = P(a), B = P(b);
      g.strokeStyle = '#40c4ff'; g.lineWidth = 3; g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
      g.fillStyle = '#40c4ff'; g.beginPath(); g.arc(A[0], A[1], 4, 0, 2 * Math.PI); g.fill();
    }
    // scale bar: 1 m
    const y0 = this.h - this.insetBottom - 10, x0 = 18;
    g.strokeStyle = COL.muted; g.lineWidth = 2; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x0 + k, y0); g.moveTo(x0, y0 - 4); g.lineTo(x0, y0 + 4); g.moveTo(x0 + k, y0 - 4); g.lineTo(x0 + k, y0 + 4); g.stroke();
    g.fillStyle = COL.muted; g.font = '12px -apple-system, Roboto, sans-serif'; g.textAlign = 'left'; g.fillText('1 m', x0 + k + 6, y0 + 1);
    if (!R) { g.fillStyle = COL.muted; g.textAlign = 'center'; g.font = '14px -apple-system, Roboto, sans-serif'; g.fillText('No room yet', this.w / 2, this.h / 2); }
  }
}
