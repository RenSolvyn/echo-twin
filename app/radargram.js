// Live echo strip shown while scanning: heading swept (x, 0 to 360 deg from the start of the turn) against
// range (y, 0 m at the top, 6 m at the bottom). Each chirp paints one column of its heading-varying echo
// energy. A wall is a sinusoid r = d - rho cos(heading - phi), so the user can watch walls appear; the
// solver's current walls are drawn on top as their predicted curves.

export class EchoStrip {
  constructor(canvas) {
    this.c = canvas; this.g = canvas.getContext('2d');
    this.maxR = 6; this.yaw0 = null; this.fits = [];
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = this.c.clientWidth || 360; this.h = this.c.clientHeight || 140;
    this.c.width = Math.round(this.w * dpr); this.c.height = Math.round(this.h * dpr);
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.reset(this.yaw0);
  }

  // Start a new turn at heading yaw0 (rad). Clears the strip and draws the axes.
  reset(yaw0) {
    this.yaw0 = yaw0; this.fits = []; this.sign = 0;
    const g = this.g;
    g.fillStyle = '#080a0d'; g.fillRect(0, 0, this.w, this.h);
    this.img = g.getImageData(0, 0, this.c.width, this.c.height);
    this.axes();
  }

  x(yaw) { const s = this.sign || 1; return 34 + (this.w - 40) * Math.min(1, Math.max(0, s * (yaw - this.yaw0) / (2 * Math.PI))); }
  y(r) { return 4 + (this.h - 18) * Math.min(1, Math.max(0, r / this.maxR)); }

  axes() {
    const g = this.g;
    g.fillStyle = '#8a97a6'; g.font = '10px -apple-system, Roboto, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const r of [0, 2, 4, 6]) g.fillText(r + ' m', 28, this.y(r));
    g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    g.fillText('0', this.x(this.yaw0 ?? 0), this.h - 2); g.fillText('turn 360', this.w - 30, this.h - 2);
  }

  // One chirp: residual echo energy per range bin (profile minus the local background), at heading yaw.
  add(yaw, residual, binSize) {
    if (this.yaw0 == null) this.reset(yaw);
    if (!this.sign && Math.abs(yaw - this.yaw0) > 0.15) this.sign = Math.sign(yaw - this.yaw0);
    if (!this.sign) return;
    const dpr = this.c.width / this.w, x0 = Math.round(this.x(yaw) * dpr), wcol = Math.max(2, Math.round(2.2 * dpr));
    let mx = 1e-9; for (let i = 0; i < residual.length; i++) if (residual[i] > mx) mx = residual[i];
    const D = this.img.data, W = this.c.width;
    for (let py = Math.round(this.y(0) * dpr); py < Math.round(this.y(this.maxR) * dpr); py++) {
      const r = (py / dpr - 4) / (this.h - 18) * this.maxR;
      const v = residual[Math.min(residual.length - 1, Math.max(0, Math.round(r / binSize)))] / mx;
      const a = Math.max(0, Math.min(1, Math.sqrt(v)));
      for (let dx = 0; dx < wcol; dx++) {
        const o = (py * W + x0 + dx) * 4;
        if (x0 + dx >= W) break;
        D[o] = 20 + 44 * a; D[o + 1] = 26 + 170 * a; D[o + 2] = 34 + 221 * a; D[o + 3] = 255;
      }
    }
    this.paint();
  }

  // Predicted curves of the solver's current walls over the swept headings.
  setFit(walls, rho) { this.fits = (walls || []).filter(w => !w.inferred).map(w => ({ phi: w.phi, d: w.d, rho: rho || 0.3 })); this.paint(); }

  paint() {
    const g = this.g;
    g.putImageData(this.img, 0, 0);
    if (this.sign) for (const f of this.fits) {
      g.strokeStyle = 'rgba(255,213,79,0.8)'; g.lineWidth = 1.2; g.beginPath();
      for (let k = 0; k <= 72; k++) {
        const yaw = this.yaw0 + this.sign * 2 * Math.PI * k / 72;
        const r = f.d - f.rho * Math.cos(yaw - f.phi);
        const X = this.x(yaw), Y = this.y(r);
        k ? g.lineTo(X, Y) : g.moveTo(X, Y);
      }
      g.stroke();
    }
    this.axes();
  }
}
