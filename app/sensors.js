// Motion sensors: heading of the phone's bottom edge (unwrapped, from the gyro-fused orientation),
// and vertical displacement during the lift by zero-velocity-corrected integration of acceleration.
// Every sample keeps its event timestamp (performance.now() clock) for interpolation at chirp time.
// v0.6.2: compass heading of the top edge (degrees clockwise from magnetic north): iOS webkitCompassHeading,
// elsewhere 360 - alpha of an absolute orientation event. The yaw above is not tied to north (iOS alpha starts
// at an arbitrary direction); the compass lets field-test bearings be matched to walls (core/fieldscore.js).

export class Motion {
  constructor() {
    this.yawS = []; this.acc = []; this.hdgS = [];
    this.unwrapped = null; this.last = null;
    this.on = false;
    this.orient = (e) => this.onOrient(e);
    this.motion = (e) => this.onMotion(e);
    this.orientAbs = (e) => this.onCompass(e, true);
  }

  // Must run inside a user gesture on iOS.
  static async requestPermission() {
    const res = {};
    for (const [name, C] of [['orientation', window.DeviceOrientationEvent], ['motion', window.DeviceMotionEvent]]) {
      if (C && typeof C.requestPermission === 'function') {
        try { res[name] = await C.requestPermission(); } catch (e) { res[name] = 'denied'; }
      } else res[name] = C ? 'granted' : 'unsupported';
    }
    return res;
  }

  start() {
    if (this.on) return;
    window.addEventListener('deviceorientation', this.orient);
    window.addEventListener('devicemotion', this.motion);
    window.addEventListener('deviceorientationabsolute', this.orientAbs);
    this.on = true;
  }

  stop() {
    window.removeEventListener('deviceorientation', this.orient);
    window.removeEventListener('devicemotion', this.motion);
    window.removeEventListener('deviceorientationabsolute', this.orientAbs);
    this.on = false;
  }

  // Compass sample: iOS gives webkitCompassHeading (and webkitCompassAccuracy, -1 when unknown); an absolute
  // orientation event gives alpha from north (counterclockwise), so the top edge's heading is 360 - alpha.
  onCompass(e, absEvent) {
    let h = null, acc = null;
    if (typeof e.webkitCompassHeading === 'number' && e.webkitCompassHeading >= 0) { h = e.webkitCompassHeading; acc = e.webkitCompassAccuracy >= 0 ? e.webkitCompassAccuracy : null; }
    else if ((absEvent || e.absolute === true) && e.alpha != null) h = (360 - e.alpha) % 360;
    if (h == null) return;
    this.hdgS.push([e.timeStamp || performance.now(), h, acc]);
    if (this.hdgS.length > 4000) this.hdgS.splice(0, 1000);
  }

  // Latest compass reading { deg, acc, ageMs } or null. Time O(1).
  heading() {
    const s = this.hdgS[this.hdgS.length - 1];
    return s ? { deg: s[1], acc: s[2], ageMs: performance.now() - s[0] } : null;
  }

  // Compass heading at time t (nearest sample within 250 ms), or null. Time O(log n).
  hdgAt(t) {
    const S = this.hdgS;
    if (!S.length) return null;
    let lo = 0, hi = S.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m][0] <= t) lo = m; else hi = m; }
    const k = Math.abs(S[lo][0] - t) <= Math.abs(S[hi][0] - t) ? lo : hi;
    return Math.abs(S[k][0] - t) <= 250 ? S[k][1] : null;
  }

  onOrient(e) {
    this.onCompass(e, false);
    if (e.alpha == null) return;
    const D = Math.PI / 180;
    const a = e.alpha * D, b = (e.beta || 0) * D, g = (e.gamma || 0) * D;
    // world direction of the device -Y axis (bottom edge) from the W3C Z-X-Y rotation
    const cA = Math.cos(a), sA = Math.sin(a), cB = Math.cos(b), sB = Math.sin(b);
    const vx = sA * cB, vy = -cA * cB;
    let yaw = Math.atan2(vy, vx);
    if (Math.hypot(vx, vy) < 0.2) yaw = a; // nearly vertical: fall back to alpha
    if (this.last == null) this.unwrapped = yaw;
    else {
      let d = yaw - this.last;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      this.unwrapped += d;
    }
    this.last = yaw;
    this.yawS.push([e.timeStamp || performance.now(), this.unwrapped, e.beta || 0, e.gamma || 0]);
    if (this.yawS.length > 4000) this.yawS.splice(0, 1000);
  }

  onMotion(e) {
    const a = e.acceleration;
    if (!a || a.z == null) return;
    this.acc.push([e.timeStamp || performance.now(), a.z, e.rotationRate ? e.rotationRate.alpha || 0 : 0]);
    if (this.acc.length > 6000) this.acc.splice(0, 2000);
  }

  get ready() { return this.yawS.length > 2; }

  yawAt(t) {
    const S = this.yawS;
    if (!S.length) return 0;
    if (t <= S[0][0]) return S[0][1];
    if (t >= S[S.length - 1][0]) return S[S.length - 1][1];
    let lo = 0, hi = S.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m][0] <= t) lo = m; else hi = m; }
    const u = (t - S[lo][0]) / (S[hi][0] - S[lo][0] || 1);
    return S[lo][1] + u * (S[hi][1] - S[lo][1]);
  }

  // Vertical displacement over [t0, t1] (ms), zero velocity assumed at both ends. Returns f(t) -> metres.
  liftProfile(t0, t1) {
    const A = this.acc.filter(s => s[0] >= t0 && s[0] <= t1);
    if (A.length < 10) return null;
    const v = [0], z = [0];
    for (let i = 1; i < A.length; i++) {
      const dt = (A[i][0] - A[i - 1][0]) / 1000;
      v.push(v[i - 1] + 0.5 * (A[i][1] + A[i - 1][1]) * dt);
    }
    const T = (A[A.length - 1][0] - A[0][0]) || 1;
    const vEnd = v[v.length - 1];
    for (let i = 0; i < v.length; i++) v[i] -= vEnd * (A[i][0] - A[0][0]) / T;
    for (let i = 1; i < A.length; i++) z.push(z[i - 1] + 0.5 * (v[i] + v[i - 1]) * (A[i][0] - A[i - 1][0]) / 1000);
    return (t) => {
      if (t <= A[0][0]) return z[0];
      if (t >= A[A.length - 1][0]) return z[z.length - 1];
      let i = 1; while (A[i][0] < t) i++;
      const u = (t - A[i - 1][0]) / (A[i][0] - A[i - 1][0] || 1);
      return z[i - 1] + u * (z[i] - z[i - 1]);
    };
  }
}
