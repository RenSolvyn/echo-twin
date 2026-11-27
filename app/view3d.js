// Third-person 3D view of the twin. Plan coordinates (x, y) map to three.js (x, -y) with y up.
import {
  WebGLRenderer, Scene, PerspectiveCamera, Color, Fog, Group, Mesh, MeshBasicMaterial, MeshStandardMaterial,
  PlaneGeometry, CylinderGeometry, SphereGeometry, BoxGeometry, ConeGeometry, BufferGeometry, Float32BufferAttribute,
  Line, LineSegments, LineBasicMaterial, LineDashedMaterial, Points, PointsMaterial, Vector3, DoubleSide,
  HemisphereLight, DirectionalLight, Shape, ShapeGeometry, AdditiveBlending, BackSide
} from '../vendor/three.min.js';

const P = (x, y, h) => new Vector3(x, h, -y);

export class View3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.scene = new Scene();
    this.scene.background = new Color(0x0b0d10);
    this.scene.fog = new Fog(0x0b0d10, 9, 22);
    this.camera = new PerspectiveCamera(55, 1, 0.05, 80);
    this.scene.add(new HemisphereLight(0xbfd8ff, 0x101418, 0.9));
    const sun = new DirectionalLight(0xffffff, 0.6); sun.position.set(3, 8, 2); this.scene.add(sun);
    this.room = new Group(); this.scene.add(this.room);
    this.dyn = new Group(); this.scene.add(this.dyn);
    this.truthGroup = new Group(); this.scene.add(this.truthGroup);
    this.heading = 0; this.rho = 0.3; this.phoneH = 1.2; this.ceilH = 2.5;
    this.orbit = { yaw: 0, pitch: 0.72, dist: 6.2, follow: true };
    this.makeEgo();
    this.makePoints();
    this.blips = [];
    this.bindControls();
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.running = true;
    const loop = () => { if (this.running) { if (!this.paused) this.frame(); requestAnimationFrame(loop); } };
    requestAnimationFrame(loop);
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth, h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  makeEgo() {
    const g = new Group();
    const body = new Mesh(new CylinderGeometry(0.17, 0.2, 1.25, 20), new MeshStandardMaterial({ color: 0xdfe6ee, roughness: 0.6 }));
    body.position.y = 0.72;
    const head = new Mesh(new SphereGeometry(0.13, 20, 16), new MeshStandardMaterial({ color: 0xdfe6ee, roughness: 0.5 }));
    head.position.y = 1.52;
    g.add(body, head);
    this.phone = new Mesh(new BoxGeometry(0.075, 0.01, 0.15), new MeshStandardMaterial({ color: 0x40c4ff, emissive: 0x0a3a50 }));
    g.add(this.phone);
    const beam = new Mesh(new ConeGeometry(0.9, 2.6, 24, 1, true), new MeshBasicMaterial({ color: 0x40c4ff, transparent: true, opacity: 0.07, side: DoubleSide, depthWrite: false }));
    beam.rotation.x = -Math.PI / 2;
    this.beam = new Group(); this.beam.add(beam); beam.position.z = -1.3;
    g.add(this.beam);
    this.ego = g; this.scene.add(g);
  }

  makePoints() {
    this.nPts = 2400; this.pi = 0;
    const pos = new Float32Array(this.nPts * 3).fill(0), col = new Float32Array(this.nPts * 3).fill(0);
    this.ptGeom = new BufferGeometry();
    this.ptGeom.setAttribute('position', new Float32BufferAttribute(pos, 3));
    this.ptGeom.setAttribute('color', new Float32BufferAttribute(col, 3));
    this.ptAge = new Float32Array(this.nPts).fill(99);
    this.pts = new Points(this.ptGeom, new PointsMaterial({ size: 0.05, vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false, blending: AdditiveBlending }));
    this.scene.add(this.pts);
  }

  clear(group) { while (group.children.length) { const c = group.children.pop(); c.geometry && c.geometry.dispose(); } }

  // room: { poly, walls, edgeWall?, openEdges, floor, ceil, inferred flags }
  setRoom(room) {
    this.clear(this.room);
    this.roomData = room;
    if (!room || !room.poly) return;
    const hf = room.floor ? room.floor.h : 1.2, hc = room.ceil ? room.ceil.h : 1.3;
    this.phoneH = hf; this.ceilH = hf + hc; this.rho = room.rho || 0.3;
    const H = this.ceilH;
    // floor
    const shape = new Shape(room.poly.map(([x, y]) => ({ x, y: y })));
    const fl = new Mesh(new ShapeGeometry(shape), new MeshBasicMaterial({ color: 0x141a21, side: DoubleSide }));
    fl.rotation.x = -Math.PI / 2;
    this.room.add(fl);
    // grid lines clipped to the bounding box of the plan
    const xs = room.poly.map(p => p[0]), ys = room.poly.map(p => p[1]);
    const g = [];
    for (let x = Math.ceil(Math.min(...xs)); x <= Math.max(...xs); x += 0.5) g.push(x, 0.002, -Math.min(...ys), x, 0.002, -Math.max(...ys));
    for (let y = Math.ceil(Math.min(...ys)); y <= Math.max(...ys); y += 0.5) g.push(Math.min(...xs), 0.002, -y, Math.max(...xs), 0.002, -y);
    const gg = new BufferGeometry(); gg.setAttribute('position', new Float32BufferAttribute(g, 3));
    this.room.add(new LineSegments(gg, new LineBasicMaterial({ color: 0x1f2a35 })));
    // walls
    const n = room.poly.length;
    for (let i = 0; i < n; i++) {
      const a = room.poly[i], b = room.poly[(i + 1) % n];
      const open = room.openEdges && room.openEdges[i];
      const wi = room.edgeWall ? room.edgeWall[i] : -1;
      const w = wi >= 0 ? room.walls[wi] : null;
      const inferred = open || (w && w.inferred);
      const sd = w ? w.sD : 0.2;
      const color = inferred ? 0xffb74d : sd < 0.03 ? 0x4dd0e1 : sd < 0.08 ? 0x81c784 : 0xffd54f;
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (L < 0.05) continue;
      if (!open) {
        // inner face only: from the third-person camera outside the room, near walls do not hide the scene
        const m = new Mesh(new PlaneGeometry(L, H), new MeshBasicMaterial({ color, transparent: true, opacity: inferred ? 0.10 : 0.20, side: BackSide, depthWrite: false }));
        m.position.copy(P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, H / 2));
        m.rotation.y = Math.atan2(b[1] - a[1], b[0] - a[0]);
        this.room.add(m);
      }
      const pts = [P(a[0], a[1], 0), P(b[0], b[1], 0), P(b[0], b[1], H), P(a[0], a[1], H), P(a[0], a[1], 0)];
      const lg = new BufferGeometry().setFromPoints(pts);
      const line = inferred ? new Line(lg, new LineDashedMaterial({ color, dashSize: 0.15, gapSize: 0.1 })) : new Line(lg, new LineBasicMaterial({ color }));
      if (inferred) line.computeLineDistances();
      this.room.add(line);
    }
    // objects as soft Gaussian discs
    for (const o of room.objects || []) {
      const s = new Mesh(new SphereGeometry(Math.max(0.08, o.st), 16, 12), new MeshBasicMaterial({ color: 0xce93d8, transparent: true, opacity: 0.18, depthWrite: false }));
      s.scale.y = 0.6; s.position.copy(P(o.x, o.y, hf));
      this.room.add(s);
    }
  }

  setTruth(poly, H) {
    this.clear(this.truthGroup);
    if (!poly) return;
    const pts = poly.concat([poly[0]]);
    for (const h of [0, H]) {
      const lg = new BufferGeometry().setFromPoints(pts.map(([x, y]) => P(x, y, h)));
      const l = new Line(lg, new LineDashedMaterial({ color: 0x69f0ae, dashSize: 0.1, gapSize: 0.08 }));
      l.computeLineDistances();
      this.truthGroup.add(l);
    }
  }

  setHeading(yaw) { this.heading = yaw; }
  // camera: 'follow' (third person behind the phone), 'orbit' (free, not tied to heading), 'top' (map from above;
  // the app's plan view in app/plan2d.js replaced it as the map in v0.5)
  setMode(mode) { this.mode = mode; if (mode === 'top') this.orbit.dist = Math.max(this.orbit.dist, 7); }

  phonePos() { return [this.rho * Math.cos(this.heading), this.rho * Math.sin(this.heading)]; }

  // echo candidates of one chirp, drawn along the current heading (bearing is not measured per echo)
  addEchoes(cands) {
    const [px, py] = this.phonePos();
    const c = Math.cos(this.heading), s = Math.sin(this.heading);
    const pos = this.ptGeom.attributes.position.array, col = this.ptGeom.attributes.color.array;
    for (const e of cands) {
      if (e.score < 0.35) continue;
      const i = this.pi = (this.pi + 1) % this.nPts;
      const v = P(px + e.r * c, py + e.r * s, this.phoneH + (Math.random() - 0.5) * 0.06);
      pos[3 * i] = v.x; pos[3 * i + 1] = v.y; pos[3 * i + 2] = v.z;
      const k = Math.min(1, e.score);
      col[3 * i] = 0.25 * k; col[3 * i + 1] = 0.75 * k; col[3 * i + 2] = 1.0 * k;
      this.ptAge[i] = 0;
    }
    this.ptGeom.attributes.position.needsUpdate = true;
  }

  // traced first-order paths from the phone to the walls it can hear now (sound tracing view)
  setTraces(walls) {
    if (this.traceLines) { this.dyn.remove(this.traceLines); this.traceLines.geometry.dispose(); }
    if (!walls || !walls.length) return;
    const [px, py] = this.phonePos();
    const seg = [];
    for (const w of walls) {
      const n = [Math.cos(w.phi), Math.sin(w.phi)];
      const dist = w.d - (n[0] * px + n[1] * py);
      if (dist <= 0) continue;
      const a = P(px, py, this.phoneH), b = P(px + n[0] * dist, py + n[1] * dist, this.phoneH);
      seg.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const g = new BufferGeometry(); g.setAttribute('position', new Float32BufferAttribute(seg, 3));
    this.traceLines = new LineSegments(g, new LineBasicMaterial({ color: 0x40c4ff, transparent: true, opacity: 0.35 }));
    this.dyn.add(this.traceLines);
  }

  // moving targets: range and radial speed along the heading they were heard at
  setTracks(tracks) {
    for (const b of this.blips) this.dyn.remove(b);
    this.blips = [];
    for (const t of tracks || []) {
      const c = Math.cos(t.yaw), s = Math.sin(t.yaw);
      const px = this.rho * c, py = this.rho * s;
      const g = new Group();
      const col = t.rdot < -0.1 ? 0xff5252 : t.rdot > 0.1 ? 0x448aff : 0xffffff;
      const m = new Mesh(new CylinderGeometry(0.18, 0.18, 1.6, 16), new MeshBasicMaterial({ color: col, transparent: true, opacity: 0.45 }));
      m.position.y = 0.8;
      g.add(m);
      const arrowLen = Math.max(0.05, Math.min(1.5, Math.abs(t.rdot)));
      const ar = new Mesh(new ConeGeometry(0.08, arrowLen, 10), new MeshBasicMaterial({ color: col }));
      ar.rotation.z = t.rdot < 0 ? Math.PI / 2 : -Math.PI / 2; // local +x is the radial direction away from you
      ar.position.set(t.rdot < 0 ? -arrowLen / 2 : arrowLen / 2, 1.75, 0);
      g.add(ar);
      g.position.copy(P(px + t.r * c, py + t.r * s, 0));
      g.rotation.y = t.yaw;
      this.dyn.add(g); this.blips.push(g);
    }
  }

  bindControls() {
    let last = null, pinch = null;
    const el = this.canvas;
    el.addEventListener('pointerdown', (e) => { last = [e.clientX, e.clientY]; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', (e) => {
      if (!last || pinch) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      const k = 3 / (el.clientWidth || 400);
      this.orbit.yaw -= dx * k; this.orbit.pitch = Math.max(0.08, Math.min(1.45, this.orbit.pitch + dy * k));
    });
    el.addEventListener('pointerup', () => { last = null; });
    el.addEventListener('wheel', (e) => { this.orbit.dist = Math.max(1.5, Math.min(16, this.orbit.dist * (1 + Math.sign(e.deltaY) * 0.1))); }, { passive: true });
    el.addEventListener('touchstart', (e) => { if (e.touches.length === 2) pinch = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }, { passive: true });
    el.addEventListener('touchmove', (e) => {
      if (e.touches.length === 2 && pinch) {
        const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        this.orbit.dist = Math.max(1.5, Math.min(16, this.orbit.dist * pinch / d)); pinch = d;
      }
    }, { passive: true });
    el.addEventListener('touchend', () => { pinch = null; });
    el.addEventListener('dblclick', () => { this.orbit.yaw = 0; this.orbit.pitch = 0.72; this.orbit.dist = 6.2; });
  }

  frame() {
    const [px, py] = this.phonePos();
    this.phone.position.copy(P(px, py, this.phoneH));
    this.phone.rotation.y = -this.heading + Math.PI / 2;
    this.beam.position.copy(P(px, py, this.phoneH));
    this.beam.rotation.y = -this.heading + Math.PI / 2;
    // fade old echoes
    const col = this.ptGeom.attributes.color.array;
    for (let i = 0; i < this.nPts; i++) {
      if (this.ptAge[i] > 6) continue;
      this.ptAge[i] += 1 / 60;
      const f = Math.max(0, 1 - this.ptAge[i] / 6);
      col[3 * i] *= 0.995; col[3 * i + 1] *= 0.995; col[3 * i + 2] *= 0.995;
      if (f === 0) { col[3 * i] = col[3 * i + 1] = col[3 * i + 2] = 0; }
    }
    this.ptGeom.attributes.color.needsUpdate = true;
    // third-person camera: behind and above, looking ahead along the heading
    const mode = this.mode || 'follow';
    const yaw = (mode === 'follow' ? this.heading + Math.PI : 0) + this.orbit.yaw;
    const d = this.orbit.dist;
    const pitch = mode === 'top' ? 1.5 : this.orbit.pitch;
    const cx = Math.cos(yaw) * d * Math.cos(pitch), cy = Math.sin(yaw) * d * Math.cos(pitch);
    const target = mode === 'follow' ? P(0.8 * Math.cos(this.heading), 0.8 * Math.sin(this.heading), 0.9) : P(0, 0, 0.9);
    const want = P(cx, cy, 1.0 + d * Math.sin(pitch));
    this.camera.position.lerp(want, 0.12);
    this.camera.lookAt(target);
    this.renderer.render(this.scene, this.camera);
  }
}
