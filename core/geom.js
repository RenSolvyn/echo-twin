// 2D geometry helpers. Polygons are arrays of [x, y], counter-clockwise.
// A wall (half-plane) is { phi, d }: outward normal n = (cos phi, sin phi), the room is n . x <= d.

export const TAU = 2 * Math.PI;
export const DEG = Math.PI / 180;

export function wrap(a) {
  a = (a + Math.PI) % TAU;
  if (a < 0) a += TAU;
  return a - Math.PI;
}

export function angDiff(a, b) { return Math.abs(wrap(a - b)); }

export function area(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    s += x0 * y1 - x1 * y0;
  }
  return s / 2;
}

// Intersection of lines n1 . x = d1 and n2 . x = d2. Returns null for near-parallel lines.
export function intersectWalls(w1, w2) {
  const a1 = Math.cos(w1.phi), b1 = Math.sin(w1.phi);
  const a2 = Math.cos(w2.phi), b2 = Math.sin(w2.phi);
  const det = a1 * b2 - a2 * b1;
  if (Math.abs(det) < 0.17) return null; // under about 10 degrees apart
  return [(w1.d * b2 - w2.d * b1) / det, (a1 * w2.d - a2 * w1.d) / det];
}

// Sutherland-Hodgman clip of a convex polygon (vertices carry the id of the edge that starts at them)
// by the half-plane n . x <= d. Returns a new labelled polygon.
function clipLabelled(poly, n, d, id) {
  const out = [];
  const m = poly.length;
  for (let i = 0; i < m; i++) {
    const P = poly[i], Q = poly[(i + 1) % m];
    const sp = n[0] * P.x + n[1] * P.y - d;
    const sq = n[0] * Q.x + n[1] * Q.y - d;
    if (sp <= 0) out.push({ x: P.x, y: P.y, id: P.id });
    if ((sp <= 0) !== (sq <= 0)) {
      const t = sp / (sp - sq);
      const X = { x: P.x + t * (Q.x - P.x), y: P.y + t * (Q.y - P.y) };
      // entering the kept side: edge continues along the original edge; leaving: new edge on the clip line
      X.id = sp <= 0 ? id : P.id;
      out.push(X);
    }
  }
  return out;
}

// Intersection of wall half-planes, bounded by a square of half-size `bound`.
// Returns { poly: [[x,y]], edgeWall: [wall index or -1 for the bounding box], used: Set of wall indices }.
export function roomPolygon(walls, bound = 12) {
  let poly = [
    { x: -bound, y: -bound, id: -1 }, { x: bound, y: -bound, id: -1 },
    { x: bound, y: bound, id: -1 }, { x: -bound, y: bound, id: -1 }
  ];
  walls.forEach((w, i) => {
    poly = clipLabelled(poly, [Math.cos(w.phi), Math.sin(w.phi)], w.d, i);
  });
  // drop zero-length edges
  const clean = [];
  for (let i = 0; i < poly.length; i++) {
    const P = poly[i], Q = poly[(i + 1) % poly.length];
    if (Math.hypot(Q.x - P.x, Q.y - P.y) > 1e-6) clean.push(P);
  }
  const used = new Set(clean.map(p => p.id).filter(i => i >= 0));
  return { poly: clean.map(p => [p.x, p.y]), edgeWall: clean.map(p => p.id), used };
}

// Area of the intersection of two convex CCW polygons.
export function convexIntersectionArea(A, B) {
  let poly = A.map(([x, y]) => ({ x, y, id: 0 }));
  for (let i = 0; i < B.length && poly.length; i++) {
    const [x0, y0] = B[i], [x1, y1] = B[(i + 1) % B.length];
    // outward normal of a CCW edge is (dy, -dx)
    const nx = y1 - y0, ny = -(x1 - x0);
    const L = Math.hypot(nx, ny);
    const n = [nx / L, ny / L];
    poly = clipLabelled(poly, n, n[0] * x0 + n[1] * y0, 0);
  }
  return poly.length >= 3 ? Math.abs(area(poly.map(p => [p.x, p.y]))) : 0;
}

export function iou(A, B) {
  const inter = convexIntersectionArea(A, B);
  const u = Math.abs(area(A)) + Math.abs(area(B)) - inter;
  return u > 0 ? inter / u : 0;
}

export function pointInConvex(poly, x, y) {
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    if ((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) < -1e-9) return false;
  }
  return true;
}

export function rotate([x, y], a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c * x - s * y, s * x + c * y];
}
