// Small dense linear algebra for the smoother and the Kalman filters (n under about 30).
// Matrices are arrays of Float64Array rows.

export function zeros(n, m = n) {
  const A = new Array(n);
  for (let i = 0; i < n; i++) A[i] = new Float64Array(m);
  return A;
}

export function eye(n, s = 1) {
  const A = zeros(n);
  for (let i = 0; i < n; i++) A[i][i] = s;
  return A;
}

export function clone(A) { return A.map(r => Float64Array.from(r)); }

// Cholesky factor L (lower) of a symmetric positive definite matrix; adds jitter if needed.
export function cholesky(A) {
  const n = A.length;
  for (let attempt = 0; attempt < 6; attempt++) {
    const jitter = attempt === 0 ? 0 : 1e-12 * Math.pow(100, attempt);
    const L = zeros(n);
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j] + (i === j ? jitter : 0);
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 0) { ok = false; break; }
          L[i][i] = Math.sqrt(s);
        } else L[i][j] = s / L[j][j];
      }
    }
    if (ok) return L;
  }
  return null;
}

export function cholSolve(L, b) {
  const n = L.length;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
    y[i] = s / L[i][i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

export function inverseSPD(A) {
  const n = A.length;
  const L = cholesky(A);
  if (!L) return null;
  const inv = zeros(n);
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n); e[j] = 1;
    const x = cholSolve(L, e);
    for (let i = 0; i < n; i++) inv[i][j] = x[i];
  }
  return inv;
}

// Inverse of a 2x2 [[a,b],[c,d]].
export function inv2(a, b, c, d) {
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-300) return null;
  return [d / det, -b / det, -c / det, a / det];
}
