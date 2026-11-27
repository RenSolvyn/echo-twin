// Solves a scan off the main thread so the 3D view stays smooth while the room builds up.
import { solveScan, EST } from '../core/estimate.js';
import { EchoMLP } from '../core/model.js';
import { persistence, isConfident, PERSIST } from '../core/persist.js';

let mlp = null;

self.onmessage = (e) => {
  const { id, frames, opt, model } = e.data;
  if (model && !mlp) mlp = new EchoMLP(model);
  const scorer = mlp && opt.useModel !== false ? (x) => mlp.score(x) : null;
  let res;
  try {
    res = solveScan(frames, { ...opt, scorer });
    // final solve only: re-solve under small perturbations; trusted only if the room persists
    if (opt.persist && res.ok) {
      // v0.4: confidence from hypothesis odds (no re-solves). Perturbation persistence only runs if its
      // threshold was ever calibrated (it was not: SANDBOX.md).
      res.oddsMin = EST.oddsMin;
      if (EST.oddsMin != null) res.confident = !!(res.plausible && res.quality && res.quality.odds != null && res.quality.odds >= EST.oddsMin);
      else if (PERSIST.tau != null) { res.persist = persistence(frames, { ...opt, scorer }, res, 1); res.confident = isConfident(res, res.persist); res.tau = PERSIST.tau; }
    }
  } catch (err) {
    res = { ok: false, reason: 'Solver error: ' + err.message };
  }
  // keep what the app needs, drop bulky internals
  if (res.ok) {
    const { walls, wallIndex, poly, edgeWall, openEdges, closed, floor, ceil, rho, drift, sigma, inflate, objects, solveMs, span, inv, hs, nAssoc, nCands, plausible, persist, confident, tau, oddsMin, quality } = res;
    res = { ok: true, walls, wallIndex, poly, edgeWall, openEdges, closed, floor, ceil, rho, drift, sigma, inflate, objects, solveMs, span, inv, hs, nAssoc, nCands, plausible, persist, confident, tau, oddsMin, quality, explained: res.explained.length };
  } else res = { ok: false, reason: res.reason, span: res.span };
  self.postMessage({ id, res });
};
