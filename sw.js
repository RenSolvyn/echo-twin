// Offline cache: every app file is cached on install; cache-first with background refresh.
const CACHE = 'echo-twin-v0.6.0';
const FILES = [
  './', 'index.html', 'app.css', 'manifest.webmanifest', 'rec-worklet.js',
  'app/main.js', 'app/audio.js', 'app/sensors.js', 'app/view3d.js', 'app/plan2d.js', 'app/radargram.js', 'app/worker.js', 'app/liveworker.js',
  'core/config.js', 'core/dsp.js', 'core/estimate.js', 'core/fft.js', 'core/geom.js', 'core/linalg.js',
  'core/metrics.js', 'core/model.js', 'core/persist.js', 'core/rng.js', 'core/select.js', 'core/fieldscore.js', 'core/merge.js', 'core/calib.js', 'core/stereo.js', 'core/echomodel.js', 'core/live.js', 'core/sim.js', 'core/twin.js',
  'model/echo-mlp.json', 'model/echo-e.json', 'model/live-w.json', 'vendor/three.min.js', 'icons/icon-192.png', 'icons/icon-512.png'
];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then(hit => {
    const net = fetch(e.request).then(r => {
      if (r.ok && new URL(e.request.url).origin === location.origin) { const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); }
      return r;
    }).catch(() => hit);
    return hit || net;
  }));
});
