// Sonar audio engine: plays chirps on the audio clock, records the microphone through an AudioWorklet
// into a ring buffer indexed by audio frame, and cuts one segment per chirp around its direct path.
// v0.5: asks for two channels; where the browser delivers two, both are cut at the same samples
// (core/stereo.js decides later whether they come from two distinct microphones).

import { makeChirp, EchoProcessor } from '../core/dsp.js';
import { fft, nextPow2 } from '../core/fft.js';
import { DEFAULTS } from '../core/config.js';

const RING = 1 << 20;

export class SonarAudio {
  constructor() {
    this.ctx = null; this.stream = null; this.node = null;
    this.ring = new Float32Array(RING);
    this.ring2 = null;
    this.latest = -1;
    this.pending = [];
    this.timer = null;
    this.onFrame = null;
    this.latency = null;
  }

  // Must be called from a user gesture (iOS).
  async open() {
    if (navigator.audioSession) { try { navigator.audioSession.type = 'play-and-record'; } catch (e) { /* older Safari */ } }
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC({ latencyHint: 'interactive' });
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: 2 } }
    });
    await this.ctx.audioWorklet.addModule(new URL('../rec-worklet.js', import.meta.url));
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'echo-rec', { channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'discrete' });
    const mute = this.ctx.createGain(); mute.gain.value = 0;
    src.connect(this.node); this.node.connect(mute); mute.connect(this.ctx.destination);
    this.node.port.onmessage = (e) => this.write(e.data.frame, e.data.data, e.data.data2);
    this.fs = this.ctx.sampleRate;
    const tr = this.stream.getAudioTracks()[0];
    this.trackSettings = tr && tr.getSettings ? tr.getSettings() : {};
    return { fs: this.fs, settings: this.trackSettings, channels: this.trackSettings.channelCount ?? null, session: navigator.audioSession ? navigator.audioSession.type : 'n/a' };
  }

  write(frame, data, data2) {
    for (let i = 0; i < data.length; i++) this.ring[(frame + i) % RING] = data[i];
    if (data2) {
      if (!this.ring2) this.ring2 = new Float32Array(RING);
      for (let i = 0; i < data2.length; i++) this.ring2[(frame + i) % RING] = data2[i];
    }
    this.latest = frame + data.length - 1;
    this.drain();
  }

  read(from, n, ring = this.ring) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = ring[((from + i) % RING + RING) % RING];
    return out;
  }

  configure({ band, period, rangeOffset, tempC }) {
    this.band = band; this.period = period;
    this.proc = new EchoProcessor({ fs: this.fs, f0: band.f0, f1: band.f1, chirpDur: DEFAULTS.chirpDur, tempC, rangeOffset });
    this.buffers = {};
    for (const dir of [1, -1]) {
      const c = makeChirp(this.fs, band.f0, band.f1, DEFAULTS.chirpDur, dir);
      const b = this.ctx.createBuffer(1, c.length, this.fs);
      b.copyToChannel(Float32Array.from(c, v => v * 0.9), 0);
      this.buffers[dir] = b;
    }
  }

  emit(t, dir) {
    const s = this.ctx.createBufferSource();
    s.buffer = this.buffers[dir];
    s.connect(this.ctx.destination);
    s.start(t);
  }

  // Audio-clock time to performance.now() time (when the sound actually leaves the speaker).
  perfAt(ctxTime) {
    const ts = this.ctx.getOutputTimestamp ? this.ctx.getOutputTimestamp() : null;
    if (ts && ts.performanceTime) return ts.performanceTime + (ctxTime - ts.contextTime) * 1000;
    return performance.now() + (ctxTime - this.ctx.currentTime) * 1000 + (this.ctx.outputLatency || 0) * 1000;
  }

  // One lone chirp; finds the round-trip latency from the direct path and measures its SNR.
  async soundCheck() {
    const fs = this.fs;
    const t = this.ctx.currentTime + 0.15;
    this.emit(t, 1);
    const E = Math.round(t * fs);
    await new Promise(r => setTimeout(r, 900));
    const n = Math.round(0.55 * fs);
    const seg = this.read(E, n);
    const chirp = this.proc.chirps[1];
    const N = nextPow2(n + chirp.length);
    const xr = new Float64Array(N), xi = new Float64Array(N), cr = new Float64Array(N), ci = new Float64Array(N);
    seg.forEach((v, i) => { xr[i] = v; });
    chirp.forEach((v, i) => { cr[i] = v; });
    fft(xr, xi); fft(cr, ci);
    for (let k = 0; k < N; k++) { const a = xr[k] * cr[k] + xi[k] * ci[k], b = xi[k] * cr[k] - xr[k] * ci[k]; xr[k] = a; xi[k] = b; }
    fft(xr, xi, true);
    let best = 0, bi = 0;
    const env = new Float64Array(n);
    for (let i = 0; i < n - chirp.length; i++) { env[i] = Math.abs(xr[i]); if (env[i] > best) { best = env[i]; bi = i; } }
    const sorted = Array.from(env.slice(0, n - chirp.length)).sort((a, b) => a - b);
    const floor = sorted[sorted.length >> 1] || 1e-12;
    this.latency = bi;
    const peakAbs = seg.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    return { latencyMs: 1000 * bi / fs, snrDb: 20 * Math.log10(best / floor), clipped: peakAbs > 0.98, peakAbs };
  }

  // Periodic chirps; alternate sweep direction when alt is true (Doppler radar mode).
  startTrain({ alt = false, phase = 'scan' } = {}) {
    this.stopTrain();
    this.alt = alt; this.phase = phase;
    this.next = this.ctx.currentTime + 0.1;
    this.count = 0;
    this.timer = setInterval(() => this.schedule(), 25);
  }

  stopTrain() { if (this.timer) clearInterval(this.timer); this.timer = null; this.pending = []; }

  schedule() {
    while (this.next < this.ctx.currentTime + 0.2) {
      const dir = this.alt && (this.count % 2) ? -1 : 1;
      this.emit(this.next, dir);
      this.pending.push({ E: Math.round(this.next * this.fs), dir, tCtx: this.next, perf: this.perfAt(this.next), phase: this.phase });
      this.next += this.period;
      this.count++;
    }
    this.drain();
  }

  drain() {
    if (this.latency == null || !this.proc) return;
    const p = this.proc;
    while (this.pending.length) {
      const q = this.pending[0];
      const start = q.E + this.latency - p.pre;
      if (this.latest < start + p.segLen + 8) break;
      this.pending.shift();
      const seg = this.read(start, p.segLen);
      const seg2 = this.ring2 ? this.read(start, p.segLen, this.ring2) : null;
      // interference learning from the quiet gap just before the next chirp
      if (this.period >= 0.1 && (this.count % 4 === 0)) {
        const gapStart = q.E + this.latency + Math.round((this.period - 0.035) * this.fs);
        if (this.latest > gapStart + 1024) p.learnNoise(this.read(gapStart, 1024));
      }
      if (this.onFrame) this.onFrame({ seg, seg2, dir: q.dir, tCtx: q.tCtx, perf: q.perf, phase: q.phase });
    }
  }

  async close() {
    this.stopTrain();
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx) await this.ctx.close();
    if (navigator.audioSession) { try { navigator.audioSession.type = 'playback'; navigator.audioSession.type = 'auto'; } catch (e) { /* ignore */ } }
  }
}
