// AudioWorklet recorder: forwards microphone samples in blocks with the audio-clock frame index
// of their first sample, so every chunk is stamped at acquisition (never at arrival on the main thread).
// When the browser delivers two input channels, both are forwarded (v0.5, two microphones).
class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(2048);
    this.buf2 = new Float32Array(2048);
    this.n = 0;
    this.start = 0;
    this.two = false;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    const ch2 = inputs[0] && inputs[0][1];
    if (ch) {
      if (this.n === 0) { this.start = currentFrame; this.two = !!ch2; }
      this.buf.set(ch, this.n);
      if (this.two && ch2) this.buf2.set(ch2, this.n);
      this.n += ch.length;
      if (this.n >= this.buf.length) {
        const msg = { frame: this.start, data: this.buf.slice(0, this.n) };
        if (this.two) msg.data2 = this.buf2.slice(0, this.n);
        this.port.postMessage(msg);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('echo-rec', Recorder);
