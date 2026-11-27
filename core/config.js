// Shared constants for the browser app, the simulator and the Node tools.
// Units: metres, seconds, radians, hertz.

export const BANDS = {
  high: { name: 'high', f0: 16000, f1: 20000, label: 'Near-ultrasonic 16-20 kHz' },
  audible: { name: 'audible', f0: 3000, f1: 9000, label: 'Audible 3-9 kHz' },
  // v0.5: loud audible sweep, range resolution c / 2B of about 1 cm (SANDBOX.md E3); never chosen automatically.
  // Sealed test: fewer walls than 16-20 kHz (pooled recall -0.09), kept only to collect real recordings.
  wide: { name: 'wide', f0: 2000, f1: 20000, label: 'Wideband 2-20 kHz (loud, experimental)' }
};

export const DEFAULTS = {
  chirpDur: 0.010,   // s, one linear FM sweep
  period: 0.12,      // s between chirps
  preSec: 0.003,     // s of signal kept before the direct path
  maxRange: 6.0,     // m
  minRange: 0.35,    // m, closer echoes are masked by the direct path and the body
  binSize: 0.01,     // m per range bin
  tempC: 20          // C, assumed air temperature
};

// Speed of sound in dry air, m/s.
export function soundSpeed(tempC) {
  return 331.3 * Math.sqrt(1 + tempC / 273.15);
}

// Half the speaker-to-microphone distance, added back to every range.
// Per device class, not self-calibrated (see PREREG.md known limits).
export const DEVICE_PROFILES = {
  android: { rangeOffset: 0.01, label: 'Android: bottom speaker, bottom mic' },
  iphone: { rangeOffset: 0.07, label: 'iPhone: earpiece out, bottom mic' },
  other: { rangeOffset: 0.03, label: 'Other device' }
};

export function guessDeviceClass(ua) {
  const s = (ua || '').toLowerCase();
  if (/iphone|ipad|ipod/.test(s)) return 'iphone';
  if (/android/.test(s)) return 'android';
  return 'other';
}

export const VERSION = '0.6.1';
