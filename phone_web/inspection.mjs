// Pure, testable trigger state. RGB probes are sampled from the inspection area.
export function changedFraction(a, b, threshold = 22) {
  if (!a || !b || a.length !== b.length || !a.length) return 1;
  let changed = 0;
  for (let i = 0; i < a.length; i += 4) {
    if ((Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) +
         Math.abs(a[i + 2] - b[i + 2])) / 3 > threshold) changed++;
  }
  return changed / (a.length / 4);
}

export class InspectionGate {
  constructor({settleMs = 1250, emptyMs = 750} = {}) {
    this.settleMs = settleMs;
    this.emptyMs = emptyMs;
    this.token = 0;
    this.armed = true;
    this.valid = false;
    this.previous = null;
    this.anchor = null;
    this.stableSince = null;
    this.emptySince = null;
  }
  invalidate() { if (this.valid) this.token++; this.valid = false; }
  retry() {
    this.invalidate();
    this.armed = true;
    this.stableSince = null;
    this.anchor = null;
  }
  accepts(token) { return this.valid && token === this.token; }
  update(sample, background, now) {
    const foreground = changedFraction(sample, background);
    const motion = changedFraction(sample, this.previous);
    // Anchor comparison also rejects slow cumulative movement between probes.
    const moving = motion > .015 || (this.anchor && changedFraction(sample, this.anchor) > .015);
    this.previous = sample;
    if (foreground < .05) {
      this.invalidate();
      this.anchor = null;
      this.stableSince = null;
      this.emptySince ??= now;
      if (now - this.emptySince >= this.emptyMs) this.armed = true;
      return {state: 'empty', foreground, motion};
    }
    this.emptySince = null;
    if (moving) {
      this.invalidate();
      this.stableSince = null;
      this.anchor = null;
      return {state: this.armed ? 'moving' : 'clear', foreground, motion};
    }
    if (!this.armed) return {state: this.valid ? 'holding' : 'clear', foreground, motion};
    if (this.stableSince === null) {
      this.stableSince = now;
      this.anchor = sample;
    }
    if (now - this.stableSince < this.settleMs) return {state: 'settling', foreground, motion};
    this.armed = false;
    this.valid = true;
    this.token++;
    return {state: 'capture', token: this.token, foreground, motion};
  }
}
