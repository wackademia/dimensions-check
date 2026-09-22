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

// Waits for a manually-placed item to stop moving before a single inspection capture.
// The operator only presses "Start inspection" once an item is already in view, so
// unlike the old always-on InspectionGate this makes no attempt to detect presence or
// removal itself — a bad or empty frame is instead rejected by the server's own
// frame-quality checks (underwear_sizes.check_inspection_mask/landmarks) after capture.
export class SettleGate {
  constructor({settleMs = 1250} = {}) {
    this.settleMs = settleMs;
    this.previous = null;
    this.anchor = null;
    this.stableSince = null;
  }
  update(sample, now) {
    const motion = changedFraction(sample, this.previous);
    // Anchor comparison also rejects slow cumulative movement between probes.
    const moving = motion > .015 || (this.anchor && changedFraction(sample, this.anchor) > .015);
    this.previous = sample;
    if (moving) {
      this.stableSince = null;
      this.anchor = null;
      return {state: 'moving', motion};
    }
    if (this.stableSince === null) {
      this.stableSince = now;
      this.anchor = sample;
    }
    if (now - this.stableSince < this.settleMs) return {state: 'settling', motion};
    return {state: 'ready', motion};
  }
}
