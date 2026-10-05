"""Rolling diagnostic recorder.

Jitter has several possible sources that look identical from the outside:
noisy measurement, a controller reacting to that noise, or the motor
reversing direction against backlash. Telling them apart needs the raw
per-frame numbers side by side, which is what this records and dumps.

The summary at the top of a dump is the part that actually answers it:

  ball_px std while the ball is still  -> measurement noise
  target changes / s, reversals / s    -> how often the motor is retasked
  cmd tilt std                         -> how much the controller is moving
"""

import math
import time
from collections import deque


class DiagRecorder:
    COLUMNS = ("t_ms", "state", "snr", "ball_px", "ball_mm", "err_mm",
               "vel_mms", "vfilt_mms", "cmd_deg", "meas_deg", "tgt_stp",
               "pos_stp")

    def __init__(self, settings):
        self.st = settings
        self.buf = deque(maxlen=settings.diag_buffer_len)
        self.t0 = time.time()

    def record(self, now, app):
        st = self.st
        tr = app.tracker
        ctrl = app.ctrl
        cur, tgt, zoff = app.motor.snapshot()
        span_mm = st.roi_span_m * 500.0     # half-span in mm
        self.buf.append((
            (now - self.t0) * 1000.0,
            (tr.state if tr else "none"),
            (tr.snr if tr else 0.0),
            (app.ball_px if app.ball_px is not None else float("nan")),
            (app.ball_norm * span_mm if app.ball_norm is not None
             else float("nan")),
            app.err_m * 1000.0,
            app.vel_mps * 1000.0,
            ctrl.vel_filt * 1000.0,
            math.degrees(ctrl.cmd_tilt),
            math.degrees(app.meas_tilt),
            tgt - zoff,
            cur - zoff,
        ))

    # ---------------- analysis ----------------
    def summary(self):
        if len(self.buf) < 5:
            return ["not enough samples yet"]
        rows = list(self.buf)
        dur = (rows[-1][0] - rows[0][0]) / 1000.0
        n = len(rows)
        fps = (n - 1) / dur if dur > 0 else 0.0

        def col(i):
            return [r[i] for r in rows]

        def std(vals):
            vals = [v for v in vals if v == v]      # drop NaN
            if len(vals) < 2:
                return float("nan")
            m = sum(vals) / len(vals)
            return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

        tgt = col(10)
        pos = col(11)
        changes = sum(1 for a, b in zip(tgt, tgt[1:]) if a != b)

        def reversals(seq):
            d = [b - a for a, b in zip(seq, seq[1:]) if b != a]
            return sum(1 for a, b in zip(d, d[1:]) if a * b < 0)

        locked = sum(1 for s in col(1) if s == "lock")
        ball_px = [v for v in col(3) if v == v]

        out = [
            "samples %d over %.2f s (%.0f fps)" % (n, dur, fps),
            "locked %d/%d frames (%.0f%%)" % (locked, n, 100.0 * locked / n),
            "ball_px  std %.2f px   range %.1f px" % (
                std(col(3)),
                (max(ball_px) - min(ball_px)) if ball_px else float("nan")),
            "err      std %.2f mm   mean %.2f mm" % (
                std(col(5)), sum(v for v in col(5)) / n),
            "vel_raw  std %.1f mm/s   vel_filt std %.1f mm/s" % (
                std(col(6)), std(col(7))),
            "cmd_tilt std %.4f deg  range %.4f deg" % (
                std(col(8)), max(col(8)) - min(col(8))),
            "target   changed %d times (%.1f /s), reversals %d (%.1f /s)" % (
                changes, changes / dur if dur else 0,
                reversals(tgt), reversals(tgt) / dur if dur else 0),
            "position reversals %d (%.1f /s)" % (
                reversals(pos), reversals(pos) / dur if dur else 0),
            "snr      mean %.1f  min %.1f" % (
                sum(col(2)) / n, min(col(2))),
        ]

        # a plain-language read on what the numbers imply
        hints = []
        rev_rate = reversals(tgt) / max(dur, 1e-6)
        if rev_rate > 5:
            hints.append("target reverses >5x/s -> motor is being retasked "
                         "constantly; raise step_deadband")
        if std(col(3)) > 3.0 and abs(std(col(6))) > 30:
            hints.append("ball_px and velocity both noisy -> measurement "
                         "noise dominates; check baseline/lighting, raise "
                         "ball_sigma or lower gate_px")
        if std(col(8)) > 0.03 and std(col(3)) < 2.0:
            hints.append("command moves a lot while the ball barely does "
                         "-> gain too high for this rig; lower gain x100")
        if locked < 0.9 * n:
            hints.append("tracker not locked every frame -> detection is "
                         "marginal, fix that before tuning control")
        if hints:
            out.append("")
            out.extend("HINT: " + h for h in hints)
        return out

    # ---------------- output ----------------
    def dump(self, app, path=None):
        st = self.st
        rows = list(self.buf)[-st.diag_dump_rows:]
        lines = []
        lines.append("=== ball&beam diagnostic dump ===")
        lines.append("mode=%s  control_on=%s  driver=%s" % (
            app.ctrl.mode, app.control_on, app.motor.enabled))
        lines.append(
            "gain=%.2f wn=%.2f zeta=%.2f deadband_mm=%.1f d_filter=%.2f "
            "vel_dead_mms=%.1f" % (
                st.gain_scale, st.wn, st.zeta, st.deadband_m * 1000,
                st.d_filter, st.vel_deadband_mps * 1000))
        lines.append(
            "tilt_limit=%.2fdeg tilt_rate=%.0fdps max_jump=%d "
            "step_deadband=%d microstep=%d steps_rev=%d" % (
                st.tilt_limit_deg, st.tilt_rate_dps, st.max_step_jump,
                st.step_deadband, st.microstep, st.steps_per_rev))
        lines.append(
            "roi_span_m=%.3f gate_px=%d ball_sigma=%.1f snr_keep=%.1f "
            "w_dark=%.2f w_tex=%.2f" % (
                st.roi_span_m, st.gate_px, st.ball_sigma_px, st.snr_keep,
                st.w_dark, st.w_tex))
        lines.append("")
        lines.extend(self.summary())
        lines.append("")
        lines.append("  ".join("%9s" % c for c in self.COLUMNS))
        for r in rows:
            lines.append("  ".join([
                "%9.1f" % r[0], "%9s" % r[1], "%9.1f" % r[2],
                "%9.2f" % r[3], "%9.2f" % r[4], "%9.2f" % r[5],
                "%9.1f" % r[6], "%9.1f" % r[7], "%9.4f" % r[8],
                "%9.4f" % r[9], "%9d" % r[10], "%9d" % r[11]]))
        text = "\n".join(lines)

        if path is None:
            path = "diag_%s.txt" % time.strftime("%H%M%S")
        try:
            with open(path, "w") as f:
                f.write(text + "\n")
            saved = path
        except Exception as e:
            saved = "(write failed: %s)" % e
        print(text)
        print("\n[diagnostic written to %s - copy the block above]" % saved)
        return saved
