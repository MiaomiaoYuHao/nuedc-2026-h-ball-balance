"""Main loop: capture -> detect -> control -> draw -> keys."""

import math
import time

import cv2
import numpy as np

from . import ui, vision
from .camera import CameraSource
from .diagrec import DiagRecorder
from .controller import BalanceController
from .encoder import AngleEncoder
from .kinematics import (encoder_deg_in_range, shaft_deg_to_tilt,
                         step_limit, clamp_tilt_to_encoder_range)
from .motor import Motor

HELP = """
keys
    space  start/stop closed loop (also enables/disables the driver)
    p      control mode: pid <-> bangbang
    c      set current position as level / zero (motor AND encoder)
    a / d  nudge zero by 5 steps
    [ / ]  move the setpoint left / right
    o      driver enable on/off (off = beam free to move by hand)
    r      reselect pipe ROI
    b      capture empty-groove baseline (remove the ball first)
    v      save ROI + baseline to disk
    t      force the tracker to re-acquire
    j      motor direction self-test (ball must be free to roll)
    e      toggle auto exposure <-> frozen exposure
    E / w  brighten / darken 25% (only while exposure is frozen)
    m      view: normal / binary
    x      display colour mode
    g      toggle GUI drawing (fps boost)
    l      start/stop CSV logging
    z      dump a diagnostic block (paste this when asking for help)
    q      quit

The 'control panel' window has clickable buttons for the numbered setup
steps and the run/stop toggle, and scrolls with the mouse wheel.
"""


class App:
    def __init__(self, settings, gui=True, use_motor=True):
        self.st = settings
        self.gui = gui
        self.motor = Motor(settings, use_motor=use_motor)
        self.enc = AngleEncoder(settings)
        self.ctrl = BalanceController(settings, self.motor, self.enc)
        self.cam = None

        self.roi = None
        self.base_mean = None
        self.base_std = None
        self.tracker = None
        self.kernel = None
        self.kernel_half = 0
        self.last_sigma = None

        self.control_on = False   # starts paused; hand-level, then space
        self.logf = None
        self.dirtest = None
        self.btn_panel = None
        self.slider_panel = None
        self.diag = DiagRecorder(settings)
        self.lost_frames = 0

        # per-frame state kept for the HUD
        self.ball_px = None
        self.ball_norm = None
        self.err_m = 0.0
        self.vel_mps = 0.0
        self.meas_tilt = 0.0
        self.resp = np.zeros(1, dtype=np.float32)
        self.valid = np.ones(1, dtype=bool)
        self.proc = None
        self.n_valid = 0
        self.roi_level = 0.0
        self.ae_gain = 1.0
        self.mad = 0.0
        self.auto_board_thresh = 0.0
        self.mask_failed = False
        self.gate_lo = self.gate_hi = None
        self.rect = (0, 0, 0, 0)
        self.loststep_count = 0
        self.resyncs = 0
        self.fps = 0.0
        self.frame_count = 0

    # ---------------- setup ----------------
    def start(self):
        st = self.st
        cv2.setNumThreads(2)
        print("KP=%.3f rad/m   KD=%.3f rad/(m/s)   (wn=%.1f zeta=%.1f)"
              % (st.kp, st.kd, st.wn, st.zeta))
        print("tilt limit %.2f deg -> +/-%d steps   (mech max %.2f deg)"
              % (st.tilt_limit_deg, step_limit(st),
                 math.degrees(st.max_mech_tilt)))
        if not self.motor.use_motor:
            print("MOTOR DISABLED (no lgpio or --no-motor): simulating")

        self.motor.setup()
        self.motor.start()

        self.cam = CameraSource(st)
        self.roi, self.base_mean, self.base_std = ui.load_baseline(st)

        print("encoder absolute safety window: [%.0f, %.0f] deg (margin %.0f)"
              % (st.enc_abs_min, st.enc_abs_max, st.enc_abs_margin))
        if self.enc.alive() and not encoder_deg_in_range(st, self.enc.raw_deg):
            print("WARNING: encoder reads %.1f deg, already outside that "
                  "window. Hand-level the beam and press 'c' before running."
                  % self.enc.raw_deg)

        if self.gui:
            self._open_gui()
        self._rebuild_kernel()
        print(HELP)

    def _rebuild_kernel(self):
        self.kernel = vision.make_kernel(self.st, self.st.ball_sigma_px)
        self.kernel_half = self.kernel.size // 2
        self.last_sigma = self.st.ball_sigma_px

    def _open_gui(self):
        st = self.st
        ui.make_windows(st)
        try:
            self.btn_panel = ui.ButtonPanel(st)
            self.slider_panel = ui.SliderPanel(st)
        except Exception as e:
            print("WARNING: a panel window failed to open (%s) - falling "
                  "back to keyboard-only control. Update opencv if this "
                  "persists: pip3 install --upgrade opencv-python" % e)
            self.btn_panel = None
            self.slider_panel = None
        # Windows default to the same top-left corner under some window
        # managers (common over VNC without a WM running), so extra windows
        # can end up hidden exactly behind the first one. Force all three
        # apart so every one is visible without dragging anything.
        try:
            vid_x = 40
            vid_w = st.frame_width + st.plot_height
            cv2.moveWindow(ui.WINDOW_NAME, vid_x, 40)
            if self.btn_panel is not None:
                cv2.moveWindow(ui.BTN_WINDOW, vid_x + vid_w + 20, 40)
            if self.slider_panel is not None:
                btn_h = self.btn_panel._height() if self.btn_panel else 300
                cv2.moveWindow(ui.SLIDER_WINDOW, vid_x + vid_w + 20,
                               40 + btn_h + 50)
        except Exception as e:
            print("window positioning skipped (%s) - drag the '%s' and "
                  "'%s' windows into view if you don't see them"
                  % (e, ui.BTN_WINDOW, ui.SLIDER_WINDOW))

    def close(self):
        if self.logf is not None:
            self.logf.close()
        if self.cam is not None:
            self.cam.close()
        self.enc.close()
        cv2.destroyAllWindows()
        self.motor.teardown()
        print("motor disabled, GPIO cleaned up")

    # ---------------- vision ----------------
    def process_frame(self, frame_bgr, dt):
        st = self.st
        self.ball_px = None
        self.ball_norm = None
        self.mask_failed = False
        self.auto_board_thresh = 0.0
        self.resp = np.zeros(1, dtype=np.float32)
        self.valid = np.ones(1, dtype=bool)
        self.proc = None
        self.n_valid = 0
        self.roi_level = 0.0
        self.ae_gain = 1.0
        self.gate_lo = self.gate_hi = None
        self.mad = 0.0
        if self.roi is None:
            return

        x, y, w, h = self.roi
        x = max(0, min(st.frame_width - 1, x))
        y = max(0, min(st.frame_height - 1, y))
        w = max(8, min(st.frame_width - x, w))
        h = max(4, min(st.frame_height - y, h))
        self.rect = (x, y, w, h)

        sub = frame_bgr[y:y + h, x:x + w]
        trim = int(h * st.row_trim_pct / 100)
        if h - 2 * trim >= 3:
            sub = sub[trim:h - trim]
        roi_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        self.roi_level = float(roi_gray.mean())

        self.proc = vision.preprocess(roi_gray, st.use_binarize, st.bin_thresh)
        mp, sp = vision.compute_profile(self.proc)

        self.valid, self.auto_board_thresh, self.mask_failed = \
            vision.build_valid_mask(st, roi_gray, w, self.kernel_half)
        self.n_valid = int(self.valid.sum())

        if self.tracker is None or self.tracker.w != w:
            self.tracker = vision.BallTracker(st, w / st.roi_span_m, w)

        if self.base_mean is None or self.base_mean.shape != mp.shape:
            return

        self.ae_gain = vision.ae_scale(st, mp, self.base_mean, self.valid)
        sc = vision.raw_score(st, mp, sp, self.base_mean, self.base_std,
                              self.ae_gain)
        self.resp = vision.matched_response(sc, self.kernel)
        self.gate_lo, self.gate_hi = self.tracker.gate_bounds()

        # predict with the tilt the beam ACTUALLY has, not the one we asked
        # for - they differ during a slew
        tilt_in = self.meas_tilt if self.enc.alive() else self.ctrl.cmd_tilt
        self.tracker.update(self.resp, self.valid, self.kernel_half, dt,
                            tilt_in)
        self.mad = self.tracker.ref

        px, vel_px = self.tracker.estimate()
        if px is not None:
            self.ball_px = float(np.clip(px, 0, w - 1))
            self.ball_norm = (self.ball_px - w * 0.5) / (w * 0.5)
            self.vel_mps = vel_px / self.tracker.px_per_m

        self._adapt_baseline(mp, sp, w)

    def _adapt_baseline(self, mp, sp, w):
        """Slowly re-learn the empty groove away from the ball, so a gradual
        lighting change doesn't require pressing 'b' again."""
        st = self.st
        if st.base_adapt <= 0 or self.tracker.state != "lock":
            return
        keep = np.ones(w, dtype=bool)
        if self.ball_px is not None:
            a = max(0, int(self.ball_px) - st.base_keepout_px)
            b = min(w, int(self.ball_px) + st.base_keepout_px)
            keep[a:b] = False
        keep &= self.valid
        if not keep.any():
            return
        g = self.ae_gain
        self.base_mean[keep] += st.base_adapt * (mp[keep] / g
                                                 - self.base_mean[keep])
        self.base_std[keep] += st.base_adapt * (sp[keep] / g
                                                - self.base_std[keep])

    # ---------------- control ----------------
    def run_control(self, dt):
        st = self.st
        if self.ball_norm is None:
            self.lost_frames += 1
            self.ctrl.vel_filt = 0.0
            self.ctrl.sat_count = 0
            if self.control_on and self.dirtest is None:
                if self.lost_frames > st.lost_search_frames:
                    # memory recovery: roll it back from whichever end it
                    # disappeared at, instead of parking level
                    self.ctrl.search(self.lost_frames, dt)
                elif st.return_to_level_on_lost:
                    self.ctrl.go_level()
            return

        self.lost_frames = 0
        self.ctrl.note_seen(self.ball_norm, self.vel_mps)
        meas_m = self.ball_norm * (st.roi_span_m * 0.5)
        set_m = st.setpoint_norm * (st.roi_span_m * 0.5)
        self.err_m = set_m - meas_m
        if self.control_on and self.dirtest is None:
            self.ctrl.update(self.err_m, self.vel_mps, dt)

    def run_dirtest(self, now):
        """Tilt one way, see which way the ball rolls, report the sign."""
        st = self.st
        phase, t0, p0 = self.dirtest
        if phase == 0:
            safe = clamp_tilt_to_encoder_range(st, st.tilt_limit, self.enc)
            self.ctrl.cmd_tilt, _ = self.motor.command_tilt(safe)
            self.ctrl.prev_cmd_tilt = self.ctrl.cmd_tilt
            self.dirtest = (1, now, self.ball_norm)
        elif phase == 1 and now - t0 > 1.5:
            self.ctrl.go_level()
            if p0 is None or self.ball_norm is None:
                print("dir test: ball not seen, cannot judge")
            else:
                d = self.ball_norm - p0
                print("dir test: +tilt moved the ball by %+.3f norm" % d)
                if abs(d) < 0.02:
                    print("  -> ball did not move: check the linkage, raise "
                          "tilt_limit_deg, or the ball is stuck")
                elif d > 0:
                    print("  -> +tilt pushes the ball RIGHT. For negative "
                          "feedback set motor_dir_sign = %+d"
                          % (-st.motor_dir_sign))
                else:
                    print("  -> +tilt pushes the ball LEFT. Current "
                          "motor_dir_sign = %+d is correct" % st.motor_dir_sign)
            self.dirtest = None

    def check_encoder(self):
        """Inner loop: compare commanded crank angle with the measured one.

        The stepper is open loop, so a lost step is a silent, permanent angle
        error that the outer loop would spend the rest of the run fighting.
        The encoder closes that gap.
        """
        st = self.st
        cur, tgt, zoff = self.motor.snapshot()
        if not self.enc.alive():
            self.meas_tilt = self.ctrl.cmd_tilt
            return cur, tgt, zoff

        meas_shaft = self.enc.shaft_deg()
        self.meas_tilt = shaft_deg_to_tilt(st, meas_shaft)
        cmd_shaft = (cur - zoff) * 360.0 / st.steps_per_rev
        ang_err = cmd_shaft - meas_shaft

        if abs(ang_err) > st.loststep_deg:
            self.loststep_count += 1
        else:
            self.loststep_count = 0

        if self.loststep_count > st.loststep_frames:
            self.loststep_count = 0
            self.resyncs += 1
            if st.enc_resync:
                self.motor.resync_to(
                    int(round(meas_shaft * st.steps_per_rev / 360.0)))
                print("lost steps: %+.2f deg mismatch, step counter resynced "
                      "from the encoder (#%d). Raise the driver current or "
                      "lower tilt_rate_dps." % (ang_err, self.resyncs))
        return cur, tgt, zoff

    # ---------------- drawing ----------------
    def draw(self, frame_bgr, cur, zoff):
        st = self.st
        if st.view_mode == 1:
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            disp = cv2.cvtColor(
                cv2.threshold(g, st.bin_thresh, 255, cv2.THRESH_BINARY)[1],
                cv2.COLOR_GRAY2BGR)
        else:
            disp = frame_bgr.copy()

        x, y, w, h = self.rect
        if self.roi is not None:
            cv2.rectangle(disp, (x, y), (x + w, y + h), (200, 200, 200), 1)

            bad = np.where(~self.valid)[0]
            if bad.size:
                overlay = disp.copy()
                for run in np.split(bad, np.where(np.diff(bad) != 1)[0] + 1):
                    if run.size:
                        cv2.rectangle(overlay, (x + int(run[0]), y),
                                      (x + int(run[-1]) + 1, y + h),
                                      (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.28, disp, 0.72, 0, disp)

            if self.gate_lo is not None:
                cv2.rectangle(disp, (x + self.gate_lo, y),
                              (x + self.gate_hi, y + h), (0, 180, 0), 1)

            sp_px = x + int(w * 0.5 + st.setpoint_norm * w * 0.5)
            cv2.line(disp, (sp_px, y - 8), (sp_px, y + h + 8), (0, 220, 220), 2)
            if self.ball_px is not None:
                bx = x + int(round(self.ball_px))
                cv2.line(disp, (bx, y - 8), (bx, y + h + 8), (0, 255, 0), 2)
                cv2.line(disp, (bx, y + h // 2), (sp_px, y + h // 2),
                         (0, 0, 255), 2)

        # motor position bar
        by = st.frame_height - 26
        cv2.line(disp, (20, by), (st.frame_width - 20, by), (110, 110, 110), 2)
        midx = st.frame_width // 2
        cv2.line(disp, (midx, by - 7), (midx, by + 7), (200, 200, 200), 1)
        span = (st.frame_width - 40) * 0.5
        frac = max(-1.0, min(1.0, (cur - zoff) / float(max(1, step_limit(st)))))
        cv2.circle(disp, (int(midx + frac * span), by), 6, (0, 165, 255), -1)

        state = self.tracker.state if self.tracker else "none"
        snr = self.tracker.snr if self.tracker else 0.0
        col = (0, 255, 0) if state == "lock" else (0, 165, 255)
        in_range = (not self.enc.alive()
                    or encoder_deg_in_range(st, self.enc.raw_deg))
        info = [
            "%s  [%s]  %.0f fps%s" % (
                "RUN" if self.control_on else "PAUSED", self.ctrl.mode,
                self.fps,
                "  LOW FPS!" if 0 < self.fps < st.min_healthy_fps else ""),
            "track %-8s snr %5.1f  noise %.4f" % (state, snr, self.mad),
            "tilt %+.2f deg   steps %+d" % (math.degrees(self.ctrl.cmd_tilt),
                                            cur - zoff),
            "err %+.1f mm   vel %+.0f mm/s" % (self.err_m * 1000,
                                               self.vel_mps * 1000),
            "valid %d/%d  wd%.1f wt%.1f sig%.0f" % (
                self.n_valid, w, st.w_dark, st.w_tex, st.ball_sigma_px),
            "enc %s %+.2f deg  cmd %+.2f deg  resync %d" % (
                "ok" if self.enc.alive() else "--",
                math.degrees(self.meas_tilt),
                math.degrees(self.ctrl.cmd_tilt), self.resyncs),
            "abs %.1f deg [%d,%d]%s" % (
                self.enc.raw_deg if self.enc.alive() else -1.0,
                st.enc_abs_min, st.enc_abs_max,
                "" if in_range else "  OUT OF RANGE"),
            "lum %.0f  ae %s x%.2f  exp %dus g%.1f" % (
                self.roi_level, "auto" if self.cam.ae_on else "LOCK",
                self.ae_gain, self.cam.exposure_us, self.cam.gain),
        ]
        for i, s in enumerate(info):
            cv2.putText(disp, s, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, col if i == 1 else (0, 255, 255), 1)

        warn = []
        if self.base_mean is None:
            warn.append(("no baseline: press r then b", (0, 165, 255)))
        if self.mask_failed:
            warn.append(("board_mask killed every column - ignored. lower "
                         "board_thresh or set it to 0 (auto)", (0, 0, 255)))
        if 0 < self.roi_level < 45:
            warn.append(("ROI TOO DARK (lum %.0f): press e" % self.roi_level,
                         (0, 0, 255)))
        if self.ctrl.sat_count > st.stuck_frames:
            warn.append(("SATURATED + NOT MOVING: press j, raise "
                         "tilt_limit, or re-level with c", (0, 0, 255)))
        if self.dirtest is not None:
            warn.append(("DIRECTION TEST RUNNING", (255, 0, 255)))
        if self.ctrl.searching:
            warn.append(("BALL LOST %d frames - SEARCHING (tilt %+.2f deg)"
                         % (self.lost_frames, math.degrees(self.ctrl.cmd_tilt)),
                         (255, 200, 0)))
        for i, (msg, c) in enumerate(warn):
            cv2.putText(disp, msg, (10, st.frame_height - 44 - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)
        if self.logf is not None:
            cv2.putText(disp, "REC", (st.frame_width - 60, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        plot = ui.render_plot(st, self.resp, self.valid, self.ball_px,
                              self.gate_lo, self.gate_hi,
                              self.mad * st.snr_keep, st.frame_height)
        disp = np.hstack([disp, cv2.rotate(plot, cv2.ROTATE_90_CLOCKWISE)])
        cv2.imshow(ui.WINDOW_NAME, disp)

    def print_status(self, cur, zoff):
        st = self.st
        state = self.tracker.state if self.tracker else "none"
        snr = self.tracker.snr if self.tracker else 0.0
        if self.base_mean is None:
            print("no baseline - press r then b")
        elif self.ball_norm is not None:
            print("ball %+.4f  snr %5.1f  err %+.1fmm  vel %+.0fmm/s  "
                  "tilt %+.2fdeg  steps %+d  %-7s %.0ffps%s%s"
                  % (self.ball_norm, snr, self.err_m * 1000,
                     self.vel_mps * 1000, math.degrees(self.ctrl.cmd_tilt),
                     cur - zoff, state, self.fps,
                     "" if self.control_on else " [PAUSED]",
                     "  [SAT/STUCK]"
                     if self.ctrl.sat_count > st.stuck_frames else ""))
        else:
            print("no ball [%s]  snr %.1f  noise %.4f  valid %d/%d  %.0ffps"
                  % (state, snr, self.mad, self.n_valid, self.rect[2],
                     self.fps))

    def log_row(self, now, cur, tgt):
        if self.logf is None:
            return
        self.logf.write("%.4f,%s,%.5f,%.5f,%.5f,%.5f,%d,%d,%.2f,%s\n" % (
            now,
            ("%.5f" % self.ball_norm) if self.ball_norm is not None else "",
            self.err_m, self.vel_mps, self.ctrl.cmd_tilt, self.meas_tilt,
            tgt, cur,
            self.tracker.snr if self.tracker else 0.0,
            self.tracker.state if self.tracker else "none"))

    # ---------------- actions (shared by keys and panel buttons) ----------
    def _say(self, msg, ok=True):
        print(msg)
        if self.btn_panel is not None:
            self.btn_panel.set_status(msg, ok)

    def do_action(self, action, frame_bgr=None, now=None):
        """Every user command routes through here, so a panel button and its
        keyboard shortcut can never drift apart."""
        st = self.st
        if action == "quit":
            return False

        elif action == "toggle_run":
            self.control_on = not self.control_on
            self.motor.enable(self.control_on)
            # sync so the rate limiter doesn't jump from a stale value left
            # over from before the beam was hand-moved
            self.ctrl.reset(self.meas_tilt if self.enc.alive() else 0.0)
            self.lost_frames = 0
            self._say("control %s%s" % (
                "RESUMED" if self.control_on else "PAUSED",
                " (driver enabled)" if self.control_on
                else " (driver off - beam free)"))

        elif action == "toggle_mode":
            self._say("control mode -> %s" % self.ctrl.toggle_mode())

        elif action == "set_level":
            self.motor.set_zero_here()
            self.enc.set_zero()
            self.ctrl.reset(0.0)
            self.loststep_count = 0
            self._say("level set at current position")

        elif action == "select_roi":
            if frame_bgr is None:
                return True
            nr = ui.select_roi(st, frame_bgr)
            if nr is not None:
                self.roi = nr
                self.base_mean = self.base_std = None
                self.tracker = None
                self._say("ROI set - now remove the ball and press BASELINE")

        elif action == "baseline":
            if self.proc is None:
                self._say("no ROI yet - press SELECT ROI first", ok=False)
            else:
                self.base_mean, self.base_std = vision.compute_profile(self.proc)
                if self.tracker:
                    self.tracker.reset()
                self._say("baseline captured - now press SAVE")

        elif action == "save":
            ok, msg = ui.save_baseline(st, self.roi, self.base_mean,
                                       self.base_std)
            self._say(msg, ok)

        elif action == "toggle_driver":
            self.motor.enable(not self.motor.enabled)
            self._say("driver %s" % ("ENABLED" if self.motor.enabled
                                     else "OFF (beam free)"))

        elif action == "reset_tracker":
            if self.tracker:
                self.tracker.reset()
            self._say("tracker reset - re-acquiring")

        elif action == "dirtest":
            if self.dirtest is None:
                self.dirtest = (0, now or time.time(), self.ball_norm)
                self._say("direction test: tilting to +limit for 1.5 s")

        elif action == "toggle_ae":
            if self.cam.ae_on:
                self.cam.lock_exposure()
                self._say("exposure frozen - baseline is stale, redo BASELINE")
            else:
                self.cam.resume_auto()
                self._say("auto exposure resumed")
            if self.tracker:
                self.tracker.reset()

        elif action == "exp_up":
            self.cam.bump_exposure(1.25)
        elif action == "exp_down":
            self.cam.bump_exposure(0.8)

        elif action == "setpoint_left":
            st.setpoint_norm = max(-0.8, st.setpoint_norm - st.setpoint_step)
            self._say("setpoint %+.2f" % st.setpoint_norm)
        elif action == "setpoint_right":
            st.setpoint_norm = min(0.8, st.setpoint_norm + st.setpoint_step)
            self._say("setpoint %+.2f" % st.setpoint_norm)

        elif action == "trim_left":
            self.motor.trim_zero(-5)
        elif action == "trim_right":
            self.motor.trim_zero(+5)

        elif action == "view_mode":
            st.view_mode = (st.view_mode + 1) % 2
        elif action == "color_mode":
            st.color_mode = (st.color_mode + 1) % 3

        elif action == "toggle_gui":
            self.gui = not self.gui
            if not self.gui:
                cv2.destroyAllWindows()
                self.btn_panel = None
                self.slider_panel = None
            else:
                self._open_gui()

        elif action == "dump_diag":
            path = self.diag.dump(self)
            self._say("diagnostic written to %s" % path)

        elif action == "toggle_log":
            if self.logf is None:
                fn = "run_%s.csv" % time.strftime("%H%M%S")
                self.logf = open(fn, "w")
                self.logf.write("t,ball_norm,err_m,vel_mps,tilt_cmd_rad,"
                                "tilt_meas_rad,target_steps,pos_steps,snr,"
                                "state\n")
                self._say("logging to %s" % fn)
            else:
                self.logf.close()
                self.logf = None
                self._say("logging stopped")
        return True

    KEYMAP = {
        'q': "quit", ' ': "toggle_run", 'p': "toggle_mode",
        'c': "set_level", 'r': "select_roi", 'b': "baseline", 'v': "save",
        'o': "toggle_driver", 't': "reset_tracker", 'j': "dirtest",
        'e': "toggle_ae", 'E': "exp_up", 'w': "exp_down",
        '[': "setpoint_left", ']': "setpoint_right",
        'a': "trim_left", 'd': "trim_right",
        'm': "view_mode", 'x': "color_mode", 'g': "toggle_gui",
        'l': "toggle_log", 'z': "dump_diag",
    }

    def handle_key(self, k, frame_bgr, now):
        action = self.KEYMAP.get(chr(k)) if 0 < k < 256 else None
        if action is None:
            return True
        return self.do_action(action, frame_bgr, now)

    # ---------------- loop ----------------
    def run(self):
        st = self.st
        prev_t = time.time()
        t_fps = time.time()
        seq = 0
        try:
            while True:
                if abs(st.ball_sigma_px - self.last_sigma) > 0.5:
                    self._rebuild_kernel()

                frame, seq = self.cam.read(seq)
                if frame is None:
                    continue
                frame_bgr = ui.apply_color_mode(st, frame)

                now = time.time()
                dt = min(max(now - prev_t, 1e-3), 0.2)
                prev_t = now

                self.process_frame(frame_bgr, dt)
                self.run_control(dt)
                if self.dirtest is not None:
                    self.run_dirtest(now)
                cur, tgt, zoff = self.check_encoder()

                self.diag.record(now, self)
                self.log_row(now, cur, tgt)
                if self.frame_count % st.print_every_n == 0:
                    self.print_status(cur, zoff)
                if self.gui and self.frame_count % st.draw_every == 0:
                    self.draw(frame_bgr, cur, zoff)
                    if self.btn_panel is not None:
                        self.btn_panel.draw(self.control_on, self.motor.enabled,
                                            self.ctrl.mode, self.logf is not None,
                                            self.cam.ae_on)
                    if self.slider_panel is not None:
                        self.slider_panel.draw()

                stop = False
                if self.btn_panel is not None:
                    for action in self.btn_panel.poll():
                        if not self.do_action(action, frame_bgr, now):
                            stop = True
                if stop:
                    break

                k = (cv2.waitKey(1) & 0xFF) if self.gui else 255
                if k != 255 and not self.handle_key(k, frame_bgr, now):
                    break

                self.frame_count += 1
                if self.frame_count % 60 == 0:
                    self.cam.read_ae()
                if self.frame_count % 30 == 0:
                    self.fps = 30.0 / max(1e-6, time.time() - t_fps)
                    t_fps = time.time()
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            self.close()
