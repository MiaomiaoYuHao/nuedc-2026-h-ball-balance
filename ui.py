"""Windows, the interactive control panel, overlay drawing, and file I/O.

The control panel is drawn by hand rather than using cv2 trackbars, because
trackbar windows can't scroll: once there are twenty parameters the bottom
ones are simply unreachable. This panel renders into an image sized to the
current window, so it is resizable, scrolls with the mouse wheel, and has
clickable buttons for the steps that used to be keyboard-only.

Sliders write straight into the Settings object, so a value changed by a
keypress shows up on the panel immediately - there is no separate slider
state to keep in sync.
"""

import math
import os

import cv2
import numpy as np

WINDOW_NAME = "ball beam"
BTN_WINDOW = "control panel"
SLIDER_WINDOW = "sliders"

# (label, settings attr, scale, offset, slider max)
#   slider_value = attr * scale + offset
_SECTIONS = [
    ("workflow / response", [
        ("gain x100",      "gain_scale",       100,  0,  300),
        ("wn x100",        "wn",               100,  0, 1000),
        ("zeta x100",      "zeta",             100,  0,  300),
        ("deadband_mm",    "deadband_m",      1000,  0,   30),
        ("d_filter x100",  "d_filter",         100,  0,   99),
        ("vel_dead_mms",   "vel_deadband_mps", 1000, 0,   60),
        ("setpoint x100",  "setpoint_norm",    100, 80,  160),
    ]),
    ("motor limits", [
        ("step_rate_sps",  "step_rate_sps",       1,  0, 6000),
        ("tilt_limit x10", "tilt_limit_deg",    10,  0,   50),
        ("tilt_rate_dps",  "tilt_rate_dps",      1,  0,  300),
        ("max_step_jump",  "max_step_jump",      1,  0,  400),
        ("step_deadband",  "step_deadband",      1,  0,   60),
    ]),
    ("lost-ball recovery", [
        ("lost_frames",    "lost_search_frames", 1,  0,  180),
        ("search_tilt x10", "search_tilt_deg",  10,  0,   50),
        ("flip_frames",    "search_flip_frames", 1,  0,  300),
    ]),
    ("encoder window", [
        ("enc_min",        "enc_abs_min",        1,  0,  360),
        ("enc_max",        "enc_abs_max",        1,  0,  360),
        ("enc_margin",     "enc_abs_margin",     1,  0,   30),
    ]),
    ("vision", [
        ("w_dark x10",     "w_dark",            10,  0,   30),
        ("w_tex x10",      "w_tex",             10,  0,   30),
        ("ball_sigma",     "ball_sigma_px",      1,  0,   40),
        ("snr_acq x10",    "snr_acquire",       10,  0,  300),
        ("snr_keep x10",   "snr_keep",          10,  0,  300),
        ("gate_px",        "gate_px",            1,  0,  300),
        ("margin_L",       "margin_left",        1,  0,  250),
        ("margin_R",       "margin_right",       1,  0,  250),
        ("board_mask",     "use_board_mask",     1,  0,    1),
        ("board_thresh",   "board_thresh",       1,  0,  255),
        ("min_board_%",    "min_board_pct",      1,  0,  100),
        ("binarize",       "use_binarize",       1,  0,    1),
        ("bin_thresh",     "bin_thresh",         1,  0,  255),
    ]),
]

# lower/upper bounds applied after a drag, so a slider at 0 can't break math
_FLOORS = {
    "ball_sigma_px": 2.0, "snr_acquire": 1.0, "snr_keep": 0.5, "gate_px": 8,
    "wn": 0.1, "zeta": 0.05, "max_step_jump": 1, "tilt_limit_deg": 0.1,
    "tilt_rate_dps": 1, "search_tilt_deg": 0.1, "lost_search_frames": 1,
    "step_rate_sps": 100,
}
_CEILS = {"d_filter": 0.99}

# (label, action, column span) - the numbered ones are the setup workflow
_BUTTONS = [
    ("1. SET LEVEL", "set_level", 1),
    ("2. SELECT ROI", "select_roi", 1),
    ("3. BASELINE", "baseline", 1),
    ("4. SAVE", "save", 1),
    ("START / STOP", "toggle_run", 2),
    ("DRIVER on/off", "toggle_driver", 1),
    ("MODE pid/bang", "toggle_mode", 1),
    ("RESET TRACK", "reset_tracker", 1),
    ("DIR TEST", "dirtest", 1),
    ("EXPOSURE", "toggle_ae", 1),
    ("LOG CSV", "toggle_log", 1),
    ("DUMP DIAG", "dump_diag", 2),
]

_BG = (28, 28, 32)
_FG = (225, 225, 225)
_DIM = (150, 150, 155)
_ACCENT = (0, 200, 255)
_OK = (90, 200, 90)
_WARN = (60, 140, 240)


class ButtonPanel:
    """Status line + every clickable action. Short and fixed-height - the
    button list is small enough that it fits without scrolling on any
    screen, which is the whole point of giving it its own window instead of
    sharing one with the (much taller) slider list.
    """
    BTN_H = 34
    PAD = 8

    def __init__(self, settings):
        self.st = settings
        self.pending = []
        self.status = "ready"
        self.status_col = _DIM
        self._rows = []
        h = self._height()
        cv2.namedWindow(BTN_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(BTN_WINDOW, 440, h)
        cv2.setMouseCallback(BTN_WINDOW, self._on_mouse)

    def _height(self):
        rows = 1     # status line
        col = 0
        for _l, _a, span in _BUTTONS:
            if span == 2 or col == 1:
                rows += 1
                col = 0
            else:
                col = 1
        if col == 1:
            rows += 1
        return self.PAD * 2 + 24 + rows * (self.BTN_H + 6)

    def set_status(self, msg, ok=True):
        self.status = msg
        self.status_col = _OK if ok else _WARN

    def poll(self):
        out, self.pending = self.pending, []
        return out

    def _on_mouse(self, event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for y0, y1, action in self._rows:
            if y0 <= y <= y1:
                self.pending.append(action)
                return

    def draw(self, run_state, driver_on, mode, logging_on, ae_on):
        vw, vh = 440, self._height()
        img = np.full((vh, vw, 3), _BG, np.uint8)
        self._rows = []

        cv2.putText(img, self.status[:56], (self.PAD, self.PAD + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.status_col, 1)
        y = self.PAD + 24

        col = 0
        bw_full = vw - 2 * self.PAD
        bw_half = (bw_full - 8) // 2
        for label, action, span in _BUTTONS:
            if span == 2 and col == 1:
                y += self.BTN_H + 6
                col = 0
            bx = self.PAD + (0 if col == 0 else bw_half + 8)
            bw = bw_full if span == 2 else bw_half

            col_bg = (58, 58, 66)
            txt = label
            if action == "toggle_run":
                col_bg = (40, 110, 40) if run_state else (110, 40, 40)
                txt = "STOP (running)" if run_state else "START (paused)"
            elif action == "toggle_driver":
                col_bg = (40, 90, 40) if driver_on else (70, 70, 78)
                txt = "DRIVER on" if driver_on else "DRIVER off"
            elif action == "toggle_mode":
                txt = "MODE: %s" % mode
            elif action == "toggle_log":
                col_bg = (40, 40, 120) if logging_on else (58, 58, 66)
                txt = "LOG: rec" if logging_on else "LOG CSV"
            elif action == "toggle_ae":
                txt = "EXP: auto" if ae_on else "EXP: lock"
            elif action == "dump_diag":
                col_bg = (90, 60, 20)

            cv2.rectangle(img, (bx, y), (bx + bw, y + self.BTN_H), col_bg, -1)
            cv2.rectangle(img, (bx, y), (bx + bw, y + self.BTN_H),
                          (90, 90, 98), 1)
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.46, 1)
            cv2.putText(img, txt, (bx + max(6, (bw - tw) // 2),
                                   y + self.BTN_H // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, _FG, 1)
            self._rows.append((y, y + self.BTN_H, action))

            if span == 2 or col == 1:
                y += self.BTN_H + 6
                col = 0
            else:
                col = 1

        cv2.imshow(BTN_WINDOW, img)


class SliderPanel:
    """Every tunable value, nothing else, all listed at once - no scrolling,
    no up/down bar. The window is sized to the full content height at
    creation, so every slider is visible without any extra interaction. If
    that's taller than the screen, resize/drag the window with the normal
    OS window controls; the content itself never hides anything.
    """
    ROW_H = 24
    PAD = 8

    def __init__(self, settings):
        self.st = settings
        self._rows = []
        self._drag_attr = None
        vw = 430
        vh = self._content_h(vw)
        self._view = (vw, vh)
        cv2.namedWindow(SLIDER_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(SLIDER_WINDOW, vw, vh)
        cv2.setMouseCallback(SLIDER_WINDOW, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONUP:
            self._drag_attr = None
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            for y0, y1, kind, payload in self._rows:
                if y0 <= y <= y1 and kind == "slider":
                    self._drag_attr = payload
                    self._set_from_x(payload, x)
                    return
        if event == cv2.EVENT_MOUSEMOVE and self._drag_attr is not None \
                and (flags & cv2.EVENT_FLAG_LBUTTON):
            self._set_from_x(self._drag_attr, x)

    def _set_from_x(self, spec, x):
        _label, attr, scale, offset, top = spec
        vw = self._view[0]
        x0, x1 = self.PAD + 140, vw - self.PAD
        frac = (x - x0) / float(max(1, x1 - x0))
        raw = max(0, min(top, int(round(frac * top))))
        val = (raw - offset) / float(scale)
        if attr in _FLOORS:
            val = max(_FLOORS[attr], val)
        if attr in _CEILS:
            val = min(_CEILS[attr], val)
        cur = getattr(self.st, attr)
        setattr(self.st, attr, int(round(val)) if isinstance(cur, int) else val)
        if self.st.enc_abs_min > self.st.enc_abs_max:
            self.st.enc_abs_min, self.st.enc_abs_max = \
                self.st.enc_abs_max, self.st.enc_abs_min

    def _content_h(self, vw=None):
        h = self.PAD
        for title, items in _SECTIONS:
            h += 20 + len(items) * self.ROW_H
        return h + self.PAD

    def draw(self):
        vw, vh = self._view
        img = np.full((vh, vw, 3), _BG, np.uint8)
        self._rows = []
        y = self.PAD

        for title, items in _SECTIONS:
            cv2.line(img, (self.PAD, y + 10), (vw - self.PAD, y + 10),
                     (60, 60, 68), 1)
            cv2.putText(img, " %s " % title, (self.PAD + 8, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _ACCENT, 1)
            y += 20
            for spec in items:
                label, attr, scale, offset, top = spec
                val = getattr(self.st, attr)
                raw = max(0, min(top, int(round(val * scale)) + offset))
                cv2.putText(img, label, (self.PAD, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _DIM, 1)
                x0, x1 = self.PAD + 140, vw - self.PAD - 44
                cv2.line(img, (x0, y + 12), (x1, y + 12), (70, 70, 78), 3)
                px = x0 + int((x1 - x0) * raw / float(max(1, top)))
                cv2.line(img, (x0, y + 12), (px, y + 12), _ACCENT, 3)
                cv2.circle(img, (px, y + 12), 5, _FG, -1)
                shown = ("%.2f" % val) if isinstance(val, float) else str(val)
                cv2.putText(img, shown, (x1 + 5, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _FG, 1)
                self._rows.append((y, y + self.ROW_H, "slider", spec))
                y += self.ROW_H

        cv2.imshow(SLIDER_WINDOW, img)


# ---------------- image helpers ----------------

def make_windows(st):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, st.frame_width + st.plot_height,
                     st.frame_height)


def apply_color_mode(st, frame):
    if st.color_mode == 1:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if st.color_mode == 2:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return frame


def select_roi(st, frame_bgr):
    print("drag a box over the INSIDE of the pipe, then press ENTER")
    r = cv2.selectROI(WINDOW_NAME, frame_bgr, showCrosshair=True,
                      fromCenter=False)
    cv2.destroyWindow(WINDOW_NAME)
    make_windows(st)
    x, y, w, h = [int(v) for v in r]
    return (x, y, w, h) if w > 0 and h > 0 else None


def baseline_path(st):
    """Absolute path, so saving doesn't depend on the working directory."""
    if os.path.isabs(st.config_path):
        return st.config_path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, st.config_path)


def save_baseline(st, roi, bm, bs):
    """Write ROI + baseline. Returns (ok, message).

    Writes to a temp file and renames, so an interrupted save can't leave a
    truncated .npz that fails to load next boot. Any exception is reported
    with its text instead of taking down the control loop.
    """
    if roi is None:
        return False, "no ROI yet - press SELECT ROI first"
    if bm is None or bs is None:
        return False, "no baseline yet - remove ball, press BASELINE"
    path = baseline_path(st)
    tmp = path + ".tmp.npz"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(tmp,
                 roi=np.asarray(roi, dtype=np.int32),
                 base_mean=np.ascontiguousarray(bm, dtype=np.float32),
                 base_std=np.ascontiguousarray(bs, dtype=np.float32))
        os.replace(tmp, path)
    except Exception as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False, "save failed: %s" % e
    return True, "saved %s" % os.path.basename(path)


def load_baseline(st):
    path = baseline_path(st)
    if not os.path.exists(path):
        return None, None, None
    try:
        with np.load(path) as d:
            roi = tuple(int(v) for v in d["roi"])
            bm = np.array(d["base_mean"], dtype=np.float32)
            bs = np.array(d["base_std"], dtype=np.float32)
        print("loaded ROI + baseline from %s" % path)
        return roi, bm, bs
    except Exception as e:
        print("baseline load failed (%s) - press SELECT ROI then BASELINE" % e)
        return None, None, None


def render_plot(st, resp, valid, ball_x, gate_lo, gate_hi, thr_level, length):
    """Score profile strip. Vectorised: one polyline, not hundreds of lines."""
    ph = st.plot_height
    canvas = np.zeros((ph, length, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)
    n = resp.size
    if n < 2:
        return canvas

    idx = (np.arange(length) * n // length).clip(0, n - 1)
    vals = resp[idx]
    lo_v = min(-0.02, float(vals.min()) * 1.2)
    hi_v = max(0.05, float(vals.max()) * 1.25)
    span = max(1e-6, hi_v - lo_v)

    def to_y(v):
        v = np.clip(v, lo_v, hi_v)
        return ph - 1 - (v - lo_v) / span * (ph - 1)

    if valid is not None and valid.size == n:
        canvas[:, ~valid[idx]] = (55, 40, 40)

    if gate_lo is not None:
        a = int(gate_lo * length / n)
        b = int(gate_hi * length / n)
        sub = canvas[:, max(0, a):max(0, b)]
        if sub.size:
            sub[:] = (45, 60, 45)

    cv2.line(canvas, (0, int(to_y(0.0))), (length, int(to_y(0.0))),
             (70, 70, 70), 1)
    if thr_level is not None:
        yt = int(to_y(thr_level))
        cv2.line(canvas, (0, yt), (length, yt), (0, 120, 190), 1)

    ys = to_y(vals).astype(np.int32)
    pts = np.stack([np.arange(length, dtype=np.int32), ys], axis=1)
    cv2.polylines(canvas, [pts], False, (120, 220, 255), 1)

    if ball_x is not None:
        px = int(ball_x * length / n)
        cv2.line(canvas, (px, 0), (px, ph), (0, 255, 0), 1)
    return canvas
