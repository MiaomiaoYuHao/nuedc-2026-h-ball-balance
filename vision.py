"""Ball detection: matched filter -> SNR gate -> Kalman tracker.

Pipeline per frame:
  1. Collapse the ROI to two 1-D column profiles (mean and std).
  2. Score each column against the stored empty-groove baseline, rescaled
     to the current brightness (ae_scale) so auto exposure is harmless.
  3. Correlate with a ZERO-MEAN, ball-shaped kernel. Zero mean cancels any
     uniform or slowly-varying illumination, so shadows and AE breathing
     can't create a fake peak.
  4. Estimate noise robustly (MAD) and work in SNR, which is dimensionless
     and self-calibrating - the thresholds stay valid when lighting changes.
  5. Feed the measurement to a Kalman filter driven by the measured tilt,
     so the search gate follows a PREDICTED position and dropouts coast on
     physics instead of losing the ball.
"""

import math

import cv2
import numpy as np

from .config import K_PLANT


def make_kernel(st, sigma):
    """Zero-mean, unit-energy, ball-shaped matched filter."""
    r = max(2, int(st.kernel_span * sigma))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xs / float(sigma)) ** 2)
    k -= k.mean()
    n = float(np.sqrt((k * k).sum()))
    if n > 1e-9:
        k /= n
    return k


def smooth1d(arr, win):
    win = int(win)
    if win < 2:
        return arr
    if win % 2 == 0:
        win += 1
    return np.convolve(arr, np.ones(win, dtype=np.float32) / win, mode="same")


def preprocess(roi_gray, use_bin, bin_thresh):
    if use_bin:
        _, b = cv2.threshold(roi_gray, bin_thresh, 255, cv2.THRESH_BINARY)
        return b
    return roi_gray


def board_mask_columns(roi_gray, board_thresh, min_board_pct):
    return (roi_gray >= board_thresh).mean(axis=0) * 100.0 >= min_board_pct


def compute_profile(img):
    f = img.astype(np.float32)
    return f.mean(axis=0), f.std(axis=0)


def ae_scale(st, mean_prof, base_mean, valid):
    """Robust brightness ratio between this frame and the stored baseline.

    A median over the valid columns is dominated by empty groove, not by the
    ball (the ball is a few percent of the columns), so this tracks the ISP
    gain rather than the target. Clamped so a bad frame can't blow up.
    """
    if not st.ae_compensate or not valid.any():
        return 1.0
    b = float(np.median(base_mean[valid]))
    m = float(np.median(mean_prof[valid]))
    if b < 1.0 or m < 1.0:
        return 1.0
    return float(min(4.0, max(0.25, m / b)))


def raw_score(st, mean_prof, std_prof, base_mean, base_std, scale=1.0):
    dark = (base_mean * scale - mean_prof) / (255.0 * scale)
    texture = (std_prof - base_std * scale) / (255.0 * scale)
    return smooth1d(st.w_dark * dark + st.w_tex * texture, st.smooth_win)


def matched_response(score, kernel):
    return np.convolve(score, kernel[::-1], mode="same").astype(np.float32)


def robust_noise(resp, valid):
    v = resp[valid]
    if v.size < 8:
        return 0.0, 1e-6
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, 1e-6)


def subpixel(resp, valid, pk, half):
    lo = max(0, pk - half)
    hi = min(resp.size, pk + half + 1)
    seg = resp[lo:hi].copy()
    seg[~valid[lo:hi]] = 0.0
    seg[seg < 0] = 0.0
    tot = float(seg.sum())
    if tot <= 1e-9:
        return float(pk)
    idx = np.arange(lo, hi, dtype=np.float32)
    return float((seg * idx).sum() / tot)


def build_valid_mask(st, roi_gray, w, kernel_half):
    """Which columns may hold the ball. Returns (valid, auto_thresh, failed).

    The board mask is never allowed to delete the whole pipe: if it does,
    it's retried with a threshold derived from the image, and if that also
    fails the mask is ignored for this frame rather than going blind.
    """
    valid = np.ones(w, dtype=bool)
    ml = min(st.margin_left, max(0, w - 4))
    mr = min(st.margin_right, max(0, w - ml - 4))
    if ml > 0:
        valid[:ml] = False
    if mr > 0:
        valid[w - mr:] = False

    auto_thresh = 0.0
    failed = False
    if st.use_board_mask:
        bt = st.board_thresh
        if bt <= 0:                        # 0 on the slider = auto
            bt = float(np.percentile(roi_gray, 55))
        m = board_mask_columns(roi_gray, bt, st.min_board_pct)
        if int((valid & m).sum()) < 20:
            bt = float(np.percentile(roi_gray, 40))
            m = board_mask_columns(roi_gray, bt, st.min_board_pct)
            if int((valid & m).sum()) < 20:
                m = np.ones(w, dtype=bool)
                failed = True
        valid &= m
        auto_thresh = bt

    # the matched filter needs room; kill the convolution edges
    valid[:kernel_half] = False
    valid[w - kernel_half:] = False
    return valid, auto_thresh, failed


class BallTracker:
    """Matched filter + SNR gate + constant-acceleration Kalman filter.

    State is kept in PIXELS inside the ROI. The measured tilt is fed in as a
    control input, so the prediction is physical rather than a straight
    linear extrapolation.
    """

    def __init__(self, settings, px_per_m, width):
        self.st = settings
        self.px_per_m = px_per_m
        self.w = int(width)
        self.p = None            # px
        self.v = 0.0             # px/s
        self.P = np.eye(2, dtype=np.float64) * 1e3
        self.misses = 0
        self.state = "acquire"
        self.snr = 0.0
        self.ref = 0.0

    def reset(self):
        self.p = None
        self.v = 0.0
        self.P = np.eye(2, dtype=np.float64) * 1e3
        self.misses = 0
        self.state = "acquire"
        self.snr = 0.0

    # ---------- Kalman ----------
    def predict(self, dt, tilt_rad):
        if self.p is None:
            return
        a = K_PLANT * math.sin(tilt_rad) * self.px_per_m   # px/s^2
        self.p += self.v * dt + 0.5 * a * dt * dt
        self.v += a * dt
        # a coasting estimate must not leave the pipe
        if self.p < 0.0:
            self.p, self.v = 0.0, max(0.0, self.v)
        elif self.p > self.w - 1:
            self.p, self.v = float(self.w - 1), min(0.0, self.v)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        sa = 3.0 * self.px_per_m          # unmodelled accel m/s^2 -> px/s^2
        q = np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                      [dt ** 3 / 2.0, dt ** 2]]) * (sa * sa)
        self.P = F.dot(self.P).dot(F.T) + q

    def correct(self, z, r_px):
        H = np.array([[1.0, 0.0]])
        S = float(self.P[0, 0] + r_px * r_px)
        K = self.P.dot(H.T) / S
        y = z - self.p
        self.p += float(K[0, 0] * y)
        self.v += float(K[1, 0] * y)
        self.P = (np.eye(2) - K.dot(H)).dot(self.P)

    # ---------- measurement ----------
    def measure(self, resp, valid, kernel_half):
        st = self.st
        med, mad = robust_noise(resp, valid)
        if not valid.any():
            return None, 0.0, med, mad

        acquiring = (self.p is None) or (self.misses > st.max_misses)
        if acquiring:
            lo, hi = 0, resp.size
            g = resp.size
        else:
            g = int(st.gate_px * min(st.gate_max_mult,
                                     1.0 + st.gate_grow * self.misses))
            c = int(round(self.p))
            lo = max(0, c - g)
            hi = min(resp.size, c + g + 1)

        seg = resp[lo:hi].copy()
        segv = valid[lo:hi]
        if not segv.any():
            return None, 0.0, med, mad
        seg[~segv] = -1e9
        pk = lo + int(np.argmax(seg))
        snr = (float(resp[pk]) - med) / mad

        # a wider search window sees more noise samples, so it must demand a
        # higher SNR - otherwise a coasting tracker latches onto noise
        need = st.snr_acquire if acquiring else st.snr_keep
        need *= 1.0 + 0.35 * math.log10(
            max(1.0, 2.0 * g / max(8.0, st.gate_px)))
        if snr < need:
            return None, snr, med, mad

        # runner-up test, only when acquiring (inside the gate there's one ball)
        if acquiring:
            masked = resp.copy()
            masked[~valid] = -1e9
            a = max(0, pk - 3 * kernel_half)
            b = min(resp.size, pk + 3 * kernel_half + 1)
            masked[a:b] = -1e9
            if masked.max() > -1e8:
                snr2 = (float(masked.max()) - med) / mad
                if snr2 > st.runnerup_ratio * snr:
                    return None, snr, med, mad

        return subpixel(resp, valid, pk, kernel_half), snr, med, mad

    def update(self, resp, valid, kernel_half, dt, tilt_rad):
        st = self.st
        self.predict(dt, tilt_rad)
        z, snr, med, mad = self.measure(resp, valid, kernel_half)
        self.snr = snr
        self.ref = mad

        if z is None:
            self.misses += 1
            if self.p is None:
                self.state = "no ball"
            elif self.misses <= st.coast_frames:
                self.state = "coast"
            else:
                self.state = "lost"
            return None

        if self.p is None or self.misses > st.max_misses:
            self.p = z
            self.v = 0.0
            self.P = np.diag([4.0, 1e4])
            self.state = "acquired"
        else:
            r = 1.5 + 6.0 / max(1.0, snr)      # better SNR -> trust it more
            self.correct(z, r)
            self.state = "lock"
        self.misses = 0
        return self.p

    def gate_bounds(self):
        """Current search window, for drawing. None if not tracking yet."""
        if self.p is None:
            return None, None
        st = self.st
        g = int(st.gate_px * (1.0 + st.gate_grow * self.misses))
        return max(0, int(self.p) - g), min(self.w, int(self.p) + g)

    def estimate(self):
        """Position/velocity for control, still valid while coasting."""
        if self.p is None:
            return None, 0.0
        if self.misses > self.st.coast_frames:
            return None, 0.0
        return self.p, self.v
