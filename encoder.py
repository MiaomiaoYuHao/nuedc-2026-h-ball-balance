"""Absolute shaft angle from the MT6816 encoder built into the MS42CG.

Why the PWM output and not the A/B pair: A/B is 4096 counts/rev after
quadrature, ~5 kHz of edges while the crank swings. Python can't service
that without dropping counts, and a dropped count is a PERMANENT angle
error. The PWM line carries the ABSOLUTE angle in every frame at ~1 kHz, so
a missed frame costs nothing - the next one is still correct. It also
survives a power cycle, which an incremental count does not.
"""

import time

from . import gpio_backend as gp
from .gpio_backend import lgpio


class AngleEncoder:
    def __init__(self, settings):
        self.st = settings
        self.ok = False
        self.h = None
        self.raw_deg = 0.0        # 0..360, straight from the duty cycle
        self.cont_deg = 0.0       # unwrapped, continuous
        self.zero_deg = 0.0       # captured by the 'c' key
        self.frames = 0
        self.last_update = 0.0
        self._t_rise = None
        self._high = None
        self._period = None
        self._prev = None
        self._turns = 0
        self._cb = None

        if not settings.use_encoder:
            return
        if lgpio is None:
            print("lgpio not installed - encoder disabled, running open loop")
            return
        try:
            self.h = gp.open_chip(settings.gpio_chip)
            lgpio.gpio_claim_alert(self.h, settings.pin_enc_pwm,
                                   lgpio.BOTH_EDGES)
            self._cb = lgpio.callback(self.h, settings.pin_enc_pwm,
                                      lgpio.BOTH_EDGES, self._edge)
        except Exception as e:
            print("encoder GPIO claim failed: %s" % e)
            self.h = None
            return
        self.ok = True
        print("encoder: MT6816 PWM on BCM %d via lgpio" % settings.pin_enc_pwm)

    def _edge(self, chip, gpio, level, tick):
        # lgpio: level 1=rising, 0=falling, 2=watchdog timeout (ignored).
        # tick is nanoseconds; differences stay valid across the wrap.
        st = self.st
        if level == 1:
            if self._t_rise is not None:
                self._period = tick - self._t_rise
            self._t_rise = tick
        elif level == 0 and self._t_rise is not None:
            self._high = tick - self._t_rise
        if self._high is None or not self._period:
            return
        duty = self._high / float(self._period)
        if not (0.0 < duty < 1.0):
            return
        span = max(1e-3, st.enc_pwm_max - st.enc_pwm_min)
        a = (duty - st.enc_pwm_min) / span * 360.0
        a = max(0.0, min(360.0, a))
        if st.enc_invert:
            a = 360.0 - a
        self.raw_deg = a
        if self._prev is not None:
            d = a - self._prev
            if d > 180.0:
                self._turns -= 1
            elif d < -180.0:
                self._turns += 1
        self._prev = a
        self.cont_deg = self._turns * 360.0 + a
        self.frames += 1
        self.last_update = time.time()

    def alive(self):
        return self.ok and (time.time() - self.last_update) < 0.5

    def wait_alive(self, timeout=2.0):
        """The first PWM callback takes a moment to land; checking alive()
        immediately after construction always fails."""
        t0 = time.time()
        while not self.alive() and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.alive()

    def set_zero(self):
        if self.ok:
            self.zero_deg = self.cont_deg

    def shaft_deg(self):
        """Crank angle relative to the level position, in degrees."""
        return (self.cont_deg - self.zero_deg) / max(1e-6, self.st.gear_ratio)

    def close(self):
        if self._cb is not None:
            try:
                self._cb.cancel()
            except Exception:
                pass
            self._cb = None
        if self.h is not None:
            gp.free(self.h, self.st.pin_enc_pwm)
            gp.close_chip(self.h)
            self.h = None
