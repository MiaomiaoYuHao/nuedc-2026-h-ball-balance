"""Stepper motor: step generation thread, position target, safety limits.

The motor is driven as an ABSOLUTE POSITION servo: callers set a target step
count and a background thread walks the current position toward it one step
at a time. Nothing commands "speed" or "move by N" - that's what makes the
open-loop stepper's position meaningful and what lets the encoder verify it.
"""

import math
import threading
import time

from . import gpio_backend as gp
from .gpio_backend import HAVE_GPIO
from .kinematics import step_limit, steps_to_tilt, tilt_to_steps


class Motor:
    def __init__(self, settings, use_motor=True):
        self.st = settings
        self.use_motor = use_motor and HAVE_GPIO
        self.h = None
        self.lock = threading.Lock()
        self.target = 0
        self.current = 0
        self.zero_offset = 0
        self.enabled = False     # starts disabled: hand-level the beam first
        self.running = True
        self._thread = None

    # ---------------- hardware ----------------
    def setup(self):
        if not self.use_motor:
            return
        st = self.st
        self.h = gp.open_chip(st.gpio_chip)
        gp.claim_output(self.h, st.pin_dir, 0)
        gp.claim_output(self.h, st.pin_step, 0)
        if st.pin_en is not None:
            gp.claim_output(self.h, st.pin_en,
                            0 if st.enable_active_low else 1)
        self.enable(False)   # stay limp until the loop is actually started

    def enable(self, on):
        """Driver enable. Disabled means the coils are de-energised and the
        beam can be moved by hand - that is also the emergency stop."""
        self.enabled = bool(on)
        if not self.use_motor or self.st.pin_en is None or self.h is None:
            return
        if self.st.enable_active_low:
            gp.write(self.h, self.st.pin_en, 0 if on else 1)
        else:
            gp.write(self.h, self.st.pin_en, 1 if on else 0)

    def teardown(self):
        self.running = False
        time.sleep(0.05)
        if self.h is None:
            return
        self.enable(False)
        for pin in (self.st.pin_step, self.st.pin_dir, self.st.pin_en):
            if pin is not None:
                gp.free(self.h, pin)
        gp.close_chip(self.h)
        self.h = None

    # ---------------- stepping ----------------
    def start(self):
        fn = self._run_hw if self.use_motor else self._run_sim
        self._thread = threading.Thread(target=fn, daemon=True)
        self._thread.start()

    def _run_hw(self):
        st = self.st
        last_dir = None
        while self.running:
            with self.lock:
                target, cur, enabled = self.target, self.current, self.enabled
            if not enabled or target == cur:
                time.sleep(0.0008)
                continue
            forward = target > cur
            phys_forward = forward if st.motor_dir_sign > 0 else (not forward)
            if phys_forward != last_dir:
                gp.write(self.h, st.pin_dir, 1 if phys_forward else 0)
                last_dir = phys_forward
                time.sleep(st.dir_setup_s)
            gp.write(self.h, st.pin_step, 1)
            time.sleep(st.pulse_width_s)
            gp.write(self.h, st.pin_step, 0)
            time.sleep(max(0.0, 2.0 * st.step_delay - st.pulse_width_s))
            with self.lock:
                self.current += 1 if forward else -1

    def _run_sim(self):
        """No hardware: the 'motor' walks to target so the rest still runs."""
        while self.running:
            with self.lock:
                if self.current < self.target:
                    self.current += 1
                elif self.current > self.target:
                    self.current -= 1
            time.sleep(2.0 * self.st.step_delay)

    # ---------------- commands ----------------
    def command_tilt(self, theta):
        """Convert a tilt command to a step target and apply it.

        Two limits, for two different problems:

        step_deadband - if the new target is within this many steps of the
        one already commanded, the command is DROPPED. Tracker noise of a
        fraction of a millimetre otherwise makes the target wiggle by a step
        every frame, and since each wiggle can flip the sign of
        (target - current) the motor spends its time reversing instead of
        moving. Each reversal also takes up mechanical backlash, which turns
        electrical dither into audible chatter.

        max_step_jump - caps how far one accepted command may move the
        target, so a sudden setpoint flip can't ask for a bigger swing than
        the mechanics can clear in one go.

        Returns the tilt actually being held (not the tilt requested) so the
        caller's rate limiter integrates from reality rather than drifting
        while commands are being dropped.
        """
        st = self.st
        lim = step_limit(st)
        theta = max(-st.tilt_limit, min(st.tilt_limit, theta))
        steps = int(round(tilt_to_steps(st, theta)))
        steps = max(-lim, min(lim, steps))
        with self.lock:
            last_target = self.target - self.zero_offset
            if abs(steps - last_target) < st.step_deadband:
                return steps_to_tilt(st, last_target), last_target
            delta = max(-st.max_step_jump,
                        min(st.max_step_jump, steps - last_target))
            steps = last_target + delta
            self.target = steps + self.zero_offset
        return theta, steps

    def set_target_steps(self, steps):
        """Raw step target, bypassing tilt conversion. For diagnostics."""
        with self.lock:
            self.target = steps

    def set_zero_here(self):
        with self.lock:
            self.current = 0
            self.target = 0
            self.zero_offset = 0

    def trim_zero(self, delta):
        with self.lock:
            self.zero_offset += delta
            self.target += delta

    def resync_to(self, steps):
        """Rewrite the step counter from an external truth (the encoder)."""
        with self.lock:
            self.current = steps + self.zero_offset

    def snapshot(self):
        with self.lock:
            return self.current, self.target, self.zero_offset

    def at_target(self):
        with self.lock:
            return self.current == self.target
