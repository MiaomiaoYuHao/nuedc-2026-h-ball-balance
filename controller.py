"""Outer control loop: ball position error -> beam tilt command.

Two modes, switchable live with 'p':

  pid       The standard cascade design. This outer position loop (P+D,
            I=0) turns ball error into a target angle; the stepper's own
            open-loop accuracy, verified by the encoder, serves as the
            inner angle loop. Integral stays 0 - the plant is inherently
            unstable and integral only makes it harder to settle.

  bangbang  Slam the tilt one way, flip when the ball crosses the setpoint.
            Rougher on the ball but trivial to reason about, useful as a
            first "does it respond at all" check.

Every command goes out through Motor.command_tilt(), which applies the step
deadband and jump cap, and through clamp_tilt_to_encoder_range(), which
keeps the crank inside its absolute safety window.
"""

import math

from .config import K_PLANT
from .kinematics import clamp_tilt_to_encoder_range, predicted_encoder_deg


class BalanceController:
    def __init__(self, settings, motor, encoder):
        self.st = settings
        self.motor = motor
        self.enc = encoder
        self.mode = settings.control_mode
        self.prev_cmd_tilt = 0.0
        self.cmd_tilt = 0.0
        self.integ = 0.0
        self.vel_filt = 0.0
        self.bang_dir = 1.0
        self.sat_count = 0
        # memory for lost-ball recovery: where it was and which way it went
        self.last_seen_norm = 0.0
        self.last_seen_vel = 0.0
        self.searching = False

    def reset(self, tilt=0.0):
        self.integ = 0.0
        self.vel_filt = 0.0
        self.prev_cmd_tilt = tilt
        self.sat_count = 0

    def note_seen(self, ball_norm, vel_mps):
        """Remember the last confirmed sighting, for recovery later."""
        self.last_seen_norm = ball_norm
        self.last_seen_vel = vel_mps
        self.searching = False

    def search(self, lost_frames, dt):
        """Ball lost: tilt against the direction it was last heading.

        A ball is usually lost at one END of the pipe, and going level
        leaves it parked there forever. Tilting the other way rolls it back
        into view. +tilt accelerates the ball toward +x, so to bring back a
        ball last seen at +x we command negative tilt.

        Position is the better cue for "which end did it go to", with the
        last velocity as a fallback for a ball lost mid-flight. If the guess
        was wrong the direction flips periodically, turning this into a slow
        sweep rather than a stuck bet.
        """
        st = self.st
        cue = self.last_seen_norm
        if abs(cue) < 0.05:
            cue = self.last_seen_vel
        direction = -1.0 if cue >= 0 else 1.0

        elapsed = lost_frames - st.lost_search_frames
        if st.search_flip_frames > 0 and \
                (elapsed // st.search_flip_frames) % 2 == 1:
            direction = -direction

        self.searching = True
        self.integ = 0.0
        target = direction * math.radians(st.search_tilt_deg)
        return self._apply(target, st.tilt_rate_limit, dt)

    def toggle_mode(self):
        self.mode = "pid" if self.mode == "bangbang" else "bangbang"
        self.integ = 0.0
        return self.mode

    def filter_velocity(self, vel_mps):
        """Low-pass then deadband the velocity used by the D term.

        Near the setpoint the measured velocity is mostly tracker noise, and
        the D term is the only thing still acting there (the P term has gone
        to zero with the error). Damping against noise is exactly what makes
        the motor buzz once the ball has settled.
        """
        st = self.st
        if st.d_filter > 0.0:
            self.vel_filt = (st.d_filter * self.vel_filt
                             + (1.0 - st.d_filter) * vel_mps)
            v = self.vel_filt
        else:
            v = vel_mps
        return 0.0 if abs(v) < st.vel_deadband_mps else v

    def _apply(self, target_tilt, rate_limit, dt):
        du = max(-rate_limit * dt, min(rate_limit * dt,
                                       target_tilt - self.prev_cmd_tilt))
        safe = clamp_tilt_to_encoder_range(self.st, self.prev_cmd_tilt + du,
                                           self.enc)
        self.cmd_tilt, _ = self.motor.command_tilt(safe)
        self.prev_cmd_tilt = self.cmd_tilt
        return self.cmd_tilt

    def _bangbang(self, err_m, dt):
        st = self.st
        # flip only once the ball is clearly past the setpoint on the other
        # side; inside the hysteresis band keep pushing the same way, which
        # carries the ball through the crossing instead of stalling on it
        if err_m > st.bang_hyst_m:
            self.bang_dir = 1.0
        elif err_m < -st.bang_hyst_m:
            self.bang_dir = -1.0
        target = self.bang_dir * math.radians(st.bang_tilt_deg)

        # if the ball wants this direction but it would run into the
        # absolute boundary, that direction is a dead end - pushing harder
        # achieves nothing. Flip instead, so the motor always has something
        # useful to do rather than stalling against a wall.
        if self.enc.alive():
            pred = predicted_encoder_deg(st, target, self.enc)
            if not (st.enc_abs_min + st.enc_abs_margin <= pred
                    <= st.enc_abs_max - st.enc_abs_margin):
                self.bang_dir = -self.bang_dir
                target = self.bang_dir * math.radians(st.bang_tilt_deg)

        self.integ = 0.0
        return self._apply(target, st.bang_rate_limit, dt)

    def _pid(self, err_m, v_use, dt):
        st = self.st
        kp = st.wn * st.wn / K_PLANT
        kd = 2.0 * st.zeta * st.wn / K_PLANT
        if abs(err_m) < st.deadband_m:
            # inside the deadband the ball is where we want it: damp real
            # motion only, never react to a position error this small
            u = kd * (-v_use)
            self.integ *= 0.98
        else:
            if st.ki > 0.0:
                self.integ += err_m * dt
                ic = st.ki * self.integ
                if abs(ic) > st.i_limit:
                    self.integ = math.copysign(st.i_limit / st.ki, self.integ)
            u = kp * err_m + kd * (-v_use) + st.ki * self.integ
        u *= st.gain_scale
        return self._apply(u, st.tilt_rate_limit, dt)

    def update(self, err_m, vel_mps, dt):
        """Run one control tick. Returns the commanded tilt."""
        v_use = self.filter_velocity(vel_mps)
        if self.mode == "bangbang":
            tilt = self._bangbang(err_m, dt)
        else:
            tilt = self._pid(err_m, v_use, dt)

        st = self.st
        if abs(tilt) > 0.98 * st.tilt_limit and abs(v_use) < st.stuck_vel_mps:
            self.sat_count += 1
        else:
            self.sat_count = 0
        return tilt

    def go_level(self, dt=None):
        """Return to level rather than holding a stale tilt."""
        self.integ = 0.0
        self.searching = False
        safe = clamp_tilt_to_encoder_range(self.st, 0.0, self.enc)
        self.cmd_tilt, _ = self.motor.command_tilt(safe)
        self.prev_cmd_tilt = self.cmd_tilt
        self.vel_filt = 0.0
        self.sat_count = 0
        return self.cmd_tilt
