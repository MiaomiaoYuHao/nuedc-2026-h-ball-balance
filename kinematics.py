"""Crank/beam geometry and the absolute-encoder safety window.

The linkage: a crank of radius `crank_radius` lifts one end of a beam of
length `lift_arm`, so beam tilt and crank angle are related by
    h = crank_radius * sin(crank)  =  lift_arm * sin(tilt)
Every conversion below is that one relation, solved for whichever end is
wanted, expressed in motor microsteps or degrees as convenient.
"""

import math


def tilt_to_steps(st, theta):
    h = st.lift_arm * math.sin(theta)
    s = max(-1.0, min(1.0, h / st.crank_radius))
    return math.asin(s) * st.steps_per_rev / (2.0 * math.pi)


def steps_to_tilt(st, steps):
    """Inverse of tilt_to_steps: crank steps -> beam tilt in radians."""
    crank = steps * 2.0 * math.pi / st.steps_per_rev
    h = st.crank_radius * math.sin(crank)
    return math.asin(max(-1.0, min(1.0, h / st.lift_arm)))


def tilt_to_crank_deg(st, theta):
    """Beam tilt (rad) -> crank angle (deg). Used to predict where the
    ENCODER will read for a given commanded tilt."""
    h = st.lift_arm * math.sin(theta)
    s = max(-1.0, min(1.0, h / st.crank_radius))
    return math.degrees(math.asin(s))


def shaft_deg_to_tilt(st, deg):
    h = st.crank_radius * math.sin(math.radians(deg))
    return math.asin(max(-1.0, min(1.0, h / st.lift_arm)))


def step_limit(st):
    """Step target corresponding to the configured tilt limit, never past
    the mechanical singularity of the linkage."""
    return int(abs(tilt_to_steps(st, min(st.tilt_limit,
                                         st.max_mech_tilt * 0.99))))


# ---------------- absolute encoder safety window ----------------

def encoder_deg_in_range(st, deg):
    return st.enc_abs_min <= deg <= st.enc_abs_max


def predicted_encoder_deg(st, theta, enc):
    """What the MT6816 would read if the crank reached tilt theta, given the
    zero point currently stored in enc. Assumes no full revolutions during
    normal operation - true for a beam that swings a few degrees."""
    crank_deg = tilt_to_crank_deg(st, theta) * st.gear_ratio
    return (enc.zero_deg + crank_deg) % 360.0


def clamp_tilt_to_encoder_range(st, theta, enc):
    """Shrink |theta| toward 0 (level) until the PREDICTED encoder reading
    stays inside [min+margin, max-margin].

    theta=0 is assumed safe; if it isn't, the beam was zeroed outside the
    allowed window and needs 'c' pressed somewhere else. Bisection is valid
    because tilt_to_crank_deg is monotonic in theta over the whole operating
    range (short of the mechanical singularity).

    This runs BEFORE every command, which is what keeps the beam inside the
    window without ever having to stop the control loop.
    """
    if not enc.alive():
        return theta   # can't check without the encoder - open loop only
    lo = st.enc_abs_min + st.enc_abs_margin
    hi = st.enc_abs_max - st.enc_abs_margin

    if lo <= predicted_encoder_deg(st, theta, enc) <= hi:
        return theta        # already safe, nothing to do

    lo_ok, hi_bad = 0.0, theta
    for _ in range(20):
        mid = (lo_ok + hi_bad) / 2.0
        if lo <= predicted_encoder_deg(st, mid, enc) <= hi:
            lo_ok = mid
        else:
            hi_bad = mid
    return lo_ok
