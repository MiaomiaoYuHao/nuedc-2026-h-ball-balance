"""All tunable values in one place.

Settings is a single mutable object passed to every subsystem. The GUI
sliders write straight into it, so "live tunable" needs no globals and no
re-import tricks - whoever holds the object sees the current value.

Fields are grouped the same way the old single-file config block was, so
values carried over 1:1. Derived quantities (gains, step limits) are
properties, recomputed on read, because the values they depend on can be
changed at runtime by a slider.
"""

import math
from dataclasses import dataclass

G = 9.81
K_PLANT = (5.0 / 7.0) * G      # a = K_PLANT * sin(theta) for a solid ball


@dataclass
class Settings:
    # ---------- frame / files ----------
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 60
    config_path: str = "groove_config.npz"

    # ---------- camera ----------
    # Autoexposure normally ruins a baseline-subtraction detector, but a
    # hardcoded exposure is dark on a rig it wasn't tuned for. AE stays ON
    # and the baseline is brightness-compensated instead (see vision.ae_scale).
    cam_auto_lock: bool = False     # False = leave auto exposure RUNNING
    cam_settle_s: float = 2.0       # AE convergence time before freezing
    cam_exposure_us: int = 0        # 0 = use whatever AE picked
    cam_gain: float = 0.0           # 0 = use whatever AE picked
    cam_exposure_scale: float = 1.0
    cam_max_exposure_us: int = 12000   # longer than this and a fast ball smears
    cam_min_fps: int = 30           # AE may not stretch the frame beyond this
    ae_compensate: bool = True      # rescale baseline to current brightness

    # ---------- rig geometry - measure with a ruler ----------
    roi_span_m: float = 0.50        # real pipe length covered by the ROI box
    crank_radius: float = 0.030
    lift_arm: float = 0.50

    # ---------- motor pins (BCM) ----------
    # Chosen so I2C(2,3), SPI(7..11) and UART(14,15) stay free.
    pin_step: int = 18              # header pin 12
    pin_dir: int = 23               # header pin 16
    pin_en: int = 24                # header pin 18, None if not wired
    # MT6816 encoder built into the MS42CG
    pin_enc_pwm: int = 12           # header pin 32 - absolute angle
    pin_enc_a: int = 5              # header pin 29 - incremental (unused)
    pin_enc_b: int = 6              # header pin 31 - incremental (unused)
    pin_enc_z: int = 13             # header pin 33 - index (unused)
    # Encoder VCC -> 3V3. NEVER 5 V: outputs swing to VCC and 5 V on a Pi
    # input is out of spec.
    gpio_chip: int = 0

    # Enable polarity. GET THIS RIGHT OR THE MOTOR NEVER ENERGISES.
    #   WHEELTEC D36A            : EN ACTIVE HIGH (low = sleep) -> False
    #   A4988 / DRV8825 / TMC2209: /EN ACTIVE LOW               -> True
    enable_active_low: bool = False
    pulse_width_s: float = 0.000005   # STEP high time; TB6600 wants >= 2.5 us

    microstep: int = 16             # must match the driver DIP switches
    motor_steps_per_rev: int = 200  # 1.8 deg/step, MS42CG datasheet
    gear_ratio: float = 1.0         # 1.0 MS42CG, 13.8 geared MS42CGR
    step_rate_sps: int = 2500       # motor pulse rate, steps/sec (was a
                                     # fixed half-period; now live-tunable)
    dir_setup_s: float = 0.000005

    # *** FLIP IF THE BALL RUNS AWAY INSTEAD OF CENTERING (or press 'j') ***
    motor_dir_sign: int = +1

    # Cap on how far ONE accepted command may move the step target.
    max_step_jump: int = 60
    # Ignore target changes smaller than this - the anti-chatter knob.
    step_deadband: int = 8

    # ---------- encoder / inner angle loop ----------
    use_encoder: bool = True
    enc_pwm_min: float = 0.02       # duty at 0 deg   (trim with --enc-test)
    enc_pwm_max: float = 0.98       # duty at 360 deg
    enc_invert: bool = False
    loststep_deg: float = 2.0       # cmd vs measured mismatch that counts
    loststep_frames: int = 15       # ... this many frames before resync
    enc_resync: bool = True

    # Hard safety window on the RAW absolute encoder angle, independent of
    # the software zero set by 'c'. Keeps the crank off a hard stop.
    enc_abs_min: float = 190.0
    enc_abs_max: float = 310.0
    enc_abs_margin: float = 3.0     # start refusing this many deg early

    # ---------- control ----------
    control_mode: str = "pid"       # "pid" or "bangbang", 'p' toggles live
    wn: float = 2.2                 # closed loop bandwidth, rad/s
    zeta: float = 0.9
    tilt_limit_deg: float = 1.0
    # Low-pass on the velocity fed into KD. MUST be on: near the setpoint
    # that signal is mostly tracker noise, and KD amplifies it straight
    # into a jittery motor.
    d_filter: float = 0.75
    deadband_m: float = 0.005
    # Speeds below this are treated as zero before reaching the D term.
    vel_deadband_mps: float = 0.008
    ki: float = 0.0
    i_limit: float = 0.02
    # Overall multiplier on KP and KD. The WN/ZETA formula assumes an
    # idealized solid ball; real rigs are usually more sensitive, so this
    # backs the whole loop off without recalculating WN.
    gain_scale: float = 0.4
    setpoint_norm: float = 0.0
    setpoint_step: float = 0.05
    tilt_rate_dps: float = 25.0     # rad/s equivalent, protects the linkage
    coast_frames: int = 12          # keep controlling on the Kalman estimate
    return_to_level_on_lost: bool = True

    # ---------- lost-ball recovery ----------
    # The ball is easiest to lose at the ends of the pipe, where it leaves
    # the ROI or the lighting is worst. Going level there just leaves it
    # parked. Instead, after this many lost frames, tilt AGAINST the
    # direction it was last travelling to roll it back into view.
    lost_search_frames: int = 30
    search_tilt_deg: float = 0.8    # clamped by tilt_limit_deg anyway
    # If the guess was wrong (ball went the other way) flip the search
    # direction every this many frames, so it sweeps instead of giving up.
    search_flip_frames: int = 90

    # ---------- bang-bang mode ----------
    bang_tilt_deg: float = 2.0
    bang_hyst_m: float = 0.004
    bang_rate_dps: float = 400.0

    # stuck detection: saturated tilt but the ball is not moving
    stuck_vel_mps: float = 0.004
    stuck_frames: int = 45

    # ---------- vision ----------
    row_trim_pct: int = 10          # ignore top/bottom % of ROI rows
    ball_sigma_px: float = 8.0      # ball feature half width
    kernel_span: float = 3.0        # kernel support = span * sigma each side
    smooth_win: int = 5

    snr_acquire: float = 6.0        # SNR needed to grab a ball from nothing
    snr_keep: float = 3.0           # SNR needed to stay locked
    runnerup_ratio: float = 0.75    # 2nd peak must be below this * best
    gate_px: int = 60               # search radius around PREDICTED position
    gate_grow: float = 0.5          # gate widens per missed frame
    gate_max_mult: float = 3.0      # but never beyond this multiple
    max_misses: int = 10            # re-acquire after this many misses

    base_adapt: float = 0.002       # per-frame baseline learning rate
    base_keepout_px: int = 40       # do not adapt this close to the ball

    # vision slider values (tuned on the rig 2026-07-31)
    w_dark: float = 0.1
    w_tex: float = 0.2
    use_binarize: int = 0
    bin_thresh: int = 128
    use_board_mask: int = 1
    board_thresh: int = 40          # 0 = auto (percentile of the ROI)
    min_board_pct: int = 29
    margin_left: int = 0
    margin_right: int = 0

    # ---------- diagnostics ----------
    # Rolling history for the 'z' dump: at 60 fps, 240 frames is ~4 s.
    diag_buffer_len: int = 240
    diag_dump_rows: int = 120       # how many rows the dump actually prints

    # ---------- display ----------
    color_mode: int = 2             # 0 raw, 1 swap R/B, 2 gray
    view_mode: int = 0              # 0 normal, 1 binary
    plot_height: int = 140
    draw_every: int = 1
    print_every_n: int = 15
    min_healthy_fps: float = 25.0

    # ---------- derived ----------
    @property
    def steps_per_rev(self):
        return int(self.motor_steps_per_rev * self.microstep * self.gear_ratio)

    @property
    def tilt_limit(self):
        return math.radians(self.tilt_limit_deg)

    @property
    def tilt_rate_limit(self):
        return math.radians(self.tilt_rate_dps)

    @property
    def bang_rate_limit(self):
        return math.radians(self.bang_rate_dps)

    @property
    def max_mech_tilt(self):
        return math.asin(min(1.0, self.crank_radius / self.lift_arm))

    @property
    def kp(self):
        return self.wn * self.wn / K_PLANT

    @property
    def kd(self):
        return 2.0 * self.zeta * self.wn / K_PLANT

    @property
    def step_delay(self):
        """Half-period between STEP pulses, in seconds. motor.py sleeps this
        long twice per step, so 1/(2*step_delay) == step_rate_sps."""
        return 1.0 / (2.0 * max(1, self.step_rate_sps))