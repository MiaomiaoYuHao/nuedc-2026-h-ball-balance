"""Standalone bring-up checks. Run these in order before the camera.

    --enc-test    is the encoder wired and reading?
    --cal-steps   what is the driver's real microstep setting?
    --motor-test  does the beam sweep and hold position?
"""

import math
import time

from .kinematics import shaft_deg_to_tilt, step_limit


def encoder_test(st, enc):
    """Turn the crank by hand and watch the angle. Used to set
    enc_pwm_min / enc_pwm_max / enc_invert before anything else is tuned."""
    if not enc.ok:
        print("encoder not available")
        return
    print("rotate the shaft by hand. ctrl-c to stop.")
    print("expect: angle rises smoothly, no jumps except one wrap per turn")
    try:
        while True:
            time.sleep(0.2)
            print("raw %7.2f deg   continuous %8.2f   shaft %7.2f   "
                  "tilt %+6.3f deg   frames %d"
                  % (enc.raw_deg, enc.cont_deg, enc.shaft_deg(),
                     math.degrees(shaft_deg_to_tilt(st, enc.shaft_deg())),
                     enc.frames))
    except KeyboardInterrupt:
        print("\nstopped")


def cal_steps(st, enc, motor):
    """Measure the driver's real microstep setting.

    Boards without DIP switches have microstepping fixed in silicon or set
    by resistors, and guessing wrong scales every angle in the system. The
    encoder settles it: command a known number of steps, read how far the
    shaft actually turned. Fully automatic - do not touch the crank.
    """
    if not enc.wait_alive():
        print("encoder not reading - run --enc-test first")
        return

    n = st.steps_per_rev // 4        # a quarter turn if the guess is right
    print("commanding %d steps with microstep=%d assumed..." % (n, st.microstep))
    motor.enable(True)
    enc.set_zero()
    time.sleep(0.3)
    start = enc.shaft_deg()

    motor.set_target_steps(n)
    t0 = time.time()
    while time.time() - t0 < 15.0 and not motor.at_target():
        time.sleep(0.02)
    time.sleep(0.5)
    moved = enc.shaft_deg() - start

    if abs(moved) < 1.0:
        print("shaft did not move (%.2f deg). Check EN polarity, Vin, and "
              "that the motor is on the channel you wired." % moved)
    else:
        real = n * 360.0 / abs(moved) / st.gear_ratio
        micro = real / st.motor_steps_per_rev
        print("moved %+.2f deg -> %.0f steps/rev -> microstep is about %.2f"
              % (moved, real, micro))
        for c in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if abs(micro - c) < 0.15 * c:
                print("  -> set microstep = %d in config.py" % c)
                break
        else:
            print("  -> no standard value matches; check gear_ratio too")
        if moved < 0:
            print("  -> shaft turned negative for a positive command; "
                  "flip motor_dir_sign or enc_invert (only one of them)")

    motor.set_target_steps(0)
    time.sleep(2.0)


def motor_test(st, enc, motor):
    """Wiring check. Expected: the beam sweeps up, back to level, down, back
    to level, and holds position (shaft hard to turn) between moves."""
    lim = step_limit(st)
    print("STEP=BCM%d  DIR=BCM%d  EN=%s  active_low=%s  microstep=1/%d"
          % (st.pin_step, st.pin_dir,
             ("BCM%d" % st.pin_en) if st.pin_en is not None else "none",
             st.enable_active_low, st.microstep))
    print("step limit +/-%d steps for %.2f deg" % (lim, st.tilt_limit_deg))

    motor.enable(True)
    enc.wait_alive(1.0)
    enc.set_zero()
    for name, tgt in (("+limit", lim), ("level", 0),
                      ("-limit", -lim), ("level", 0)):
        print("  -> %s (%+d steps)" % (name, tgt))
        motor.set_target_steps(tgt)
        t0 = time.time()
        while time.time() - t0 < 6.0 and not motor.at_target():
            time.sleep(0.02)
        cur, _, _ = motor.snapshot()
        msg = "     reached %+d steps" % cur
        if enc.alive():
            msg += "   encoder %+.2f deg   tilt %+.3f deg" % (
                enc.shaft_deg(),
                math.degrees(shaft_deg_to_tilt(st, enc.shaft_deg())))
        print(msg)
        time.sleep(0.8)
    print("done. If nothing moved: check EN polarity, Vin, and the driver "
          "current setting (DIP 4-6).")
