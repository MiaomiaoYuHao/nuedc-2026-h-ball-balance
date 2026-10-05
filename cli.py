"""Entry point: argument parsing and mode dispatch."""

import argparse
import sys

from .app import App
from .config import Settings
from .diagnostics import cal_steps, encoder_test, motor_test
from .encoder import AngleEncoder
from .gpio_backend import HAVE_GPIO
from .motor import Motor


def main(argv=None):
    ap = argparse.ArgumentParser(description="ball & beam balancer")
    ap.add_argument("--headless", action="store_true",
                    help="no GUI windows, maximum fps")
    ap.add_argument("--no-motor", action="store_true",
                    help="vision only, don't touch the motor")
    ap.add_argument("--enc-test", action="store_true",
                    help="print the encoder angle, no camera, no motion")
    ap.add_argument("--cal-steps", action="store_true",
                    help="measure the real microstep setting via the encoder")
    ap.add_argument("--motor-test", action="store_true",
                    help="wiring check: sweep the motor, no camera")
    args = ap.parse_args(argv)

    st = Settings()
    use_motor = HAVE_GPIO and not args.no_motor

    # ---- diagnostics: no camera, minimal setup, always clean up ----
    if args.enc_test or args.cal_steps or args.motor_test:
        enc = AngleEncoder(st)
        motor = Motor(st, use_motor=use_motor)
        motor.setup()
        motor.start()
        try:
            if args.enc_test:
                encoder_test(st, enc)
            elif args.cal_steps:
                cal_steps(st, enc, motor)
            else:
                motor_test(st, enc, motor)
        finally:
            motor.teardown()
            enc.close()
        return 0

    app = App(st, gui=not args.headless, use_motor=use_motor)
    app.start()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
