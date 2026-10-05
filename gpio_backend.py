"""GPIO backend.

Uses lgpio, not RPi.GPIO or pigpio. The Pi 5 replaced the old GPIO block
with a separate chip (RP1) reached through /dev/gpiochip*. RPi.GPIO predates
that and crashes with "Cannot determine SOC peripheral base address";
pigpio needs a background daemon whose service file doesn't start cleanly on
Bookworm. lgpio talks to the kernel gpiochip interface, works on every Pi
model including the 5, and needs no daemon.

    sudo apt install -y python3-lgpio
"""

HAVE_GPIO = True
try:
    import lgpio
except Exception:      # not on a Pi, or lgpio missing - run vision only
    HAVE_GPIO = False
    lgpio = None


def open_chip(chip):
    return lgpio.gpiochip_open(chip)


def close_chip(handle):
    lgpio.gpiochip_close(handle)


def claim_output(handle, pin, initial=0):
    lgpio.gpio_claim_output(handle, pin, initial)


def write(handle, pin, value):
    lgpio.gpio_write(handle, pin, value)


def free(handle, pin):
    try:
        lgpio.gpio_free(handle, pin)
    except Exception:
        pass
