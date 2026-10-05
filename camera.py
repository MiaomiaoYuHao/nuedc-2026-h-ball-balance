"""Camera source: background capture thread + exposure handling.

Capture runs in its own thread so the control loop never blocks on the ISP.
Auto exposure is left RUNNING by default - the matched filter's zero-mean
kernel plus vision.ae_scale() make the detector immune to brightness
changes, and locking exposure to a guessed value is how the image ends up
too dark on a rig it wasn't tuned for.

The frame-duration window is bounded so AE can't quietly stretch the
exposure and drop the loop to 8 fps in dim light.
"""

import threading
import time

import cv2

HAVE_PICAM = True
try:
    from picamera2 import Picamera2
except Exception:
    HAVE_PICAM = False


class CameraSource:
    def __init__(self, settings):
        self.st = settings
        self.frame = None
        self.seq = 0
        self.stop = False
        self.cv = threading.Condition()
        self.picam2 = None
        self.cap = None
        self.exposure_us = 0
        self.gain = 0.0
        self.ae_on = True

        st = settings
        if HAVE_PICAM:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(
                main={"size": (st.frame_width, st.frame_height),
                      "format": "RGB888"},
                buffer_count=4)
            self.picam2.configure(cfg)
            fd_min = int(1e6 / st.target_fps)
            fd_max = int(1e6 / st.cam_min_fps)
            try:
                self.picam2.set_controls({
                    "FrameDurationLimits": (fd_min, fd_max),
                    "AeEnable": True, "AwbEnable": True})
            except Exception as e:
                print("camera control warning: %s" % e)
            self.picam2.start()
            time.sleep(1.0)
            if st.cam_auto_lock:
                self.lock_exposure()
            else:
                print("auto exposure ACTIVE (%d..%d fps window), "
                      "baseline is brightness-compensated"
                      % (st.cam_min_fps, st.target_fps))
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, st.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, st.frame_height)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---------------- exposure ----------------
    def read_ae(self):
        """Report what the ISP is currently using (for the HUD)."""
        if self.picam2 is None:
            return
        try:
            md = self.picam2.capture_metadata()
            self.exposure_us = int(md.get("ExposureTime", 0))
            self.gain = float(md.get("AnalogueGain", 0.0))
        except Exception:
            pass

    def resume_auto(self):
        if self.picam2 is None:
            return
        st = self.st
        fd_min = int(1e6 / st.target_fps)
        fd_max = int(1e6 / st.cam_min_fps)
        try:
            self.picam2.set_controls({"FrameDurationLimits": (fd_min, fd_max),
                                      "AeEnable": True, "AwbEnable": True})
        except Exception as e:
            print("resume auto failed: %s" % e)
            return
        self.ae_on = True
        print("auto exposure RESUMED")

    def lock_exposure(self):
        """Let AE/AWB converge, read what they picked, then freeze it."""
        if self.picam2 is None:
            return
        st = self.st
        try:
            self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
        except Exception:
            pass
        time.sleep(st.cam_settle_s)

        exp, gain = 0, 0.0
        try:
            md = self.picam2.capture_metadata()
            exp = int(md.get("ExposureTime", 0))
            gain = float(md.get("AnalogueGain", 0.0))
        except Exception as e:
            print("metadata read failed: %s" % e)

        if st.cam_exposure_us > 0:
            exp = int(st.cam_exposure_us)
        if st.cam_gain > 0:
            gain = float(st.cam_gain)
        exp = int(exp * st.cam_exposure_scale)
        if exp <= 0:
            exp = 8000
        if gain <= 0:
            gain = 2.0

        # too long an exposure smears a fast ball; trade it for gain instead
        if exp > st.cam_max_exposure_us:
            gain *= exp / float(st.cam_max_exposure_us)
            exp = st.cam_max_exposure_us
        gain = min(gain, 16.0)

        fd = max(int(1e6 / st.target_fps), exp + 1500)
        try:
            self.picam2.set_controls({
                "FrameDurationLimits": (fd, fd), "AeEnable": False,
                "AwbEnable": False, "ExposureTime": exp,
                "AnalogueGain": gain})
        except Exception as e:
            print("camera lock warning: %s" % e)
        self.exposure_us, self.gain = exp, gain
        self.ae_on = False
        print("camera locked: exposure %d us, gain %.2f, max %.0f fps"
              % (exp, gain, 1e6 / fd))
        time.sleep(0.3)

    def bump_exposure(self, factor):
        """Manual brightness trim without redoing the whole AE lock."""
        if self.picam2 is None:
            return
        if self.ae_on:
            print("auto exposure is on, press e to lock it first")
            return
        st = self.st
        exp = int(max(200, min(st.cam_max_exposure_us,
                               self.exposure_us * factor)))
        gain = self.gain
        if exp == self.exposure_us:          # already at the cap, use gain
            gain = min(16.0, max(1.0, gain * factor))
        fd = max(int(1e6 / st.target_fps), exp + 1500)
        try:
            self.picam2.set_controls({"FrameDurationLimits": (fd, fd),
                                      "ExposureTime": exp,
                                      "AnalogueGain": float(gain)})
        except Exception as e:
            print("exposure trim failed: %s" % e)
            return
        self.exposure_us, self.gain = exp, gain
        print("exposure %d us  gain %.2f  (baseline is now stale, press b)"
              % (exp, gain))

    # ---------------- capture ----------------
    def _grab(self):
        if self.picam2 is not None:
            return self.picam2.capture_array()
        ok, f = self.cap.read()
        return f if ok else None

    def _run(self):
        while not self.stop:
            f = self._grab()
            if f is None:
                time.sleep(0.005)
                continue
            with self.cv:
                self.frame = f
                self.seq += 1
                self.cv.notify_all()

    def read(self, last_seq, timeout=1.0):
        with self.cv:
            if self.seq == last_seq:
                self.cv.wait(timeout)
            return self.frame, self.seq

    def close(self):
        self.stop = True
        time.sleep(0.05)
        if self.picam2 is not None:
            self.picam2.stop()
        if self.cap is not None:
            self.cap.release()
