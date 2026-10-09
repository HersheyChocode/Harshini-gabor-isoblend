"""Haploscope display: one spanning psykit window, or two plain PsychoPy windows.

Modes
-----
`single-window` (production, recommended). ONE plain PsychoPy window covering the
    whole merged desktop, with each eye's patch placed on its own panel and
    mirrored about its own centre. One window means one `flip()`, so interocular
    delay is zero by construction. No psykit, no framebuffer round trip, and it
    does not care what shape the merged desktop is: name the panel each eye lives
    on with `eye_panels` and give the desktop size as `window_pix`. Use this when
    the merge is anything other than exactly two panels wide, which includes the
    three-wide NVIDIA Surround arrangement a GeForce card forces you into.

`left/right` (production). ONE psykit StereoWindow spanning both panels. Each eye
    is rendered into its own framebuffer and blitted side by side, so a single
    `flip()` presents both eyes in one buffer swap and interocular delay is zero
    by construction. Requires the merged desktop to be EXACTLY two panels wide,
    because psykit splits the window in half. Otherwise identical in effect to
    `single-window`, with an extra texture round trip; prefer `single-window`
    unless you specifically want psykit's `fixationVergence`.

`two-window` (fallback). Two plain PsychoPy windows, one per panel, with only the
    first waiting for the vertical retrace. The second window free-runs, so the
    eyes are NOT guaranteed to be on the same frame. Use it to check that the
    stimulus renders, not to collect data.

`dual-head` (comparison only). psykit's own two-window implementation. It has the
    same timing exposure as `two-window` and is strictly worse: see the notes on
    `_dual_head_class` and on `_TWO_WINDOW_RATIONALE` below. Kept so the two can
    be compared directly; there is no reason to prefer it.

Why the fallback is hand-rolled rather than psykit's dual-head
--------------------------------------------------------------
psykit's dual-head blits the right eye AFTER the first window's flip has already
returned:

    self._blipEyeBuffer(eye='left')
    flipTime = super().flip(...)        # left eye presented, blocks on retrace
    self.win2._setCurrent()
    self._blipEyeBuffer(eye='right')    # right eye's back buffer filled only now
    self.win2.flip(...)                 # waitBlanking=False

so the right eye's pixels are not even written until the left eye is on screen,
and a full-screen textured blit plus a context switch sit inside the gap. Two
plain windows fill both back buffers before either flip, leaving only the swap
itself to be delayed. Same architecture, less latency between the eyes, no FBO
round trip, and none of psykit's dual-head defects. Nothing is given up: psykit
applies `fixationVergence` and `fixationOffset` only in the left/right branch, so
dual-head ignores them anyway.

A static random-dot stereogram is immune to interocular delay because every frame
is identical. A dynamic one is not: one frame of lag replaces one eye's dots with
an independent sample and binocular correlation goes to zero.

Geometry note for `left/right`
------------------------------
psykit samples the central half of each eye's framebuffer and maps it 1:1 onto
that eye's half of the window, so there is no horizontal compression and a
stimulus drawn at window-pixel (0, 0) lands at the centre of each panel. Draw
each eye centred and use `align_px` for per-eye centring offsets.
"""

import ctypes
import sys

import numpy as np


def ensure_dpi_aware():
    """Tell Windows this process reports real pixels. Runs at import, before any window.

    A DPI-unaware process is handed display sizes divided by the scale factor, so a
    1920x1080 panel at 150% scaling is reported as 1280x720 and PsychoPy opens a
    window of that size. Every angular quantity then inherits a silent 1.5x error.
    This must run before pyglet enumerates the displays, which is why it is called
    at module import rather than from StereoDisplay.
    """
    if sys.platform != "win32":
        return "n/a"
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # PER_MONITOR_DPI_AWARE
        return "per-monitor"
    except OSError:
        return "already set"                                # E_ACCESSDENIED: fine
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            return "system"
        except Exception:
            return "FAILED"


DPI_AWARENESS = ensure_dpi_aware()


def windows_display_modes():
    """{(x, y): {'width','height','hz','gpu'}} per attached display, from the driver.

    Keyed by desktop position, which is how a pyglet screen is matched to it.
    Returns {} off Windows or if anything goes wrong; this is a diagnostic, never
    load-bearing.
    """
    if sys.platform != "win32":
        return {}
    try:
        class DD(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("DeviceName", ctypes.c_wchar * 32),
                        ("DeviceString", ctypes.c_wchar * 128),
                        ("StateFlags", ctypes.c_ulong), ("DeviceID", ctypes.c_wchar * 128),
                        ("DeviceKey", ctypes.c_wchar * 128)]

        class DM(ctypes.Structure):
            _fields_ = [("dmDeviceName", ctypes.c_wchar * 32), ("dmSpecVersion", ctypes.c_ushort),
                        ("dmDriverVersion", ctypes.c_ushort), ("dmSize", ctypes.c_ushort),
                        ("dmDriverExtra", ctypes.c_ushort), ("dmFields", ctypes.c_ulong),
                        ("dmPositionX", ctypes.c_long), ("dmPositionY", ctypes.c_long),
                        ("dmDisplayOrientation", ctypes.c_ulong),
                        ("dmDisplayFixedOutput", ctypes.c_ulong), ("dmColor", ctypes.c_short),
                        ("dmDuplex", ctypes.c_short), ("dmYResolution", ctypes.c_short),
                        ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short),
                        ("dmFormName", ctypes.c_wchar * 32), ("dmLogPixels", ctypes.c_ushort),
                        ("dmBitsPerPel", ctypes.c_ulong), ("dmPelsWidth", ctypes.c_ulong),
                        ("dmPelsHeight", ctypes.c_ulong), ("dmDisplayFlags", ctypes.c_ulong),
                        ("dmDisplayFrequency", ctypes.c_ulong), ("dmICMMethod", ctypes.c_ulong),
                        ("dmICMIntent", ctypes.c_ulong), ("dmMediaType", ctypes.c_ulong),
                        ("dmDitherType", ctypes.c_ulong), ("dmReserved1", ctypes.c_ulong),
                        ("dmReserved2", ctypes.c_ulong), ("dmPanningWidth", ctypes.c_ulong),
                        ("dmPanningHeight", ctypes.c_ulong)]

        user32, out, i = ctypes.windll.user32, {}, 0
        while True:
            dd = DD()
            dd.cb = ctypes.sizeof(dd)
            if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
                break
            i += 1
            if not dd.StateFlags & 0x1:                     # not attached to desktop
                continue
            dm = DM()
            dm.dmSize = ctypes.sizeof(dm)
            if user32.EnumDisplaySettingsW(dd.DeviceName, -1, ctypes.byref(dm)):
                out[(dm.dmPositionX, dm.dmPositionY)] = dict(
                    width=int(dm.dmPelsWidth), height=int(dm.dmPelsHeight),
                    hz=int(dm.dmDisplayFrequency), gpu=dd.DeviceString,
                    device=dd.DeviceName)
        return out
    except Exception:
        return {}


def _screen_mode(index):
    """Driver mode for pyglet screen `index`, matched by desktop position."""
    try:
        import pyglet

        s = pyglet.canvas.get_display().get_screens()[index]
        return windows_display_modes().get((int(s.x), int(s.y)))
    except Exception:
        return None


SPANNING_MODES = ("left/right", "right/left")
TWO_WINDOW_MODES = ("two-window", "dual-head")
SINGLE_WINDOW_MODES = ("single-window",)
SYNCED_MODES = SPANNING_MODES + SINGLE_WINDOW_MODES
MODES = SYNCED_MODES + TWO_WINDOW_MODES

# psykit's screen-copy quad, replicated verbatim from stereomode.py so a second
# copy can be built in the second window's OpenGL context. See _dual_head_class.
_SCREEN_VERTS = [1, -1, 0.0, 1, 0,
                 -1, -1, 0.0, 0, 0,
                 -1, 1, 0.0, 0, 1,
                 1, 1, 0.0, 1, 1]
_SCREEN_ATTRS = [("position", 3, False), ("tex_coords", 2, False)]
_SCREEN_IDX = [0, 1, 2, 3, 0, 2]


def _dual_head_class(StereoWindow):
    """Subclass repairing two defects in psykit's dual-head path.

    1. GL_INVALID_OPERATION on glBindVertexArray. psykit builds one screen-copy
       VAO in the main window's context, then blits the right eye with the
       SECOND window's context current. Vertex array objects are container
       objects and are never shared between contexts, even when the contexts
       share everything else. pyglet does put both windows in one share group, so
       the textures and shader programs the blit needs are valid in the second
       context; only the VAO name is not. The fix is a second VAO, generated in
       the second context, swapped in for the right-eye blit.

    2. A sequential-mode artefact drawn in dual-head. psykit calls
       `_drawBlueLine` unconditionally in the dual-head branch of `flip`. Those
       lines are the blue-line sync signal for shutter glasses and have no
       meaning on a mirror haploscope, but they are still drawn: a full-width
       3 px WHITE band across the bottom of the left eye and a BLACK one across
       the bottom of the right eye. That is a large, high-contrast,
       interocularly anticorrelated band in the near periphery, which is a
       rivalry and luster generator sitting next to a stimulus whose whole point
       is stable fusion. Suppressed here.
    """

    class DualHeadStereoWindow(StereoWindow):
        _screenVAO2 = None

        def build_second_vao(self):
            """Generate the screen-copy VAO inside win2's context. Call once."""
            from psykit import gltools

            self.win2._setCurrent()
            try:
                self._screenVAO2 = gltools.create_vertex_array(
                    _SCREEN_VERTS, _SCREEN_ATTRS, _SCREEN_IDX)
            finally:
                self._setCurrent()

        def _blipEyeBuffer(self, eye):
            if eye == "right" and self._screenVAO2 is not None:
                saved = self._screenVAO
                self._screenVAO = self._screenVAO2
                try:
                    super()._blipEyeBuffer(eye)
                finally:
                    self._screenVAO = saved
            else:
                super()._blipEyeBuffer(eye)

        def _drawBlueLine(self, eye, loc="bottom"):
            return

    return DualHeadStereoWindow


class Optics:
    """Pixel <-> visual angle for one haploscope arm.

    `distance_cm` is the whole optical path, eye -> mirror(s) -> screen, and assumes the
    mirrors are set so the virtual image is frontoparallel to the eye. Horizontal and
    vertical pitch are kept apart: they coincide only when `size_cm` matches the panel's
    pixel aspect ratio, and disparity lives on the horizontal axis, so conflating them
    would put a systematic scale error straight into the disparities.
    """

    def __init__(self, size_pix=(1920, 1080), size_cm=(34.0, 18.0), distance_cm=80.0):
        self.size_pix, self.size_cm, self.distance_cm = size_pix, size_cm, distance_cm
        self.px_per_cm_h = size_pix[0] / size_cm[0]
        self.px_per_cm_v = size_pix[1] / size_cm[1]

    @property
    def pixel_aspect(self):
        """Horizontal / vertical pixel density. 1.0 means physically square pixels."""
        return self.px_per_cm_h / self.px_per_cm_v

    def deg2cm(self, deg):
        """Frontoparallel extent subtending `deg`, centred on the line of sight."""
        return 2.0 * self.distance_cm * np.tan(np.radians(deg) / 2.0)

    def cm2deg(self, cm):
        return 2.0 * np.degrees(np.arctan(cm / (2.0 * self.distance_cm)))

    def deg2px(self, deg, vertical=False):
        ppc = self.px_per_cm_v if vertical else self.px_per_cm_h
        return self.deg2cm(deg) * ppc

    def px2deg(self, px, vertical=False):
        ppc = self.px_per_cm_v if vertical else self.px_per_cm_h
        return self.cm2deg(px / ppc)

    def arcmin2px(self, arcmin):
        return self.deg2px(arcmin / 60.0)

    def px2arcmin(self, px):
        return 60.0 * self.px2deg(px)

    @property
    def px_per_deg(self):
        return self.deg2px(1.0)

    def warnings(self):
        """List of measurement problems that would bias every disparity. Empty is good."""
        out = []
        if abs(self.pixel_aspect - 1.0) > 0.01:
            w, h = self.size_cm
            px, py = self.size_pix
            out.append(
                f"pixel aspect {self.pixel_aspect:.4f}: {w} x {h} cm is not {px}:{py}. "
                f"If the height is right the width should be {h * px / py:.2f} cm; if the "
                f"width is right the height should be {w * py / px:.2f} cm. The horizontal "
                f"figure sets every disparity, so re-measure the active area.")
        return out


class StereoDisplay:
    """Both haploscope arms, behind one interface, whichever mode is in use.

    `align_px` holds per-eye centring offsets in observer coordinates: the
    software pre-mirroring and the physical mirror cancel. `vergence_px` is a
    global inward shift and is honoured in spanning modes only, because psykit
    applies it in the left/right branch of `flip` and nowhere else.
    """

    def __init__(self, stim, optics, mode="left/right", screen=0, screen2=1,
                 mirrors_per_eye=1, align_px=((0, 0), (0, 0)), vergence_px=0.0,
                 fusion_lock=True, full_screen=True, monitor_name="haploscope",
                 photodiode=False, record_frame_intervals=True,
                 window_pix=None, eye_panels=(0, 1)):
        from psychopy import monitors, visual

        if mode not in MODES:
            raise ValueError(f"unsupported mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.optics = optics
        self.panel_pix = tuple(optics.size_pix)
        self.eye_panels = tuple(eye_panels)
        self.win = None                                 # the psykit window, if any
        # Driver-truth modes for the screens this mode actually uses, captured
        # before any window exists so warnings() can compare them.
        used = (screen, screen2) if mode in TWO_WINDOW_MODES else (screen,)
        self._screen_modes = {s: _screen_mode(s) for s in used}

        self._monitor_known = monitor_name in monitors.getAllMonitors()
        mon = monitors.Monitor(monitor_name)            # carries the gamma calibration
        self._gamma_known = mon.getGammaGrid() is not None
        mon.setDistance(optics.distance_cm)
        mon.setWidth(optics.size_cm[0])                 # one panel, not the spanned pair
        mon.setSizePix(list(optics.size_pix))

        mirrored = bool(mirrors_per_eye % 2)
        common = dict(fullscr=full_screen, units="pix", color=(0, 0, 0),
                      colorSpace="rgb", monitor=mon, allowGUI=False)
        flip_horiz = False
        pw, ph = self.panel_pix

        if mode == "single-window":
            # One plain window over the whole merged desktop, each eye's patch
            # placed on its own panel. The merged desktop's shape is irrelevant:
            # two panels wide, three panels wide, whatever the driver produced,
            # you name the panel each eye lives on. The mirror is applied per
            # patch (flipHoriz) rather than per window (viewScale), because a
            # window-wide flip would also swap which panel each eye lands on.
            win_w, win_h = (window_pix if window_pix else (pw * 2, ph))
            self.wins = [visual.Window(size=[int(win_w), int(win_h)], screen=screen,
                                       waitBlanking=True, **common)]
            targets = [self.wins[0], self.wins[0]]
            centres = [((k + 0.5) * pw - win_w / 2.0 + ax, ay)
                       for k, (ax, ay) in zip(self.eye_panels, align_px)]
            flip_horiz = mirrored
        elif mode == "two-window":
            # The pre-psykit arrangement: one plain window per panel, both back
            # buffers filled before either flip, only the first blocking on the
            # retrace. Two blocking flips per frame on unsynchronised panels can
            # halve the effective frame rate, which is why the second is free.
            common["viewScale"] = (-1, 1) if mirrored else (1, 1)
            self.wins = [
                visual.Window(size=list(optics.size_pix), screen=s,
                              waitBlanking=(i == 0), **common)
                for i, s in enumerate((screen, screen2))]
            targets = self.wins
            centres = list(align_px)
        else:
            from psykit.stereomode import StereoWindow

            common["viewScale"] = (-1, 1) if mirrored else (1, 1)
            win_size = ([pw * 2, ph] if mode in SPANNING_MODES else list(optics.size_pix))
            kwargs = dict(size=win_size, screen=screen, waitBlanking=True, **common)
            if mode == "dual-head":
                kwargs["screen2"] = screen2
                self.win = _dual_head_class(StereoWindow)(stereoMode=mode, **kwargs)
                self.win.build_second_vao()
            else:
                self.win = StereoWindow(stereoMode=mode, **kwargs)
            if vergence_px and mode in SPANNING_MODES:
                self.win.fixationVergence = vergence_px
            self.wins = [self.win]
            targets = [self.win, self.win]
            centres = list(align_px)

        self._vergence_ignored = bool(vergence_px) and mode not in SPANNING_MODES
        for w in self.wins:
            w.recordFrameIntervals = record_frame_intervals

        w_px, h_px = stim.Wpx, stim.Hpx
        blank = np.zeros((h_px, w_px), dtype=np.float32)
        self.centres = dict(zip(("left", "right"), centres))
        self.panels, self.locks, self.diodes = {}, {}, {}
        for (eye, centre), target in zip(zip(("left", "right"), centres), targets):
            self.panels[eye] = visual.ImageStim(
                target, image=blank, size=(w_px, h_px), pos=centre, units="pix",
                interpolate=False, flipHoriz=flip_horiz)
            self.locks[eye] = (self._lock(target, centre, w_px, h_px, eye,
                                          optics.px_per_deg) if fusion_lock else [])
            self.diodes[eye] = (self._diode(target, centre, self.panel_pix)
                                if photodiode else None)
        self._diode_on = False

    # ------------------------------------------------------------------ parts

    @staticmethod
    def _lock(win, centre, w_px, h_px, eye, px_per_deg):
        """Zero-disparity surround frame plus one half of a nonius pair.

        With the troughs of the grating in the background plane, mean disparity
        inside the aperture is crossed, so vergence has a standing pull on it.
        The lock gives vergence a zero-disparity reference to hold instead.
        """
        from psychopy import visual

        cx, cy = centre
        gap, length = 0.15 * px_per_deg, 0.45 * px_per_deg
        sign = +1 if eye == "left" else -1              # left eye upper, right eye lower
        return [visual.Rect(win, width=w_px + 4, height=h_px + 4, pos=centre,
                            lineColor="white", fillColor=None, lineWidth=2, units="pix"),
                visual.Line(win, start=(cx, cy + sign * gap),
                            end=(cx, cy + sign * (gap + length)),
                            lineColor="white", lineWidth=2, units="pix")]

    @staticmethod
    def _diode(win, centre, panel_pix):
        """Corner patch that alternates every frame, for photodiode timing checks.

        Placed relative to this eye's own panel centre, so it lands on that eye's
        panel whatever the merged desktop looks like.
        """
        from psychopy import visual

        side = 60
        cx, cy = centre
        pos = (cx + panel_pix[0] / 2 - side, cy - (panel_pix[1] / 2 - side))
        return visual.Rect(win, width=side, height=side, pos=pos, units="pix",
                           lineColor=None, fillColor="white")

    # ------------------------------------------------------------------- loop

    def draw(self, left, right):
        """No panel swap: image `left` goes to the buffer the left eye looks at.

        Both eyes are drawn before anything is flipped, in every mode.
        """
        self._diode_on = not self._diode_on
        for eye, img in (("left", left), ("right", right)):
            if self.win is not None:
                self.win.setBuffer(eye)
            panel = self.panels[eye]
            panel.image = img
            panel.draw()
            for s in self.locks[eye]:
                s.draw()
            diode = self.diodes[eye]
            if diode is not None:
                diode.fillColor = "white" if self._diode_on else "black"
                diode.draw()

    def flip(self):
        """Present. One swap in a spanning mode, two otherwise. Returns flip time."""
        t = self.wins[0].flip()
        for win in self.wins[1:]:
            win.flip()
        return t

    # ------------------------------------------------------------- diagnostics

    @property
    def dropped_frames(self):
        """Dropped frames on the timed window. The free-running one is not timed."""
        return self.wins[0].nDroppedFrames

    def frame_intervals(self):
        return np.asarray(self.wins[0].frameIntervals, dtype=float)

    def timing_report(self):
        """Frame-interval summary. A dynamic stereogram cannot survive dropped frames."""
        iv = self.frame_intervals()
        if iv.size == 0:
            return "no frame intervals recorded"
        ms = iv * 1000.0
        tail = ("" if self.mode in SYNCED_MODES else
                "  [first window only; the second free-runs and is not timed]")
        return (f"{iv.size} frames, median {np.median(ms):.2f} ms, "
                f"95th pct {np.percentile(ms, 95):.2f} ms, max {ms.max():.2f} ms, "
                f"dropped {self.dropped_frames}{tail}")

    def warnings(self):
        """Configuration problems worth refusing to collect data under."""
        out = []

        # The one that silently destroys a dynamic stereogram while leaving a
        # static one looking perfect: the two panels refreshing at different
        # rates. The slower eye holds a stale dot field, so on every frame the
        # fast eye has moved on and the two eyes carry independent samples.
        known = {s: m for s, m in self._screen_modes.items() if m}
        rates = {m["hz"] for m in known.values()}
        if len(rates) > 1:
            detail = ", ".join(f"screen {s} = {m['hz']} Hz" for s, m in sorted(known.items()))
            out.append(
                f"THE PANELS REFRESH AT DIFFERENT RATES ({detail}). A dynamic random-dot "
                f"stereogram cannot be fused under this: the slower panel holds a stale "
                f"dot field, so on the intervening frames the two eyes carry statistically "
                f"independent samples and binocular correlation drops to zero. A STATIC "
                f"stereogram is completely unaffected, which is why this hides. Set both "
                f"panels to the same refresh rate in Display Settings before anything else.")
        for s, m in known.items():
            if m["hz"] < 50:
                out.append(f"screen {s} is running at {m['hz']} Hz; check the cable, the "
                           f"port and the EDID, not just the Display Settings dropdown.")
        gpus = {m["gpu"] for m in known.values()}
        if len(gpus) > 1:
            out.append(
                f"the panels are on different GPUs ({sorted(gpus)}). They can never be "
                f"frame-synced and no merge technology can group them.")
        if DPI_AWARENESS == "FAILED":
            out.append("could not set DPI awareness; if Windows scaling is not 100% on the "
                       "stimulus panels, every size reported here is wrong.")
        for s, m in known.items():
            if m and self.panel_pix != (m["width"], m["height"]):
                out.append(
                    f"screen {s} is physically {m['width']}x{m['height']} but Optics is "
                    f"built on {self.panel_pix[0]}x{self.panel_pix[1]}. Every angular "
                    f"quantity is scaled by {self.panel_pix[0] / m['width']:.3f} in x.")
        if self.mode in TWO_WINDOW_MODES:
            out.append(
                f"{self.mode} runs two windows with waitBlanking=False on the second, so "
                f"the eyes are not guaranteed to be on the same frame. This is the "
                f"problem it looks like it solves. Merge the panels into one logical "
                f"display and use left/right, or verify the delay with a photodiode.")
        if self.mode == "dual-head":
            out.append(
                "dual-head additionally blits the right eye only after the left has "
                "already been presented, so it is strictly worse than two-window. There "
                "is no reason to prefer it; it is kept only for comparison.")
        if self._vergence_ignored:
            out.append(
                f"vergence_px was set but {self.mode} ignores it: psykit applies "
                f"fixationVergence in the left/right branch only. Bake the shift into "
                f"align_px instead.")

        w, h = (int(v) for v in self.wins[0].size)
        pw, ph = self.panel_pix
        if self.mode == "single-window":
            n = w / pw
            if abs(n - round(n)) > 1e-6:
                out.append(
                    f"window is {w}x{h}, not a whole number of {pw}-wide panels "
                    f"({n:.2f}). Either the merge did not take or Optics has the wrong "
                    f"panel width.")
            elif round(n) < 2:
                out.append(
                    f"window is {w}x{h}, only one panel wide. The displays are not "
                    f"merged yet, so both eyes are on one monitor.")
            elif max(self.eye_panels) >= round(n):
                out.append(
                    f"eye_panels={self.eye_panels} but the merged desktop is only "
                    f"{round(n)} panels wide.")
            if h != ph:
                out.append(f"window height {h} does not match the panel height {ph}.")
        elif self.mode in SPANNING_MODES:
            if (w, h) == (pw, ph):
                out.append(
                    f"window is {w}x{h}, exactly one panel, but a spanning mode needs "
                    f"{pw * 2}x{ph}. The two displays are not merged into one logical "
                    f"display yet, so both eyes are being squeezed onto one monitor.")
            elif w != pw * 2 or h != ph:
                out.append(
                    f"window is {w}x{h} but two {pw}x{ph} panels need {pw * 2}x{ph}. "
                    f"Optics is built on {pw}x{ph} per eye, so every angular quantity "
                    f"reported above is wrong by the ratio of those numbers.")
        elif (w, h) != (pw, ph):
            out.append(
                f"window opened at {w}x{h} but Optics is built on {pw}x{ph}. Every "
                f"angular quantity, disparity included, is scaled by {pw / max(w, 1):.3f} "
                f"in x. Fix the display mode or re-measure into Optics; do not collect "
                f"data on this.")

        if not self._monitor_known:
            out.append(
                "no saved monitor calibration was found, so PsychoPy made a temporary "
                "one. Without it there is no gamma correction and the nominal Michelson "
                "contrasts are not the contrasts on screen.")
        elif not self._gamma_known:
            out.append("monitor found but it carries no gamma grid; contrasts are nominal.")

        for eye, panel in self.panels.items():
            for name, span, off in (("x", panel.size[0], panel.pos[0]),
                                    ("y", panel.size[1], panel.pos[1])):
                edge = off - span / 2.0
                if abs(edge - round(edge)) > 1e-6:
                    out.append(
                        f"{eye} eye patch lands on a half pixel in {name} "
                        f"(edge at {edge:.1f}); nearest-neighbour sampling will snap it "
                        f"by a pixel and inject a spurious disparity.")
        return out

    def close(self):
        for win in self.wins:
            win.close()
