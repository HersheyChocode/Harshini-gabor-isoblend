"""Cyclopean square-wave disparity grating on a random-dot carrier.

Disparity convention
--------------------
The trough of the square wave sits in the *background plane*, coplanar with the
random-dot surround outside the aperture, and the peaks protrude toward the
observer by exactly `amplitude`. There is no symmetric excursion about a mean:

    disparity(x) = pedestal + amplitude * square01(x)        square01 in {0, 1}

so trough disparity == pedestal (0 by default, i.e. the surround plane) and peak
disparity == pedestal + amplitude. `pedestal` exists only so the whole corrugated
patch can be pushed off the surround plane deliberately; leave it at 0 for the
"only the peaks protrude" stimulus.

Units
-----
`H` and `W` are the dot lattice, in dots. Everything else spatial (`amplitude`,
`pedestal`, `period`, `speed`, `radius_x`, `radius_y`) is in SCREEN PIXELS, not
dots. This is deliberate: it decouples the disparity quantum from the dot size,
which halves the quantum at dot_px=2 and costs nothing. Use `from_degrees` to
build from visual angle and never touch the pixel numbers directly.

Sub-pixel rendering
-------------------
With `subpixel=True` the requested disparity is realised exactly by rendering at
the two bracketing integer-pixel shifts and blending them with the fractional
part. Dots stop being binary at depth edges (they acquire a one-pixel
anti-aliased ramp) but the disparity you asked for is the disparity you get,
which matters when the amplitude is only a few pixels. With `subpixel=False`
dots stay binary and the disparity is rounded to whole pixels; `report()` then
tells you what was actually drawn, which is the number belonging in a methods
section.
"""

import numpy as np


def to_signed(img, contrast, region=None, region_contrast=None):
    """Binary {0,1} -> PsychoPy signed grey: white -> +c, black -> -c about mean grey.

    `contrast` applies everywhere except where `region` is True, which gets
    `region_contrast` (used to give the bars a different contrast from the surround).
    """
    c = np.full(img.shape, contrast, dtype=np.float32)
    if region is not None and region_contrast is not None:
        c[region] = region_contrast
    return (img.astype(np.float32) * 2.0 - 1.0) * c


class CyclopeanGrating:
    """Random-dot field carrying a drifting square-wave disparity grating.

    Pure NumPy: no window needed. The aperture takes separate x and y radii so a
    retinally circular aperture can be drawn on a panel with non-square pixels.
    """

    def __init__(self, H=120, W=180, dot_px=2, amplitude=3.0, period=160.0,
                 speed=80.0, pedestal=0.0, duty=0.5, radius_x=120.0, radius_y=None,
                 randomize_each_frame=True, contrast_out=0.5, contrast_in=0.5,
                 subpixel=True, seed=0, optics=None):
        self.H, self.W, self.dot_px = H, W, dot_px      # dots, dots, screen px per dot
        self.Hpx, self.Wpx = H * dot_px, W * dot_px     # screen pixels
        self.amplitude = float(amplitude)               # px: protrusion of the peaks
        self.pedestal = float(pedestal)                 # px: disparity of the troughs
        self.period = float(period)                     # px: spatial period
        self.speed = float(speed)                       # px/s: sign = direction
        self.duty = float(duty)                         # fraction of a cycle that is peak
        self.radius_x = float(radius_x)                 # px
        self.radius_y = float(radius_x if radius_y is None else radius_y)
        self.randomize_each_frame = randomize_each_frame
        self.contrast_out = contrast_out                # Michelson: troughs and surround
        self.contrast_in = contrast_in                  # Michelson: bars
        self.subpixel = subpixel
        self.optics = optics                            # for reporting only
        self.position = 0.0                             # px: current drift position

        if not 0.0 < self.duty < 1.0:
            raise ValueError("duty must be strictly between 0 and 1")

        # pixel-centre coordinates relative to the patch centre
        self._xs = np.arange(self.Wpx) + 0.5 - self.Wpx / 2.0
        ys = np.arange(self.Hpx) + 0.5 - self.Hpx / 2.0
        X, Y = np.meshgrid(self._xs, ys)
        self.aperture = (X / self.radius_x) ** 2 + (Y / self.radius_y) ** 2 <= 1.0
        self._in_aperture_col = self.aperture.any(axis=0)
        self._outside = ~self.aperture

        self._rng = np.random.default_rng(seed)
        self._fixed = self._draw(np.random.default_rng(seed))
        self._current = self._draw(self._rng)

    # ------------------------------------------------------------------ build

    @classmethod
    def from_degrees(cls, optics, field_deg=4.5, dot_px=2, aperture_deg=3.0,
                     amplitude_arcmin=3.0, period_deg=2.0, speed_deg_s=1.0,
                     pedestal_arcmin=0.0, duty=0.5, **kwargs):
        """Build from visual angle. `aperture_deg` and `field_deg` are diameters/sides.

        Disparities and all horizontal extents use the horizontal pitch; only the
        field height and the aperture's vertical radius use the vertical one.
        """
        W = int(round(optics.deg2px(field_deg) / dot_px))
        H = int(round(optics.deg2px(field_deg, vertical=True) / dot_px))
        return cls(H=H, W=W, dot_px=dot_px,
                   amplitude=optics.arcmin2px(amplitude_arcmin),
                   pedestal=optics.arcmin2px(pedestal_arcmin),
                   period=optics.deg2px(period_deg),
                   speed=optics.deg2px(speed_deg_s),
                   duty=duty,
                   radius_x=optics.deg2px(aperture_deg) / 2.0,
                   radius_y=optics.deg2px(aperture_deg, vertical=True) / 2.0,
                   optics=optics, **kwargs)

    # ------------------------------------------------------------------ field

    def _draw(self, rng):
        """(cyclopean field, left fill field, right fill field), each (Hpx, Wpx) binary.

        Dots are generated on the (H, W) lattice and replicated up to screen pixels,
        so a dot is dot_px x dot_px regardless of how finely disparity is shifted.
        """
        return tuple(
            (rng.random((self.H, self.W)) < 0.5).astype(np.uint8)
            .repeat(self.dot_px, axis=0).repeat(self.dot_px, axis=1)
            for _ in range(3))

    def field(self, refresh):
        """This frame's (C, fill_left, fill_right); frozen field if not randomizing."""
        if not self.randomize_each_frame:
            return self._fixed
        if refresh:
            self._current = self._draw(self._rng)
        return self._current

    # ------------------------------------------------------------- the profile

    def square01(self):
        """0/1 square wave along x at the current position, shape (Wpx,).

        1 on the peaks (the protruding bars), 0 in the troughs. No np.sign, so
        there is no third state at the zero crossings.
        """
        phase = ((self._xs - self.position) / self.period) % 1.0
        return (phase < self.duty).astype(np.float64)

    def disparity_row(self):
        """Unrounded disparity along x, in screen pixels, shape (Wpx,).

        Troughs sit at `pedestal` (0 = the surround plane); peaks protrude to
        `pedestal + amplitude`.
        """
        return self.pedestal + self.amplitude * self.square01()

    def advance(self, dt):
        """Move the grating by speed * dt (dt in seconds)."""
        self.position += self.speed * dt

    # --------------------------------------------------------------- readback

    @property
    def rendered_peak_px(self):
        """Peak disparity actually drawn, in screen pixels."""
        if self.subpixel:
            return self.pedestal + self.amplitude
        return float(np.rint(self.pedestal + self.amplitude))

    @property
    def rendered_trough_px(self):
        """Trough disparity actually drawn, in screen pixels."""
        return self.pedestal if self.subpixel else float(np.rint(self.pedestal))

    @property
    def rendered_amplitude_px(self):
        """Peak-minus-trough disparity actually drawn, in screen pixels."""
        return self.rendered_peak_px - self.rendered_trough_px

    @property
    def temporal_freq(self):
        """Drift temporal frequency, Hz."""
        return self.speed / self.period

    def report(self):
        """One-line summary of what will actually be drawn, in stimulus units."""
        o = self.optics
        if o is None:
            return (f"{self.Wpx}x{self.Hpx} px, trough {self.rendered_trough_px:.2f} px, "
                    f"peak {self.rendered_peak_px:.2f} px, drift {self.temporal_freq:+.2f} Hz")
        want = o.px2arcmin(self.amplitude)
        got = o.px2arcmin(self.rendered_amplitude_px)
        err = "" if self.subpixel else f" [requested {want:.2f}', error {got - want:+.2f}']"
        return (f"field {o.px2deg(self.Wpx):.2f} x {o.px2deg(self.Hpx, vertical=True):.2f} deg, "
                f"aperture {2 * o.px2deg(self.radius_x):.2f} deg wide, "
                f"trough {o.px2arcmin(self.rendered_trough_px):+.2f}' (background plane), "
                f"peak {o.px2arcmin(self.rendered_peak_px):+.2f}', "
                f"protrusion {got:.2f}'{err}, "
                f"duty {self.duty:.2f}, quantum {o.px2arcmin(1):.2f}'/px, "
                f"period {o.px2deg(self.period):.2f} deg, "
                f"drift {o.px2deg(self.speed):+.2f} deg/s ({self.temporal_freq:+.2f} Hz)")

    def warnings(self):
        """List of things that will quietly corrupt the stimulus. Empty is good."""
        out = []
        o = self.optics
        if not self.subpixel:
            err = abs(self.rendered_amplitude_px - self.amplitude)
            if err > 0.01:
                pct = 100 * err / max(self.amplitude, 1e-9)
                out.append(f"amplitude rounded to whole pixels: off by {pct:.1f}%. "
                           f"Set subpixel=True or report the rendered value.")
            if self.amplitude < 3.0:
                out.append(f"amplitude is {self.amplitude:.2f} px, under 3 quantisation "
                           f"steps. Binary-dot rendering cannot resolve this cleanly.")
        if self.pedestal == 0.0 and self.amplitude > 0:
            mean = self.amplitude * self.duty
            txt = f"{o.px2arcmin(mean):.2f}'" if o else f"{mean:.2f} px"
            out.append(f"troughs are in the surround plane, so mean disparity inside the "
                       f"aperture is {txt} crossed. Use a fusion lock so vergence has a "
                       f"zero-disparity reference to hold.")
        if self.Wpx % 2 or self.Hpx % 2:
            out.append(f"patch is {self.Wpx}x{self.Hpx} px (odd side): centring it on an "
                       f"even-sized window lands on a half pixel and resamples.")
        return out

    # --------------------------------------------------------------- painting

    def _paint(self, C, fill, shift, frac, sign, bars=None):
        """Paint C into one eye. Column c lands at c + sign * shift[c].

        Vectorised over rows by column slicing rather than np.nonzero, which is
        what makes pixel-resolution painting affordable.

        Returns (img, region, frac_map): the binary image, the mask of pixels
        belonging to the protruding bars, and the per-pixel fractional shift
        carried along for sub-pixel blending.
        """
        Hpx, Wpx = self.Hpx, self.Wpx
        mask, sur = self.aperture, self._outside
        if bars is None:
            bars = mask & (self.square01() > 0)[None, :]

        img = np.empty((Hpx, Wpx), np.uint8)
        region = np.zeros((Hpx, Wpx), bool)
        covered = np.zeros((Hpx, Wpx), bool)
        frac_map = np.zeros((Hpx, Wpx), np.float32)

        def put(cols, s):
            if cols.size == 0:
                return
            tgt = cols + sign * s
            ok = (tgt >= 0) & (tgt < Wpx)
            cols, tgt = cols[ok], tgt[ok]
            if cols.size == 0:
                return
            m = mask[:, cols]
            img[:, tgt] = np.where(m, C[:, cols], img[:, tgt])
            region[:, tgt] |= m & bars[:, cols]
            frac_map[:, tgt] = np.where(m, frac[cols][None, :], frac_map[:, tgt])
            covered[:, tgt] |= m

        levels = np.unique(shift[self._in_aperture_col])
        for s in levels[levels < 0]:                    # far columns first
            put(np.where(self._in_aperture_col & (shift == s))[0], s)

        img[sur] = C[sur]                               # surround, fixation plane
        region[sur] = False
        frac_map[sur] = 0.0
        covered[sur] = True

        for s in levels[levels >= 0]:                   # near columns on top
            put(np.where(self._in_aperture_col & (shift == s))[0], s)

        img[~covered] = fill[~covered]                  # half-occlusions: fresh dots
        return img, region, frac_map

    def _eye_shifts(self, d):
        """Split the total disparity profile `d` (px) into (left, right) eye shifts.

        Sub-pixel: exact halves, so both eyes get identical treatment and the
        anti-aliasing is symmetric. Integer: the total is rounded first and then
        split, so odd totals survive; rounding each eye separately would
        quantise the total to even pixels and halve the resolution again.
        """
        if self.subpixel:
            half = d / 2.0
            return half, d - half
        n = np.rint(d)
        left = np.floor(n / 2.0)
        return left, n - left

    def _render_eye(self, C, fill, d_eye, sign, bars):
        """Signed-grey image for one eye from its continuous shift profile (px)."""
        if not self.subpixel:
            n = np.rint(d_eye).astype(np.int32)
            z = np.zeros(self.Wpx, np.float32)
            img, region, _ = self._paint(C, fill, n, z, sign, bars)
            return to_signed(img, self.contrast_out, region, self.contrast_in)

        # Bracket this eye's shift with the two nearest integer pixel shifts and
        # blend by the fractional part, so the realised disparity is exact.
        lo = np.floor(d_eye).astype(np.int32)
        frac = (d_eye - lo).astype(np.float32)
        a, ra, w = self._paint(C, fill, lo, frac, sign, bars)
        b, rb, _ = self._paint(C, fill, lo + 1, frac, sign, bars)
        sa = to_signed(a, self.contrast_out, ra, self.contrast_in)
        sb = to_signed(b, self.contrast_out, rb, self.contrast_in)
        return sa * (1.0 - w) + sb * w

    def pair(self, refresh=True):
        """(left, right, bars_left, bars_right) binary images at screen resolution.

        Nearest-integer-pixel shifts, no sub-pixel blending. Kept for inspection
        and for tests; `render` is what the experiment should call.
        """
        C, fill_l, fill_r = self.field(refresh)
        s01 = self.square01()
        bars = self.aperture & (s01 > 0)[None, :]
        d = self.pedestal + self.amplitude * s01
        n = np.rint(d)
        sl = np.floor(n / 2.0).astype(np.int32)
        sr = (n - sl).astype(np.int32)
        z = np.zeros(self.Wpx, np.float32)
        L, bl, _ = self._paint(C, fill_l, sl, z, +1, bars)
        R, br, _ = self._paint(C, fill_r, sr, z, -1, bars)
        return L, R, bl, br

    def render(self, refresh=True):
        """(left, right) signed-grey images ready for an ImageStim."""
        C, fill_l, fill_r = self.field(refresh)
        s01 = self.square01()
        bars = self.aperture & (s01 > 0)[None, :]
        dl, dr = self._eye_shifts(self.pedestal + self.amplitude * s01)
        return (self._render_eye(C, fill_l, dl, +1, bars),
                self._render_eye(C, fill_r, dr, -1, bars))

    # -------------------------------------------------------------- prerender

    def prerender(self, n_frames, dt):
        """Render `n_frames` ahead into a list of (left, right) float32 pairs.

        Live rendering costs a few ms per frame, which is a large fraction of a
        60 Hz budget and risks dropping frames. Dropping a frame on one eye and
        not the other destroys binocular correspondence in a dynamic stereogram,
        so for anything timing-critical build the trial up front and let the
        loop do nothing but upload and flip.
        """
        bank, pos0 = [], self.position
        for _ in range(n_frames):
            bank.append(self.render())
            self.advance(dt)
        self.position = pos0
        return bank
