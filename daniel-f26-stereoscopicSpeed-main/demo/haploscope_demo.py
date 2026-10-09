"""Drift a cyclopean square-wave grating on a haploscope.

Troughs sit in the background plane with the surround; only the peaks protrude.
"""

from src.apparatus import Optics, StereoDisplay
from src.stimuli import CyclopeanGrating


def displayDRDS(display, stim, duration=None, bank=None):
    """Drift `stim` on `display` for `duration` s (None = until ESC).

    If `bank` is a prerendered list of (left, right) pairs, frames are taken from
    it in order and nothing is computed inside the loop. Returns True if the full
    duration elapsed, False if ESC cut it short.
    """
    from psychopy import core, event

    clock = core.Clock()
    t_last = 0.0
    i = 0
    event.clearEvents()
    while duration is None or clock.getTime() < duration:
        if event.getKeys(keyList=["escape"]):
            return False
        if bank is not None:
            left, right = bank[i % len(bank)]
            i += 1
        else:
            t_now = clock.getTime()
            stim.advance(t_now - t_last)    # wall-clock dt: speed is frame-rate independent
            t_last = t_now
            left, right = stim.render()
        display.draw(right, left)
        display.flip()
    return True


def run(stim, optics, duration=None, prerender_s=None, frame_rate=60.0, **kwargs):
    """Open the stereo window, show `stim`, and close it again."""
    bank = None
    if prerender_s:
        n = int(round(prerender_s * frame_rate))
        print(f"prerendering {n} frames ...")
        bank = stim.prerender(n, 1.0 / frame_rate)
    display = StereoDisplay(stim, optics, **kwargs)
    for w in display.warnings():
        print(f"WARNING: {w}")
    try:
        return displayDRDS(display, stim, duration, bank)
    finally:
        print(display.timing_report())
        display.close()


if __name__ == "__main__":
    # ---- one haploscope arm (identical for both eyes) ----
    SIZE_PIX = (1920, 1080)
    SIZE_CM = (34.0, 19.125)    # measured ACTIVE AREA of one panel; gives pixel_aspect 1.000
    DISTANCE_CM = 80.0          # total optical path, eye -> mirror -> screen

    # ---- display ----
    # 'single-window': one plain window over the merged desktop, each eye on its own
    #   panel. One flip, no interocular delay, and it works with a merged desktop of
    #   any width. Set WINDOW_PIX to the merged size and EYE_PANELS to the panel
    #   index each eye looks at. Recommended.
    # 'left/right': psykit, same guarantee, but the merged desktop must be exactly
    #   two panels wide because psykit splits the window in half.
    # 'two-window': two plain windows, second free-running. Checks that the stimulus
    #   renders; not for data.
    # 'dual-head': psykit's own two-window path. Strictly worse than 'two-window'.
    STEREO_MODE = "two-window"
    SCREEN = 0                  # index of the MERGED display
    SCREEN2 = 1                 # second panel; 'two-window' and 'dual-head' only
    WINDOW_PIX = None           # merged desktop size, e.g. (3840, 1080) or (5760, 1080);
                                #   None = assume exactly two panels wide
    EYE_PANELS = (0, 1)         # panel index for (left eye, right eye), 0 = leftmost
    ALIGN_PX = ((0, 0), (0, 0)) # per-eye centring offsets, observer coordinates
    VERGENCE_PX = 0.0           # global inward shift, for nulling vergence demand
    FUSION_LOCK = False         # strongly recommended: see the amplitude note below
    PHOTODIODE = False          # corner patch alternating each frame, for timing checks

    # ---- dot field ----
    FIELD_DEG = 6             # deg: side of the square dot patch (aperture + surround)
    DOT_PX = 2                  # screen pixels per dot (no longer the disparity quantum)
    APERTURE_DEG = 3.0          # deg: aperture diameter on the retina
    RANDOMIZE_EACH_FRAME = False

    # ---- grating ----
    # Troughs sit at PEDESTAL (0 = coplanar with the surround). Peaks protrude by
    # AMPLITUDE. There is no symmetric excursion about a mean any more, so
    # AMPLITUDE is exactly the peak-to-trough depth, not half of it.
    AMPLITUDE_ARCMIN = 0.0      # peak protrusion above the background plane, crossed
    PEDESTAL_ARCMIN = 0.0       # disparity of the troughs; leave at 0
    DUTY = 0.5                  # fraction of a cycle occupied by the protruding bars
    PERIOD_DEG = 2.0            # spatial period of the corrugation
    SPEED_DEG_S = 0.0           # -ve = leftward, in observer coordinates

    # ---- rendering ----
    SUBPIXEL = True             # exact disparity via anti-aliased shifts
    CONTRAST_OUT = 0.5          # Michelson contrast of troughs and surround
    CONTRAST_IN = 0.5           # Michelson contrast of the bars
    DURATION = None             # seconds, or None to run until ESC
    PRERENDER_S = None          # e.g. 2.0 to build the trial up front; None = live
    FRAME_RATE = 60.0
    MONITOR_NAME = "haploscope" # a Monitor Center name carrying the gamma calibration

    optics = Optics(size_pix=SIZE_PIX, size_cm=SIZE_CM, distance_cm=DISTANCE_CM)
    stim = CyclopeanGrating.from_degrees(
        optics, field_deg=FIELD_DEG, dot_px=DOT_PX, aperture_deg=APERTURE_DEG,
        amplitude_arcmin=AMPLITUDE_ARCMIN, pedestal_arcmin=PEDESTAL_ARCMIN, duty=DUTY,
        period_deg=PERIOD_DEG, speed_deg_s=SPEED_DEG_S, seed=42,
        randomize_each_frame=RANDOMIZE_EACH_FRAME, subpixel=SUBPIXEL,
        contrast_out=CONTRAST_OUT, contrast_in=CONTRAST_IN)

    print(f"{optics.px_per_deg:.2f} px/deg, 1 px = {optics.px2arcmin(1):.3f} arcmin")
    print(stim.report())
    for w in optics.warnings() + stim.warnings():
        print(f"WARNING: {w}")

    run(stim, optics, duration=DURATION, prerender_s=PRERENDER_S, frame_rate=FRAME_RATE,
        mode=STEREO_MODE, screen=SCREEN, screen2=SCREEN2, mirrors_per_eye=1,
        align_px=ALIGN_PX, vergence_px=VERGENCE_PX, fusion_lock=FUSION_LOCK,
        photodiode=PHOTODIODE, full_screen=True, monitor_name=MONITOR_NAME,
        window_pix=WINDOW_PIX, eye_panels=EYE_PANELS)
