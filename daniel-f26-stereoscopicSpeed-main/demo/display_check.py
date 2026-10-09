"""What Windows, the GPU and PsychoPy each think the displays are.

Run on the rig, at the console (NOT over Remote Desktop):

    uv run python display_check.py

Three numbers have to agree before any of the stereo work means anything: the
panel's real pixel resolution, what PsychoPy sees, and what Optics is built on.
Windows DPI scaling routinely breaks the middle one, silently.
"""

import ctypes
import sys
from ctypes import wintypes

# Import BEFORE pyglet touches the displays: apparatus sets DPI awareness at import
# time, so the pyglet view below is the one the experiment will actually get. Pass
# --unaware to skip it and see the raw, scaled numbers Windows hands a naive process.
APPLY_DPI_FIX = "--unaware" not in sys.argv
DPI_RESULT = "skipped (--unaware)"
if APPLY_DPI_FIX:
    try:
        from src.apparatus import DPI_AWARENESS as DPI_RESULT
    except Exception as _e:                              # standalone copy, no src package
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            DPI_RESULT = "per-monitor"
        except Exception:
            DPI_RESULT = f"unavailable ({_e})"

ENUM_CURRENT_SETTINGS = -1
DISPLAY_DEVICE_ATTACHED_TO_DESKTOP = 0x1
DISPLAY_DEVICE_PRIMARY_DEVICE = 0x4


class DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD),
                ("DeviceName", wintypes.WCHAR * 32),
                ("DeviceString", wintypes.WCHAR * 128),
                ("StateFlags", wintypes.DWORD),
                ("DeviceID", wintypes.WCHAR * 128),
                ("DeviceKey", wintypes.WCHAR * 128)]


class DEVMODEW(ctypes.Structure):
    _fields_ = [("dmDeviceName", wintypes.WCHAR * 32),
                ("dmSpecVersion", wintypes.WORD),
                ("dmDriverVersion", wintypes.WORD),
                ("dmSize", wintypes.WORD),
                ("dmDriverExtra", wintypes.WORD),
                ("dmFields", wintypes.DWORD),
                ("dmPositionX", ctypes.c_long),
                ("dmPositionY", ctypes.c_long),
                ("dmDisplayOrientation", wintypes.DWORD),
                ("dmDisplayFixedOutput", wintypes.DWORD),
                ("dmColor", ctypes.c_short),
                ("dmDuplex", ctypes.c_short),
                ("dmYResolution", ctypes.c_short),
                ("dmTTOption", ctypes.c_short),
                ("dmCollate", ctypes.c_short),
                ("dmFormName", wintypes.WCHAR * 32),
                ("dmLogPixels", wintypes.WORD),
                ("dmBitsPerPel", wintypes.DWORD),
                ("dmPelsWidth", wintypes.DWORD),
                ("dmPelsHeight", wintypes.DWORD),
                ("dmDisplayFlags", wintypes.DWORD),
                ("dmDisplayFrequency", wintypes.DWORD),
                ("dmICMMethod", wintypes.DWORD),
                ("dmICMIntent", wintypes.DWORD),
                ("dmMediaType", wintypes.DWORD),
                ("dmDitherType", wintypes.DWORD),
                ("dmReserved1", wintypes.DWORD),
                ("dmReserved2", wintypes.DWORD),
                ("dmPanningWidth", wintypes.DWORD),
                ("dmPanningHeight", wintypes.DWORD)]


def windows_displays():
    """Real per-adapter display modes, straight from the driver. DPI-independent."""
    user32 = ctypes.windll.user32
    rows, i = [], 0
    while True:
        adapter = DISPLAY_DEVICEW()
        adapter.cb = ctypes.sizeof(adapter)
        if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(adapter), 0):
            break
        i += 1
        if not adapter.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP:
            continue
        monitor = DISPLAY_DEVICEW()
        monitor.cb = ctypes.sizeof(monitor)
        has_mon = user32.EnumDisplayDevicesW(adapter.DeviceName, 0,
                                             ctypes.byref(monitor), 0)
        dm = DEVMODEW()
        dm.dmSize = ctypes.sizeof(dm)
        ok = user32.EnumDisplaySettingsW(adapter.DeviceName, ENUM_CURRENT_SETTINGS,
                                         ctypes.byref(dm))
        rows.append(dict(
            device=adapter.DeviceName,
            gpu=adapter.DeviceString,
            monitor=monitor.DeviceString if has_mon else "?",
            primary=bool(adapter.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE),
            width=dm.dmPelsWidth if ok else None,
            height=dm.dmPelsHeight if ok else None,
            hz=dm.dmDisplayFrequency if ok else None,
            x=dm.dmPositionX if ok else None,
            y=dm.dmPositionY if ok else None))
    return rows


def dpi_state():
    """Process DPI awareness and the per-monitor scale factors."""
    out = {}
    try:
        aware = ctypes.c_int()
        ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(aware))
        out["awareness"] = {0: "UNAWARE (Windows lies about resolution)",
                            1: "SYSTEM_DPI_AWARE",
                            2: "PER_MONITOR_DPI_AWARE"}.get(aware.value, aware.value)
    except Exception as e:
        out["awareness"] = f"unavailable ({e})"
    try:
        user32 = ctypes.windll.user32
        out["GetSystemMetrics virtual desktop"] = (user32.GetSystemMetrics(78),
                                                   user32.GetSystemMetrics(79))
    except Exception as e:
        out["GetSystemMetrics virtual desktop"] = f"unavailable ({e})"
    return out


def pyglet_screens():
    """Exactly what PsychoPy's screen= indices will refer to."""
    import pyglet

    return [(i, s.x, s.y, s.width, s.height)
            for i, s in enumerate(pyglet.canvas.get_display().get_screens())]


def verdict(rows, screens):
    """Adjudicate. Returns a list of problems; empty means the display stack is sane."""
    bad = []
    if rows:
        rates = {r["hz"] for r in rows if r["hz"]}
        if len(rates) > 1:
            detail = ", ".join(f"{r['device'].split(chr(92))[-1]}={r['hz']}Hz" for r in rows)
            bad.append(f"panels refresh at different rates ({detail}). A dynamic "
                       f"stereogram cannot be fused under this; a static one is "
                       f"unaffected, which is why it hides.")
        gpus = {r["gpu"] for r in rows}
        if len(gpus) > 1:
            bad.append(f"displays span {len(gpus)} GPUs {sorted(gpus)}; panels on "
                       f"different cards can never be frame-synced.")
    if rows and screens:
        by_pos = {(r["x"], r["y"]): r for r in rows}
        for i, x, y, w, h in screens:
            m = by_pos.get((x, y))
            if m is None:
                bad.append(f"pyglet screen={i} at ({x},{y}) matches no driver display.")
            elif (w, h) != (m["width"], m["height"]):
                bad.append(f"pyglet screen={i} reports {w}x{h} but the panel is "
                           f"{m['width']}x{m['height']}: DPI scaling is still biting.")
        want = (max(r["x"] + r["width"] for r in rows) - min(r["x"] for r in rows),
                max(r["y"] + r["height"] for r in rows) - min(r["y"] for r in rows))
        try:
            got = (ctypes.windll.user32.GetSystemMetrics(78),
                   ctypes.windll.user32.GetSystemMetrics(79))
            if got != want:
                bad.append(f"virtual desktop reports {got} but the panels span {want}; "
                           f"mixed-DPI queries disagree. Set Scale to 100% everywhere.")
        except Exception:
            pass
    return bad


def main():
    print(f"python {sys.version.split()[0]} on {sys.platform}\n")
    rows, screens = [], []

    if sys.platform != "win32":
        print("Windows-only sections skipped (run this on the rig).")
    else:
        print("=" * 78)
        print("WINDOWS / DRIVER VIEW  (authoritative pixel resolution)")
        print("=" * 78)
        rows = windows_displays()
        for r in rows:
            star = " *PRIMARY*" if r["primary"] else ""
            print(f"  {r['device']}{star}")
            print(f"      GPU      : {r['gpu']}")
            print(f"      monitor  : {r['monitor']}")
            print(f"      mode     : {r['width']}x{r['height']} @ {r['hz']} Hz  "
                  f"at desktop position ({r['x']}, {r['y']})")
        gpus = {r["gpu"] for r in rows}
        if len(gpus) > 1:
            print(f"\n  !! displays are split across {len(gpus)} GPUs: {sorted(gpus)}")
            print("     Two panels on different GPUs can never be frame-synced, and")
            print("     no merge technology can group them. Both haploscope panels")
            print("     must be on the same card.")

        print("\n" + "=" * 78)
        print("DPI SCALING  (the usual reason PsychoPy sees the wrong size)")
        print("=" * 78)
        print(f"  ensure_dpi_aware() applied: {DPI_RESULT}")
        for k, v in dpi_state().items():
            print(f"  {k}: {v}")
        span = (max(r["x"] + r["width"] for r in rows) - min(r["x"] for r in rows),
                max(r["y"] + r["height"] for r in rows) - min(r["y"] for r in rows))
        print(f"  panels physically span: {span}  (this is what the line above should say)")

    print("\n" + "=" * 78)
    print("PSYCHOPY / PYGLET VIEW  (what screen= indexes, and what fullscr uses)")
    print("=" * 78)
    try:
        screens = pyglet_screens()
        for i, x, y, w, h in screens:
            print(f"  screen={i}: {w}x{h} at ({x}, {y})")
    except Exception as e:
        print(f"  pyglet unavailable: {e}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    problems = verdict(rows, screens)
    if not problems:
        print("  PASS - every panel agrees between the driver and pyglet, all panels")
        print("  share one refresh rate, and all are on one GPU. Put the driver-view")
        print("  resolution into Optics(size_pix=...) and measure the active area in cm")
        print("  so that Optics.pixel_aspect comes out at 1.000.")
    else:
        for p in problems:
            print(f"  FAIL - {p}")
        print()
        print("  Fix these before collecting data. Scaling: set Scale to 100% on the")
        print("  stimulus panels. Refresh: Display Settings -> Advanced display ->")
        print("  Choose a refresh rate, and if the rate you want is absent, suspect the")
        print("  cable, the port or the EDID rather than the dropdown.")


if __name__ == "__main__":
    main()
