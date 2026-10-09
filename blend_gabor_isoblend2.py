#!/usr/bin/env python3
"""
Plaid & Gabor-Julesz Isoluminance Weighted Sum Generator
========================================================
Python port of index.html. Every step below mirrors the JavaScript render()
pipeline (same defaults, same pixel conventions, same quantization, same PRNG).

    Plaid(x, y)   = 0.5 * [ 1 + env * C_plaid * (c1*cos(k1*u1 + ph1) + c2*cos(k2*u2 + ph2)) ]   (float, clamped)
    Texture(x, y) = T_k(x - dx, y - dy)  where  t_low_k <= Plaid(x, y) < t_high_k               (8-bit, like the canvas)
    Blend(x, y)   = round(255 * clip(w_plaid * Plaid + w_texture * Texture/255, 0, 1))

Requires numpy (+ matplotlib for display).
"""

import argparse
import math
import struct
from typing import List, Tuple
from datetime import datetime
import random
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


def js_round(a):
    """JavaScript Math.round (round half up). np.round is half-to-even and would differ."""
    return np.floor(np.asarray(a, dtype=np.float64) + 0.5)


class PlaidIsoBlender:

    # Same palettes / names as index.html
    PALETTES = {
        "chroma-isolum": [
            (244, 63, 94), (245, 158, 11), (16, 185, 129), (6, 182, 212), (99, 102, 241),
            (168, 85, 247), (236, 72, 153), (234, 179, 8), (34, 197, 94),
        ],
        "spectral": [
            (168, 85, 247), (59, 130, 246), (20, 184, 166), (34, 197, 94), (234, 179, 8),
            (249, 115, 22), (239, 68, 68), (217, 70, 239), (99, 102, 241),
        ],
        "opponent": [
            (59, 130, 246), (234, 179, 8), (239, 68, 68), (34, 197, 94),
            (59, 130, 246), (234, 179, 8),
        ],
        "warm-cool": [
            (239, 68, 68), (249, 115, 22), (234, 179, 8), (20, 184, 166),
            (59, 130, 246), (99, 102, 241), (168, 85, 247),
        ],
    }
    PALETTES["isoluminant"] = PALETTES["chroma-isolum"]  # backwards-compatible alias

    CONST_COLORS = {
        "white": (225, 225, 225),
        "cyan": (56, 189, 248),
        "amber": (251, 191, 36),
        "green": (52, 211, 153),
    }

    GABOR_BG = (11, 15, 25)  # '#0b0f19', the canvas fill behind the Gabor patches in index.html

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        w_plaid: float = 0.50,
        w_texture: float = 0.50,
        plaid_contrast: float = 1.0,
        # Relative shift of texture vs plaid (pixels)
        shift_x: int = 0,
        shift_y: int = 0,
        periodic_wrap: bool = True,          # True: wrap around, False: clamp to edge
        # Control condition: texture bands assigned randomly, uncorrelated with plaid
        random_mixture_control: bool = False,
        mixture_seed: int = 1960,            # seed for the control pattern's random choices
        # Plaid wave 1 (freq is in cycles across the image WIDTH)
        c1: float = 0.50,
        theta1_deg: float = 45.0,
        freq1: float = 6.0,
        phase1_deg: float = 0.0,
        # Plaid wave 2
        c2: float = 0.50,
        theta2_deg: float = 135.0,
        freq2: float = 6.0,
        phase2_deg: float = 0.0,
        # Gaussian envelope (>= 140 means no envelope)
        sigma: float = 150.0,
        # Texture
        num_bands: int = 5,
        texture_mode: str = "gabor",         # 'gabor', 'continuous', 'bands'
        vary_orientation: bool = False,
        ori_scheme: str = "ortho",           # 'ortho', 'stepped', 'tangent'
        ori_jitter_deg: float = 10.0,
        const_ori_deg: float = 0.0,
        vary_frequency: bool = False,
        freq_scheme: str = "alternating",      # 'divergent', 'monotonic', 'alternating'
        const_freq: float = 18.0,
        vary_color: bool = False,
        palette_name: str = "chroma-isolum",
        const_color="white",                 # name in CONST_COLORS or an (r, g, b) tuple
        density: int = 28,
        random_phase: bool = True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.w_plaid = float(w_plaid)
        self.w_texture = float(w_texture)
        self.plaid_contrast = float(plaid_contrast)

        self.shift_x = int(shift_x)
        self.shift_y = int(shift_y)
        self.periodic_wrap = bool(periodic_wrap)
        self.random_mixture_control = bool(random_mixture_control)
        self.mixture_seed = int(mixture_seed)

        self.c1, self.theta1_deg, self.freq1, self.phase1_deg = float(c1), float(theta1_deg), float(freq1), float(phase1_deg)
        self.c2, self.theta2_deg, self.freq2, self.phase2_deg = float(c2), float(theta2_deg), float(freq2), float(phase2_deg)
        self.sigma = float(sigma)

        self.num_bands = int(num_bands)
        self.texture_mode = str(texture_mode).lower()
        self.vary_orientation = bool(vary_orientation)
        self.ori_scheme = str(ori_scheme).lower()
        self.ori_jitter_deg = float(ori_jitter_deg)
        self.const_ori_deg = float(const_ori_deg)

        self.vary_frequency = bool(vary_frequency)
        self.freq_scheme = str(freq_scheme).lower()
        self.const_freq = float(const_freq)

        self.vary_color = bool(vary_color)
        self.palette = self.PALETTES.get(palette_name, self.PALETTES["chroma-isolum"])
        self.const_color = (self.CONST_COLORS.get(const_color, self.CONST_COLORS["white"])
                            if isinstance(const_color, str) else tuple(const_color))
        self.density = int(density)
        self.random_phase = bool(random_phase)

    # ------------------------------------------------------------------ bands
    def _get_band_specs(self) -> List[dict]:
        """Mirror of getBandProperties()."""
        K = self.num_bands
        bands = []
        for k in range(K):
            t_low, t_high = k / K, (k + 1) / K
            t_mid = 0.5 * (t_low + t_high)
            col = self.palette[k % len(self.palette)] if self.vary_color else self.const_color

            if self.vary_frequency:
                if self.freq_scheme == "divergent":
                    sf = 12.0 + 24 * abs(t_mid - 0.5) * 2.0
                elif self.freq_scheme == "monotonic":
                    sf = 10.0+ (k / (K - 1)) * 26.0
                else:
                    sf = 12.0 if k % 2 == 0 else 28.0
            else:
                sf = self.const_freq

            is_tangent = False
            if self.vary_orientation:
                if self.ori_scheme == "ortho":
                    avg_angle = 0.0 if k % 2 == 0 else math.pi / 2
                elif self.ori_scheme == "stepped":
                    avg_angle = (k / K) * math.pi
                else:
                    is_tangent, avg_angle = True, 0.0
            else:
                avg_angle = math.radians(self.const_ori_deg)

            bands.append(dict(k=k, t_low=t_low, t_high=t_high, color=col, sf=sf,
                              avg_angle=avg_angle, is_tangent=is_tangent))
        return bands

    def _band_index_from_plaid(self, p, bands):
        """First band with p < t_high, else the last band (same as the JS loop)."""
        idx = np.full(np.shape(p), len(bands) - 1, dtype=np.int64)
        for b in reversed(bands):
            idx = np.where(p < b["t_high"], b["k"], idx)
        return idx

    # ------------------------------------------------------------------ plaid
    def _compute_plaid(self) -> np.ndarray:
        """Mirror of computePlaidField(): float64, clamped to [0, 1], NOT quantized."""
        W, H = self.width, self.height
        nx = np.arange(W, dtype=np.float64) - W / 2.0
        ny = np.arange(H, dtype=np.float64) - H / 2.0
        X, Y = np.meshgrid(nx, ny)

        k1 = 2 * math.pi * (self.freq1 / W)
        k2 = 2 * math.pi * (self.freq2 / W)
        th1, th2 = math.radians(self.theta1_deg), math.radians(self.theta2_deg)
        ph1, ph2 = math.radians(self.phase1_deg), math.radians(self.phase2_deg)

        u1 = X * math.cos(th1) + Y * math.sin(th1)
        u2 = X * math.cos(th2) + Y * math.sin(th2)
        w1 = np.cos(k1 * u1 + ph1)
        w2 = np.cos(k2 * u2 + ph2)

        env = 1.0
        if self.sigma < 140.0:
            env = np.exp(-(X * X + Y * Y) / (2.0 * self.sigma * self.sigma))

        raw = 0.5 * (1.0 + env * self.plaid_contrast * (self.c1 * w1 + self.c2 * w2))
        return np.clip(raw, 0.0, 1.0)

    # ---------------------------------------------------------- mixture helper
    def _mix_cell_map(self, bands) -> np.ndarray:
        """Mirror of the per-cell hash used by the 'continuous' and 'bands' modes in control-mixture mode."""
        W, H = self.width, self.height
        mix_cell = max(12, int(js_round(W / self.density)))
        cy = (np.arange(H) // mix_cell).astype(np.int64)[:, None]
        cx = (np.arange(W) // mix_cell).astype(np.int64)[None, :]
        s = cy * 19349663 + cx * 83492791 + self.mixture_seed
        s = ((s + 2**31) % 2**32) - 2**31                 # JS ToInt32
        s = s ^ 0x5BF03635                                 # int32 XOR
        return np.abs(np.fmod(s, len(bands)))              # JS % keeps dividend sign, then Math.abs

    # ----------------------------------------------------------- tangent angle
    @staticmethod
    def _tangent_theta(plaid) -> np.ndarray:
        """Per-pixel angle perpendicular to the plaid gradient (clamped central differences, like the JS)."""
        gx = np.empty_like(plaid)
        gx[:, 1:-1] = plaid[:, 2:] - plaid[:, :-2]
        gx[:, 0] = plaid[:, 1] - plaid[:, 0]
        gx[:, -1] = plaid[:, -1] - plaid[:, -2]
        gy = np.empty_like(plaid)
        gy[1:-1, :] = plaid[2:, :] - plaid[:-2, :]
        gy[0, :] = plaid[1, :] - plaid[0, :]
        gy[-1, :] = plaid[-1, :] - plaid[-2, :]
        return np.arctan2(gy, gx) + math.pi / 2

    # ------------------------------------------------------------- textures
    def _texture_continuous_or_bands(self, plaid, bands, carrier_on: bool) -> np.ndarray:
        W, H = self.width, self.height
        if self.random_mixture_control:
            band_idx = self._mix_cell_map(bands)
        else:
            band_idx = self._band_index_from_plaid(plaid, bands)

        nx = (np.arange(W, dtype=np.float64) - W / 2.0) / W
        ny = (np.arange(H, dtype=np.float64) - H / 2.0) / W      # NOTE: JS divides y by W too
        NX, NY = np.meshgrid(nx, ny)

        tan_theta = self._tangent_theta(plaid) if (carrier_on and not self.random_mixture_control) else None

        tex = np.zeros((H, W, 3), dtype=np.float64)
        for b in bands:
            mask = band_idx == b["k"]
            if not mask.any():
                continue
            if carrier_on:
                if b["is_tangent"] and tan_theta is not None:
                    theta = tan_theta
                else:
                    theta = b["avg_angle"]
                carrier = np.cos(2 * math.pi * b["sf"] * (NX * np.cos(theta) + NY * np.sin(theta)))
                intensity = 0.5 + 0.45 * carrier
                val = intensity[mask]
            else:
                val = 1.0
            for c in range(3):
                tex[..., c][mask] = js_round(b["color"][c] * val) if carrier_on else b["color"][c]
        return np.clip(tex, 0, 255)

    def _texture_gabor(self, plaid, bands) -> np.ndarray:
        """
        Mirror of renderGaborTextonTexture() + drawGaborPatch().
        The HTML draws alpha-blended, round-capped line strokes on a canvas; here each stroke is
        rasterized analytically (capsule distance -> anti-aliased coverage) and composited
        source-over in the same order with the same LCG random stream.
        """
        W, H = self.width, self.height
        canvas = np.empty((H, W, 3), dtype=np.float64)
        canvas[:] = self.GABOR_BG

        density = self.density
        step = W / density
        patch_r = step * 0.95
        jitter_rad = math.radians(self.ori_jitter_deg) if self.vary_orientation else 0.0
        is_mix = self.random_mixture_control

        seed = [self.mixture_seed if is_mix else 1960]

        def rnd():
            seed[0] = (seed[0] * 9301 + 49297) % 233280
            return seed[0] / 233280

        for gy in range(-1, density + 2):
            for gx in range(-1, density + 2):
                jx = (rnd() - 0.5) * step * 0.45
                jy = (rnd() - 0.5) * step * 0.45
                cx = (gx + math.fmod(gy, 2) * 0.5) * step + jx     # fmod: JS % (gy=-1 -> -1)
                cy = gy * step + jy

                if cx < -patch_r or cx > W + patch_r or cy < -patch_r or cy > H + patch_r:
                    continue

                ix = int(max(0, min(W - 1, math.floor(cx))))
                iy = int(max(0, min(H - 1, math.floor(cy))))

                if is_mix:
                    band = bands[int(math.floor(rnd() * len(bands)))]
                else:
                    p_val = plaid[iy, ix]
                    band = bands[-1]
                    for b in bands:
                        if p_val < b["t_high"]:
                            band = b
                            break

                base_theta = band["avg_angle"]
                if band["is_tangent"] and not is_mix:
                    gxv = plaid[iy, min(W - 1, ix + 1)] - plaid[iy, max(0, ix - 1)]
                    gyv = plaid[min(H - 1, iy + 1), ix] - plaid[max(0, iy - 1), ix]
                    base_theta = math.atan2(gyv, gxv) + math.pi / 2

                theta = base_theta + ((rnd() - 0.5) * 2 * jitter_rad if jitter_rad > 0 else 0.0)
                phase = rnd() * 2 * math.pi if self.random_phase else 0.0
                f = (band["sf"] / W) * 2.5

                self._draw_gabor_patch(canvas, cx, cy, patch_r, theta, f, phase, band["color"])

        return canvas

    @staticmethod
    def _draw_gabor_patch(canvas, cx, cy, radius, theta, f, phase, col):
        H, W, _ = canvas.shape
        sigma = radius * 0.38
        two_sig_sq = 2 * sigma * sigma
        bar_spacing = 1 / (f or 0.05)
        num_bars = 4
        line_w = max(1.5, bar_spacing * 0.42)

        ext = radius + line_w / 2 + 2
        x0, x1 = max(0, int(math.floor(cx - ext))), min(W, int(math.ceil(cx + ext)))
        y0, y1 = max(0, int(math.floor(cy - ext))), min(H, int(math.ceil(cy + ext)))
        if x0 >= x1 or y0 >= y1:
            return

        # pixel centres in canvas coordinates, in the patch's rotated frame
        px = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5 - cx
        py = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5 - cy
        c, s = math.cos(theta), math.sin(theta)
        U = px * c + py * s          # along the grating normal
        T = -px * s + py * c         # along the bar

        region = canvas[y0:y1, x0:x1]
        colv = np.array(col, dtype=np.float64)


        
        for b in range(-num_bars, num_bars + 1):
            u = b * bar_spacing + (phase / (2 * math.pi)) * bar_spacing
            if abs(u) > radius:
                continue
            #gauss = math.exp(-(u * u) / two_sig_sq)
            #if gauss < 0.05:
            #    continue
            bar_len = math.sqrt(max(0.0, radius * radius - u * u)) * 1.8

            dist = np.hypot(U - u, np.maximum(np.abs(T) - bar_len / 2, 0.0))
            cover = np.clip(line_w / 2 - dist + 0.5, 0.0, 1.0)
            #a = (cover * (gauss * 0.9))[..., None]
            a = cover[..., None]
            region[:] = js_round(region * (1 - a) + colv * a)    # 8-bit canvas after every stroke

    # -------------------------------------------------------------- generate
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (blend_rgb, plaid_rgb, texture_rgb) as uint8 (H, W, 3)."""
        W, H = self.width, self.height
        plaid = self._compute_plaid()
        bands = self._get_band_specs()

        if self.texture_mode == "continuous":
            tex = self._texture_continuous_or_bands(plaid, bands, carrier_on=True)
        elif self.texture_mode == "bands":
            tex = self._texture_continuous_or_bands(plaid, bands, carrier_on=False)
        else:
            tex = self._texture_gabor(plaid, bands)

        # Relative spatial shift: out(x, y) = tex(x - dx, y - dy)
        ys = np.arange(H) - self.shift_y
        xs = np.arange(W) - self.shift_x
        if self.periodic_wrap:
            ys, xs = (ys % H + H) % H, (xs % W + W) % W
        else:
            ys, xs = np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)
        tex = tex[ys][:, xs]
        tex_u8 = np.clip(tex, 0, 255).astype(np.uint8)

        # Weighted blend in [0, 1] using the UNquantized plaid and the 8-bit texture
        blend = np.clip(self.w_plaid * plaid[..., None] + self.w_texture * (tex_u8.astype(np.float64) / 255.0), 0.0, 1.0)
        blend_u8 = js_round(blend * 255).astype(np.uint8)

        plaid_u8 = np.repeat(js_round(plaid * 255).astype(np.uint8)[..., None], 3, axis=2)
        return blend_u8, plaid_u8, tex_u8

    # ------------------------------------------------------------- overlays
    def contour_segments(self):
        """Marching-squares iso-contour segments, same 80x80 sampling as drawIsoluminanceContours()."""
        W, H = self.width, self.height
        plaid = self._compute_plaid()
        bands = self._get_band_specs()
        RES = 80
        step = W / (RES - 1)
        sy = np.floor(np.arange(RES) / (RES - 1) * (H - 1)).astype(int)
        sx = np.floor(np.arange(RES) / (RES - 1) * (W - 1)).astype(int)
        g = plaid[sy][:, sx]
        segs = []
        for b in range(len(bands) - 1):
            level = bands[b]["t_high"]
            for j in range(RES - 1):
                y0, y1 = j * step, (j + 1) * step
                for i in range(RES - 1):
                    x0, x1 = i * step, (i + 1) * step
                    v0, v1, v2, v3 = g[j, i], g[j, i + 1], g[j + 1, i + 1], g[j + 1, i]
                    m = (1 if v0 >= level else 0) | (2 if v1 >= level else 0) | (4 if v2 >= level else 0) | (8 if v3 >= level else 0)
                    if m in (0, 15):
                        continue
                    t = lambda a, c: max(0.0, min(1.0, (level - a) / ((c - a) or 1e-6)))
                    pT = (x0 + t(v0, v1) * step, y0)
                    pR = (x1, y0 + t(v1, v2) * step)
                    pB = (x0 + (1 - t(v2, v3)) * step, y1)
                    pL = (x0, y0 + (1 - t(v3, v0)) * step)
                    if m in (1, 14): segs.append((pL, pT))
                    elif m in (2, 13): segs.append((pT, pR))
                    elif m in (3, 12): segs.append((pL, pR))
                    elif m in (4, 11): segs.append((pR, pB))
                    elif m in (6, 9): segs.append((pT, pB))
                    elif m in (7, 8): segs.append((pL, pB))
        return segs

    # ------------------------------------------------------------------ BMP
    def save_bmp(self, filename: str, rgb_data=None):
        """Exports 24-bit uncompressed Windows BMP."""
        if rgb_data is None:
            rgb_data = self.generate()[0]
        W, H = self.width, self.height
        row_stride = (W * 3 + 3) & ~3
        img_size = row_stride * H
        with open(filename, "wb") as f:
            f.write(struct.pack("<2sIHHI", b"BM", 54 + img_size, 0, 0, 54))
            f.write(struct.pack("<IIIHHIIIIII", 40, W, H, 1, 24, 0, img_size, 2835, 2835, 0, 0))
            pad = b"\x00" * (row_stride - W * 3)
            for y in range(H - 1, -1, -1):
                f.write(rgb_data[y, :, ::-1].tobytes() + pad)

def randomize_values():
    #w_plaid = random.random()
    #c1 = random.random()
    #freq1 = random.randint(4)
    """
    return dict(#w_plaid = w_plaid,
            #w_texture = 1 - w_plaid,
            #plaid_contrast = random.random(),
            #c1 = c1,
            #freq1 = freq1,
            #c2 = c1,
            #freq2 = freq1,
            #texture_mode = ["gabor", "continuous", "bands"][random.randint(0,2)],
            #density = random.randint(16,42),
            random_phase = bool(random.getrandbits(1)),
            random_mixture_control = bool(random.getrandbits(1)),
            num_bands = random.randint(3,9),
            vary_frequency = bool(random.getrandbits(1)),
            vary_orientation = bool(random.getrandbits(1)),
            vary_color = bool(random.getrandbits(1)),
            #ori_scheme = ["ortho", "stepped", "tangent"][random.randint(0,2)],
            ori_jitter_deg = random.randint(0,44),
        )
        """
    freq1 = 2
    return dict(#w_plaid = w_plaid,
                #w_texture = 1 - w_plaid,
                #plaid_contrast = 0.5,
                #c1 = c1,
                freq1 = 0,
                #theta2_deg = 90,
                #c2 = c1,
                freq2 = 2,
                texture_mode = "gabor",
                #texture_mode = ["gabor", "continuous", "bands"][random.randint(0,2)],
                density = 16,
                #random_phase = bool(random.getrandbits(1)),
                random_mixture_control = False,
                num_bands = 5,
                vary_frequency = True,
                vary_orientation = True,
                #vary_color = bool(random.getrandbits(1)),
                #ori_scheme = ["ortho", "stepped", "tangent"][random.randint(0,2)],
                ori_jitter_deg = 0
            )
    
def main():
    if not HAS_MATPLOTLIB:
        raise SystemExit("matplotlib is required for PNG output / display (or use -o file.bmp)")

    index = 1 
    values = randomize_values()
    with open("values.log", "a", encoding="utf-8") as f:
        f.write(str(datetime.now()) + "\n")
        f.write(str(index) + ". " + str(values) + "\n")

    blender = PlaidIsoBlender(**values)
    blend_rgb, _, _ = blender.generate()

    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(blend_rgb, interpolation="nearest", extent=(0, 512, 512, 0))
    ax.set_xlim(0, 512); ax.set_ylim(512, 0); ax.axis("off")

    def on_key(event):
        nonlocal index
        if event.key == "right":
            index += 1
            # Generate new image
            new_values = randomize_values()
            with open("values.log", "a", encoding="utf-8") as f:
                f.write(str(index) + ". " + str(new_values) + "\n")
            new_blender = PlaidIsoBlender(**new_values)
            new_rgb, _, _ = new_blender.generate()

            # Update existing image data & re-render canvas
            im.set_data(new_rgb)
            fig.canvas.draw_idle()

        elif event.key == " ":
            # Close window and exit on Spacebar
            plt.close(fig)

    # 4. Connect event listener
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    

def main2():
    p = argparse.ArgumentParser(description="Plaid & Gabor-Julesz Isoluminance Pattern Blender (port of index.html)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("-wP", "--w-plaid", type=float, default=0.50)
    p.add_argument("-wT", "--w-texture", type=float, default=0.50)
    p.add_argument("-c", "--contrast", type=float, default=1.0)
    p.add_argument("--shift-x", type=int, default=0)
    p.add_argument("--shift-y", type=int, default=0)
    p.add_argument("--no-wrap", dest="wrap", action="store_false", help="Clamp instead of wrapping the shift")
    p.add_argument("--bands", type=int, default=5)
    p.add_argument("--mode", choices=["gabor", "continuous", "bands"], default="gabor")
    p.add_argument("--vary-ori", action="store_true", default=True)
    p.add_argument("--no-vary-ori", dest="vary_ori", action="store_false")
    p.add_argument("--ori-scheme", choices=["ortho", "stepped", "tangent"], default="ortho")
    p.add_argument("--vary-freq", action="store_true", default=False)
    p.add_argument("--vary-col", action="store_true", default=False)
    p.add_argument("--control-mixture", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=1960)
    p.add_argument("--contours", action="store_true", help="Overlay iso-contours (index.html 'Show contours')")
    p.add_argument("--crosshairs", action="store_true", help="Overlay centre crosshairs")
    p.add_argument("-o", "--output", type=str, default=None, help="Save PNG/BMP instead of showing a window")
    a = p.parse_args()

    blender = PlaidIsoBlender(
        width=a.width, height=a.height, w_plaid=a.w_plaid, w_texture=a.w_texture,
        plaid_contrast=a.contrast, shift_x=a.shift_x, shift_y=a.shift_y, periodic_wrap=a.wrap,
        random_mixture_control=a.control_mixture, mixture_seed=a.seed, num_bands=a.bands,
        texture_mode=a.mode, vary_orientation=a.vary_ori, ori_scheme=a.ori_scheme,
        vary_frequency=a.vary_freq, vary_color=a.vary_col,
    )
    blend_rgb, _, _ = blender.generate()

    if a.output and a.output.lower().endswith(".bmp"):
        blender.save_bmp(a.output, blend_rgb)
        print(f"Saved {a.output}")
        return

    if not HAS_MATPLOTLIB:
        raise SystemExit("matplotlib is required for PNG output / display (or use -o file.bmp)")

    fig = plt.figure(figsize=(a.width / 100, a.height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(blend_rgb, interpolation="nearest", extent=(0, a.width, a.height, 0))
    if a.contours and not a.control_mixture and a.mode != "bands":
        for (x0, y0), (x1, y1) in blender.contour_segments():
            ax.plot([x0, x1], [y0, y1], color=(1, 1, 1, 0.4), lw=0.7)
    if a.crosshairs:
        ax.axvline(a.width / 2, color=(56 / 255, 189 / 255, 248 / 255, 0.35), lw=0.7, ls=(0, (4, 4)))
        ax.axhline(a.height / 2, color=(56 / 255, 189 / 255, 248 / 255, 0.35), lw=0.7, ls=(0, (4, 4)))
    ax.set_xlim(0, a.width); ax.set_ylim(a.height, 0); ax.axis("off")

    if a.output:
        fig.savefig(a.output, dpi=100)
        print(f"Saved {a.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
