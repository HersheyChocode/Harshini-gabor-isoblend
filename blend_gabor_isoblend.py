#!/usr/bin/env python3
"""
Plaid & Gabor-Julesz Isoluminance Weighted Sum Generator
========================================================
Vision Science & Psychophysics Signal-in-Noise Utility

Combines an orthogonal Plaid pattern with a texture pattern that strictly follows
the isoluminance curves of the plaid, differing across bands in Orientation,
Spatial Frequency, and Color.

Includes support for relative spatial displacement (shift_x, shift_y) between
the plaid carrier and the isoluminance texture pattern.

    Plaid(x, y)   = 0.5 * [ 1 + env(x, y) * C_plaid * (C1 * cos(omega1 * x'_1) + C2 * cos(omega2 * x'_2)) ]
    Texture(x, y) = T_k(x - dx, y - dy) where T_{k-1} <= Plaid(x - dx, y - dy) < T_k
    Blend(x, y)   = clip[ w_plaid * Plaid(x, y) + w_texture * Texture(x, y), 0.0, 1.0 ]

Supports zero-dependency pure Python (native 24-bit BMP) and vectorized NumPy acceleration.
"""

import argparse
import math
import struct
import sys
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class PlaidIsoBlender:
    """
    Synthesizes weighted combinations of Plaid patterns and isoluminance-following
    texture patterns with factorial cue dimension control and spatial shifting.
    """

    PALETTES = {
        "isoluminant": [
            (244, 63, 94),   # Rose
            (245, 158, 11),  # Amber
            (16, 185, 129),  # Emerald
            (6, 182, 212),   # Cyan
            (99, 102, 241),  # Indigo
            (168, 85, 247),  # Purple
            (236, 72, 153),  # Pink
        ],
        "spectral": [
            (168, 85, 247), (59, 130, 246), (20, 184, 166), (34, 197, 94),
            (234, 179, 8), (249, 115, 22), (239, 68, 68)
        ],
        "opponent": [
            (59, 130, 246), (234, 179, 8), (239, 68, 68), (34, 197, 94)
        ],
        "monochrome": [
            (230, 230, 230), (180, 180, 180), (130, 130, 130), (80, 80, 80)
        ]
    }

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        w_plaid: float = 0.50,
        w_texture: float = 0.50,
        plaid_contrast: float = 1.0,
        # Relative Spatial Shift (dx, dy)
        shift_x: int = 0,
        shift_y: int = 0,
        periodic_wrap: bool = True,
        # Control Pattern Mode
        random_mixture_control: bool = False, #aligned to plaid
        mixture_seed: int = 1960,
         # Plaid Wave 1
        c1: float = 0.50,
        theta1_deg: float = 45.0,
        freq1: float = 6.0,
        phase1_deg: float = 0.0,
        # Plaid Wave 2 (Orthogonal)
        c2: float = 0.50,
        theta2_deg: float = 135.0,
        freq2: float = 6.0,
        phase2_deg: float = 0.0,
        # Spatial Envelope
        sigma: float = 150.0,
        # Isoluminance Texture Controls
        num_bands: int = 5,
        texture_mode: str = "gabor",  # 'gabor', 'continuous', 'bands'
        vary_orientation: bool = True,
        ori_scheme: str = "ortho",    # 'ortho', 'stepped', 'tangent'
        ori_jitter_deg: float = 10.0,
        const_ori_deg: float = 0.0,
        vary_frequency: bool = False,
        freq_scheme: str = "divergent", # 'divergent', 'monotonic', 'alternating'
        const_freq: float = 28.0,
        vary_color: bool = False,
        palette_name: str = "isoluminant",
        const_color: Tuple[int, int, int] = (225, 225, 225),
        density: int = 28,
        random_phase: bool = False,
    ):
        self.width = int(width)
        self.height = int(height)
        self.w_plaid = float(w_plaid)
        self.w_texture = float(w_texture)
        self.plaid_contrast = max(0.0, min(1.0, float(plaid_contrast)))

        self.shift_x = int(shift_x)
        self.shift_y = int(shift_y)
        self.periodic_wrap = bool(periodic_wrap)
        self.random_mixture_control = bool(random_mixture_control)
        self.mixture_seed = int(mixture_seed)

        self.c1 = float(c1)
        self.theta1_deg = float(theta1_deg)
        self.freq1 = float(freq1)
        self.phase1_deg = float(phase1_deg)

        self.c2 = float(c2)
        self.theta2_deg = float(theta2_deg)
        self.freq2 = float(freq2)
        self.phase2_deg = float(phase2_deg)

        self.sigma = float(sigma)

        self.num_bands = max(2, int(num_bands))
        self.texture_mode = str(texture_mode).lower()
        self.vary_orientation = bool(vary_orientation)
        self.ori_scheme = str(ori_scheme).lower()
        self.ori_jitter_deg = float(ori_jitter_deg)
        self.const_ori_deg = float(const_ori_deg)

        self.vary_frequency = bool(vary_frequency)
        self.freq_scheme = str(freq_scheme).lower()
        self.const_freq = float(const_freq)

        self.vary_color = bool(vary_color)
        self.palette = self.PALETTES.get(palette_name, self.PALETTES["isoluminant"])
        self.const_color = const_color

        self.density = max(8, int(density))
        self.random_phase = bool(random_phase)

    def _get_band_specs(self) -> List[dict]:
        """Builds parameter definitions for each band. returns list of dictionaries."""
        bands = []
        K = self.num_bands
        for k in range(K):
            t_low = k / K
            t_high = (k + 1) / K
            t_mid = 0.5 * (t_low + t_high)
            col = self.palette[k % len(self.palette)] if self.vary_color else self.const_color

            if self.vary_frequency:
                if self.freq_scheme == "divergent":
                    sf = 12.0 + 24.0 * abs(t_mid - 0.5) * 2.0
                elif self.freq_scheme == "monotonic":
                    sf = 10.0 + (k / (K - 1)) * 26.0
                else:
                    sf = 12.0 if (k % 2 == 0) else 28.0
            else:
                sf = self.const_freq

            if self.vary_orientation:
                if self.ori_scheme == "ortho":
                    avg_angle = 0.0 if (k % 2 == 0) else (math.pi / 2.0) #pi/2 radians aka 90 degrees
                elif self.ori_scheme == "stepped":
                    avg_angle = (k / K) * math.pi
                else:
                    avg_angle = 0.0
            else:
                avg_angle = math.radians(self.const_ori_deg)

            bands.append({
                "k": k, "t_low": t_low, "t_high": t_high,
                "color": col, "sf": sf, "avg_angle": avg_angle
            })
        return bands

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates (blend_rgb, plaid_rgb, texture_rgb) as uint8 arrays [0, 255].
        """
        return self._generate_numpy()

    def _generate_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        W = self.width
        H = self.height
        x = np.linspace(-W / 2.0, W / 2.0, W, endpoint=False, dtype=np.float32)
        y = np.linspace(-H / 2.0, H / 2.0, H, endpoint=False, dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        # 1. Base Plaid Field
        th1 = np.deg2rad(self.theta1_deg)
        th2 = np.deg2rad(self.theta2_deg)
        ph1 = np.deg2rad(self.phase1_deg)
        ph2 = np.deg2rad(self.phase2_deg)

        u1 = 2.0 * np.pi * (self.freq1 / W) * (X * np.cos(th1) + Y * np.sin(th1)) + ph1
        u2 = 2.0 * np.pi * (self.freq2 / W) * (X * np.cos(th2) + Y * np.sin(th2)) + ph2

        env = 1.0
        if self.sigma < 140.0:
            env = np.exp(-(X**2 + Y**2) / (2.0 * self.sigma**2))

        plaid_norm = np.clip(0.5 * (1.0 + env * self.plaid_contrast * (self.c1 * np.cos(u1) + self.c2 * np.cos(u2))), 0.0, 1.0)
        plaid_rgb = np.repeat(np.round(plaid_norm * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        # 2. Base Texture Field
        bands = self._get_band_specs()
        tex_rgb = np.zeros((H, W, 3), dtype=np.float32)

        if self.texture_mode == "continuous":
            if self.random_mixture_control:
                rng_mix = np.random.default_rng(self.mixture_seed)
                mix_cell = max(12, int(round(W / float(self.density))))
                grid_y = (H + mix_cell - 1) // mix_cell
                grid_x = (W + mix_cell - 1) // mix_cell
                cell_bands = rng_mix.integers(0, len(bands), size=(grid_y, grid_x))
                yy = np.arange(H) // mix_cell
                xx = np.arange(W) // mix_cell
                YY, XX = np.meshgrid(yy, xx, indexing='ij')
                cell_map = cell_bands[YY, XX]
                for k, b in enumerate(bands):
                    mask = (cell_map == k)
                    if not np.any(mask):
                        continue
                    theta = b["avg_angle"]
                    sf = b["sf"]
                    carrier = 0.5 + 0.45 * np.cos(2.0 * np.pi * (sf / W) * (X * np.cos(theta) + Y * np.sin(theta)))
                    for c in range(3):
                        tex_rgb[mask, c] = b["color"][c] * carrier[mask]
            else:
                for b in bands:
                    mask = (plaid_norm >= b["t_low"]) & (plaid_norm < b["t_high"])
                    if not np.any(mask):
                        continue
                    theta = b["avg_angle"]
                    sf = b["sf"]
                    carrier = 0.5 + 0.45 * np.cos(2.0 * np.pi * (sf / W) * (X * np.cos(theta) + Y * np.sin(theta)))
                    for c in range(3):
                        tex_rgb[mask, c] = b["color"][c] * carrier[mask]
        elif self.texture_mode == "bands":
            if self.random_mixture_control:
                rng_mix = np.random.default_rng(self.mixture_seed)
                mix_cell = max(12, int(round(W / float(self.density))))
                grid_y = (H + mix_cell - 1) // mix_cell
                grid_x = (W + mix_cell - 1) // mix_cell
                cell_bands = rng_mix.integers(0, len(bands), size=(grid_y, grid_x))
                yy = np.arange(H) // mix_cell
                xx = np.arange(W) // mix_cell
                YY, XX = np.meshgrid(yy, xx, indexing='ij')
                cell_map = cell_bands[YY, XX]
                for k, b in enumerate(bands):
                    mask = (cell_map == k)
                    for c in range(3):
                        tex_rgb[mask, c] = b["color"][c]
            else:
                for b in bands:
                    mask = (plaid_norm >= b["t_low"]) & (plaid_norm < b["t_high"])
                    for c in range(3):
                        tex_rgb[mask, c] = b["color"][c]
        else:
            rng = np.random.default_rng(self.mixture_seed if self.random_mixture_control else 1960)
            step = W / float(self.density)
            patch_sigma = step * 0.38
            two_sig_sq = 2.0 * patch_sigma**2
            weight_acc = np.zeros((H, W), dtype=np.float32)
            jitter_rad = np.deg2rad(self.ori_jitter_deg) if self.vary_orientation else 0.0

            coords = np.linspace(-W / 2.0 + step / 2, W / 2.0 - step / 2, self.density)
            for cy in coords:
                for cx in coords:
                    px = cx + rng.uniform(-0.35, 0.35) * step
                    py = cy + rng.uniform(-0.35, 0.35) * step

                    if self.random_mixture_control:
                        band = bands[int(rng.integers(0, len(bands)))]
                    else:
                        ix = int(np.clip(px + W / 2, 0, W - 1))
                        iy = int(np.clip(py + H / 2, 0, H - 1))
                        p_val = plaid_norm[iy, ix]

                        band = bands[-1]
                        for b in bands:
                            if p_val < b["t_high"]:
                                band = b
                                break

                    theta = band["avg_angle"] + (rng.uniform(-jitter_rad, jitter_rad) if jitter_rad > 0 else 0.0)
                    sf = band["sf"]
                    phase = rng.uniform(0, 2 * np.pi) if self.random_phase else 0.0

                    dx = X - px
                    dy = Y - py
                    dist_sq = dx**2 + dy**2
                    mask = dist_sq < (3.0 * patch_sigma)**2
                    if not np.any(mask):
                        continue

                    x_rot = dx[mask] * np.cos(theta) + dy[mask] * np.sin(theta)
                    env_patch = np.exp(-dist_sq[mask] / two_sig_sq)
                    carrier = 0.5 + 0.45 * np.cos(2.0 * np.pi * (sf / W) * x_rot + phase)
                    gabor = env_patch * carrier

                    weight_acc[mask] += env_patch
                    for c in range(3):
                        tex_rgb[mask, c] += gabor * band["color"][c]

            nz = weight_acc > 1e-4
            for c in range(3):
                tex_rgb[nz, c] /= weight_acc[nz]

        # 3. Apply Relative Spatial Shift
        if self.shift_x != 0 or self.shift_y != 0:
            tex_rgb = np.roll(np.roll(tex_rgb, self.shift_x, axis=1), self.shift_y, axis=0)

        tex_uint8 = np.clip(tex_rgb, 0, 255).astype(np.uint8)

        # 4. Weighted Combination
        wP = self.w_plaid
        wT = self.w_texture
        blend_rgb = np.clip(wP * plaid_rgb.astype(np.float32) + wT * tex_rgb, 0, 255).astype(np.uint8)

        return blend_rgb, plaid_rgb, tex_uint8


    def save_bmp(self, filename: str, rgb_data=None):
        """Exports 24-bit uncompressed Windows BMP without external dependencies."""
        if rgb_data is None:
            blend_rgb, _, _ = self.generate()
            rgb_data = blend_rgb

        W = self.width
        H = self.height
        row_stride = (W * 3 + 3) & ~3
        img_size = row_stride * H
        file_size = 54 + img_size

        bmp_header = struct.pack(
            "<2sIHHI",
            b"BM", file_size, 0, 0, 54
        )
        dib_header = struct.pack(
            "<IIIHHIIIIII",
            40, W, H, 1, 24, 0, img_size, 2835, 2835, 0, 0
        )

        with open(filename, "wb") as f:
            f.write(bmp_header)
            f.write(dib_header)
            pad = b"\x00" * (row_stride - W * 3)

            if HAS_NUMPY and isinstance(rgb_data, np.ndarray):
                for y in range(H - 1, -1, -1):
                    row = rgb_data[y, :, ::-1].tobytes()  # BGR
                    f.write(row)
                    if pad: f.write(pad)
            else:
                for y in range(H - 1, -1, -1):
                    row = bytearray()
                    for x in range(W):
                        idx = (y * W + x) * 3
                        row.extend((rgb_data[idx + 2], rgb_data[idx + 1], rgb_data[idx]))
                    f.write(row)
                    if pad: f.write(pad)


def main():
    parser = argparse.ArgumentParser(description="Plaid & Gabor-Julesz Isoluminance Pattern Blender")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("-wP", "--w-plaid", type=float, default=0.50, help="Plaid weight [0.0-1.0]")
    parser.add_argument("-wT", "--w-texture", type=float, default=0.50, help="Texture weight [0.0-1.0]")
    parser.add_argument("-c", "--contrast", type=float, default=1.0, help="Master plaid contrast [0.0-1.0]")
    parser.add_argument("--shift-x", type=int, default=0, help="Horizontal spatial shift in pixels (default: 0)")
    parser.add_argument("--shift-y", type=int, default=0, help="Vertical spatial shift in pixels (default: 0)")
    parser.add_argument("--bands", type=int, default=5, help="Number of isoluminance bands (default: 5)")
    parser.add_argument("--mode", choices=["gabor", "continuous", "bands"], default="gabor", help="Texture synthesis mode")
    parser.add_argument("--vary-ori", action="store_true", default=True, help="Vary orientation across bands")
    parser.add_argument("--no-vary-ori", dest="vary_ori", action="store_false")
    parser.add_argument("--vary-freq", action="store_true", default=False, help="Vary frequency across bands")
    parser.add_argument("--vary-col", action="store_true", default=False, help="Vary color across bands")
    parser.add_argument("--control-mixture", action="store_true", default=False,
                        help="Generate spatially homogeneous random mixture control pattern (uncorrelated with plaid)")
    parser.add_argument("--seed", type=int, default=1960, help="Random seed for mixture control pattern (default: 1960)")
    parser.add_argument("-o", "--output", type=str, default="isoblend_stimulus.bmp", help="Output filename")

    args = parser.parse_args()

    blender = PlaidIsoBlender(
        width=args.width,
        height=args.height,
        w_plaid=args.w_plaid,
        w_texture=args.w_texture,
        plaid_contrast=args.contrast,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        random_mixture_control=args.control_mixture,
        mixture_seed=args.seed,
        num_bands=args.bands,
        texture_mode=args.mode,
        vary_orientation=args.vary_ori,
        vary_frequency=args.vary_freq,
        vary_color=args.vary_col,
    )

    mode_str = "Random Mixture Control" if args.control_mixture else f"Isoluminance-Aligned (shift={args.shift_x}, {args.shift_y})"
    print(f"Synthesizing Plaid-Isoluminance Blend ({args.width}x{args.height}) [{mode_str}]...")
    #blender.save_bmp(args.output)
    print(f"Saved stimulus to {args.output}")

    blend_rgb, _, _ = blender.generate()
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(blend_rgb)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
