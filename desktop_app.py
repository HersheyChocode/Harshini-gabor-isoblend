"""
Standalone Desktop Application for Plaid & Gabor-Julesz Isoluminance Pattern Blender
===================================================================================
Provides interactive Tkinter desktop GUI with full control over:
  - Plaid Carrier Waves (Orientations, Frequencies, Contrasts)
  - Relative Spatial Displacement / Shift (Shift X, Shift Y)
  - Isoluminance Texture Partitioning & Cue Dimension Controls (Orientation, Frequency, Color)
  - Texture Synthesis Modes (Gabor Textons, Continuous Carrier, Iso-Bands)
  - Real-time Canvas Preview and Export options

Zero external dependencies: Built with Python's standard library (tkinter).
"""

import math
import os
import sys
import webbrowser

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:
    print("Error: Tkinter is required to run the desktop application.")
    sys.exit(1)

from blend_gabor_isoblend import PlaidIsoBlender, HAS_NUMPY

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def matrix_to_ppm_bytes(rgb_data, width: int, height: int) -> bytes:
    """Converts RGB uint8 data into binary PPM (P6) bytes for Tkinter PhotoImage."""
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    if HAS_NUMPY and hasattr(rgb_data, "shape"):
        return header + rgb_data.tobytes()
    else:
        return header + bytes(rgb_data)


class PlaidIsoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plaid & Isoluminance Pattern Blender - Psychophysics Desktop App")
        self.geometry("1180x820")
        self.minsize(960, 660)
        self.configure(bg="#0f172a")

        # State Variables
        self.width_val = 512
        self.height_val = 512
        self.w_plaid = tk.DoubleVar(value=0.50)
        self.w_texture = tk.DoubleVar(value=0.50)
        self.plaid_contrast = tk.DoubleVar(value=1.0)

        # Spatial Shift
        self.shift_x = tk.IntVar(value=0)
        self.shift_y = tk.IntVar(value=0)

        # Plaid Waves
        self.theta1 = tk.DoubleVar(value=45.0)
        self.freq1 = tk.DoubleVar(value=24.0)
        self.c1 = tk.DoubleVar(value=0.50)

        self.theta2 = tk.DoubleVar(value=135.0)
        self.freq2 = tk.DoubleVar(value=24.0)
        self.c2 = tk.DoubleVar(value=0.50)
        self.lock_ortho = tk.BooleanVar(value=True)

        # Texture Controls
        self.num_bands = tk.IntVar(value=5)
        self.texture_mode = tk.StringVar(value="gabor")
        self.vary_ori = tk.BooleanVar(value=True)
        self.vary_freq = tk.BooleanVar(value=False)
        self.vary_col = tk.BooleanVar(value=False)
        self.density = tk.IntVar(value=26)
        self.random_mixture_control = tk.BooleanVar(value=False)
        self.mixture_seed = tk.IntVar(value=1960)

        self._build_ui()
        self.after(100, self.update_preview)

    def _build_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Controls Panel
        ctrl_frame = ttk.LabelFrame(main_frame, text=" Controls ")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        canvas_ctrl = tk.Canvas(ctrl_frame, width=360, highlightthickness=0, bg="#1e293b")
        scrollbar = ttk.Scrollbar(ctrl_frame, orient=tk.VERTICAL, command=canvas_ctrl.yview)
        scrollable_frame = ttk.Frame(canvas_ctrl)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_ctrl.configure(scrollregion=canvas_ctrl.bbox("all"))
        )
        canvas_ctrl.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_ctrl.configure(yscrollcommand=scrollbar.set)

        canvas_ctrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. Blend Weights
        w_box = ttk.LabelFrame(scrollable_frame, text=" Weighted Blend ")
        w_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(w_box, text="Plaid Weight (wP):").pack(anchor="w", padx=6, pady=1)
        ttk.Scale(w_box, from_=0.0, to=1.0, variable=self.w_plaid, command=self._on_wp_change).pack(fill=tk.X, padx=6)
        ttk.Label(w_box, text="Texture Weight (wT):").pack(anchor="w", padx=6, pady=1)
        ttk.Scale(w_box, from_=0.0, to=1.0, variable=self.w_texture, command=self._on_wt_change).pack(fill=tk.X, padx=6)
        ttk.Label(w_box, text="Master Plaid Contrast:").pack(anchor="w", padx=6, pady=1)
        ttk.Scale(w_box, from_=0.0, to=1.0, variable=self.plaid_contrast, command=lambda e: self.update_preview()).pack(fill=tk.X, padx=6)

        # 2. Relative Spatial Shift
        shift_box = ttk.LabelFrame(scrollable_frame, text=" Relative Spatial Shift (ΔX, ΔY) ")
        shift_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(shift_box, text="Horizontal Shift (ΔX px):").pack(anchor="w", padx=6)
        ttk.Scale(shift_box, from_=-128, to=128, variable=self.shift_x, command=lambda e: self.update_preview()).pack(fill=tk.X, padx=6)
        ttk.Label(shift_box, text="Vertical Shift (ΔY px):").pack(anchor="w", padx=6)
        ttk.Scale(shift_box, from_=-128, to=128, variable=self.shift_y, command=lambda e: self.update_preview()).pack(fill=tk.X, padx=6)
        ttk.Button(shift_box, text="Zero Shift", command=self._reset_shift).pack(anchor="e", padx=6, pady=3)

        # 3. Plaid Waves
        plaid_box = ttk.LabelFrame(scrollable_frame, text=" Plaid Carrier Components ")
        plaid_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(plaid_box, text="Wave 1 Orientation (deg):").pack(anchor="w", padx=6)
        ttk.Scale(plaid_box, from_=0, to=360, variable=self.theta1, command=self._on_theta1_change).pack(fill=tk.X, padx=6)
        ttk.Label(plaid_box, text="Wave 1 Frequency (cyc):").pack(anchor="w", padx=6)
        ttk.Scale(plaid_box, from_=1, to=24, variable=self.freq1, command=self._on_freq1_change).pack(fill=tk.X, padx=6)

        ttk.Checkbutton(plaid_box, text="Lock Wave 2 Orthogonal", variable=self.lock_ortho, command=self._on_lock_ortho).pack(anchor="w", padx=6, pady=4)
        ttk.Label(plaid_box, text="Wave 2 Orientation (deg):").pack(anchor="w", padx=6)
        ttk.Scale(plaid_box, from_=0, to=360, variable=self.theta2, command=lambda e: self.update_preview()).pack(fill=tk.X, padx=6)

        # 4. Isoluminance Texture Controls
        tex_box = ttk.LabelFrame(scrollable_frame, text=" Isoluminance Texture ")
        tex_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(tex_box, text="Synthesis Mode:").pack(anchor="w", padx=6)
        mode_combo = ttk.Combobox(tex_box, textvariable=self.texture_mode, values=["gabor", "continuous", "bands"], state="readonly")
        mode_combo.pack(fill=tk.X, padx=6, pady=2)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self.update_preview())

        ttk.Label(tex_box, text="Number of Bands:").pack(anchor="w", padx=6)
        ttk.Scale(tex_box, from_=3, to=9, variable=self.num_bands, command=lambda e: self.update_preview()).pack(fill=tk.X, padx=6)

        ttk.Checkbutton(tex_box, text="Vary Orientation Across Bands", variable=self.vary_ori, command=self.update_preview).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(tex_box, text="Vary Frequency Across Bands", variable=self.vary_freq, command=self.update_preview).pack(anchor="w", padx=6, pady=2)
        ttk.Checkbutton(tex_box, text="Vary Color Across Bands", variable=self.vary_col, command=self.update_preview).pack(anchor="w", padx=6, pady=2)

        # 5. Spatial Arrangement / Control Condition
        arr_box = ttk.LabelFrame(scrollable_frame, text=" Spatial Arrangement / Control ")
        arr_box.pack(fill=tk.X, padx=8, pady=4)

        ttk.Checkbutton(
            arr_box,
            text="Random Mixture Control Pattern",
            variable=self.random_mixture_control,
            command=self.update_preview
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Label(arr_box, text="(Uncorrelates cues from plaid isoluminance)", font=("Segoe UI", 8, "italic")).pack(anchor="w", padx=6)

        seed_row = ttk.Frame(arr_box)
        seed_row.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(seed_row, text="Re-roll Mixture Seed", command=self._reroll_seed).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Action Buttons
        btn_box = ttk.Frame(scrollable_frame)
        btn_box.pack(fill=tk.X, padx=8, pady=10)
        ttk.Button(btn_box, text="Save Stimulus (BMP)", command=self.save_image).pack(fill=tk.X, pady=3)
        ttk.Button(btn_box, text="Open Web App in Browser", command=self.open_web_app).pack(fill=tk.X, pady=3)

        # Right Preview Area
        preview_frame = ttk.LabelFrame(main_frame, text=" Stimulus Preview ")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.preview_canvas = tk.Canvas(preview_frame, bg="#000000", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _reset_shift(self):
        self.shift_x.set(0)
        self.shift_y.set(0)
        self.update_preview()

    def _on_wp_change(self, val):
        self.w_texture.set(round(1.0 - float(val), 2))
        self.update_preview()

    def _on_wt_change(self, val):
        self.w_plaid.set(round(1.0 - float(val), 2))
        self.update_preview()

    def _on_theta1_change(self, val):
        if self.lock_ortho.get():
            self.theta2.set((float(val) + 90.0) % 360.0)
        self.update_preview()

    def _on_freq1_change(self, val):
        self.freq2.set(float(val))
        self.update_preview()

    def _on_lock_ortho(self):
        if self.lock_ortho.get():
            self.theta2.set((self.theta1.get() + 90.0) % 360.0)
            self.update_preview()

    def _reroll_seed(self):
        import random
        self.mixture_seed.set(random.randint(1000, 99999))
        self.update_preview()

    def update_preview(self):
        blender = PlaidIsoBlender(
            width=self.width_val,
            height=self.height_val,
            w_plaid=self.w_plaid.get(),
            w_texture=self.w_texture.get(),
            plaid_contrast=self.plaid_contrast.get(),
            shift_x=self.shift_x.get(),
            shift_y=self.shift_y.get(),
            random_mixture_control=self.random_mixture_control.get(),
            mixture_seed=self.mixture_seed.get(),
            theta1_deg=self.theta1.get(),
            freq1=self.freq1.get(),
            theta2_deg=self.theta2.get(),
            freq2=self.freq2.get(),
            num_bands=int(self.num_bands.get()),
            texture_mode=self.texture_mode.get(),
            vary_orientation=self.vary_ori.get(),
            vary_frequency=self.vary_freq.get(),
            vary_color=self.vary_col.get(),
            density=self.density.get()
        )
        blend_rgb, _, _ = blender.generate()

        #ppm_data = matrix_to_ppm_bytes(blend_rgb, self.width_val, self.height_val)
        #self.photo = tk.PhotoImage(data=ppm_data)

        self.preview_canvas.delete("all")
        cw = self.preview_canvas.winfo_width() or self.width_val
        ch = self.preview_canvas.winfo_height() or self.height_val

        if HAS_PIL:
            # Use PIL for crisp high-DPI handling and resampling
            img = Image.fromarray(blend_rgb)
            self.photo = ImageTk.PhotoImage(img)
            # If canvas is larger/smaller, resize cleanly using NEAREST or LANCZOS
            # To keep pixel art / gratings sharp, use Image.Resampling.NEAREST (or LANCZOS for smooth downscaling)
            # Scale up 2x using NEAREST neighbor so Retina pixels map 1-to-1 cleanly
            #retina_width = self.width_val * 2
            #retina_height = self.height_val * 2
            #img_retina = img.resize((retina_width, retina_height), Image.Resampling.NEAREST)
            #img_resized = img.resize((max(100, cw - 20), max(100, ch - 20)), Image.Resampling.NEAREST)
            #self.photo = ImageTk.PhotoImage(img_resized)
            #self.photo = ImageTk.PhotoImage(img_retina)
        else:
            # Fallback to raw PPM bytes if PIL is missing
            ppm_data = matrix_to_ppm_bytes(blend_rgb, self.width_val, self.height_val)
            self.photo = tk.PhotoImage(data=ppm_data)


        self.preview_canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")

    def save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("24-bit Bitmap", "*.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        blender = PlaidIsoBlender(
            width=self.width_val,
            height=self.height_val,
            w_plaid=self.w_plaid.get(),
            w_texture=self.w_texture.get(),
            plaid_contrast=self.plaid_contrast.get(),
            shift_x=self.shift_x.get(),
            shift_y=self.shift_y.get(),
            random_mixture_control=self.random_mixture_control.get(),
            mixture_seed=self.mixture_seed.get(),
            theta1_deg=self.theta1.get(),
            freq1=self.freq1.get(),
            theta2_deg=self.theta2.get(),
            freq2=self.freq2.get(),
            num_bands=int(self.num_bands.get()),
            texture_mode=self.texture_mode.get(),
            vary_orientation=self.vary_ori.get(),
            vary_frequency=self.vary_freq.get(),
            vary_color=self.vary_col.get()
        )
        blender.save_bmp(path)
        messagebox.showinfo("Saved", f"Stimulus saved successfully to:\n{path}")

    def open_web_app(self):
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(html_path):
            webbrowser.open(f"file:///{os.path.abspath(html_path)}")
        else:
            messagebox.showwarning("Not Found", "Could not locate index.html")


if __name__ == "__main__":
    app = PlaidIsoApp()
    app.mainloop()
