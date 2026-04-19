"""
preprocess_live.py — Live image preprocessing for motif documentation
Usage: py preprocess_live.py --image 1.jpg

Layout: Original (drag to crop) | Processed | Scrollable controls
Crop:   Click and drag on the Original panel to draw a crop box.
        The Processed panel updates live to show cropped + adjusted image.
        [Clear crop] restores the full image.
        [S] Save exports the cropped + processed image at full resolution.
Keyboard: R = reset sliders   C = clear crop   P = print params   S = save
"""

import argparse
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ── Image processing ──────────────────────────────────────────────────────────

def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 0.01:
        return img.copy()
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def apply_clahe(img, clip, tile):
    tile = max(1, int(tile))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def apply_retinex(img, sigma, strength):
    """Single-scale retinex — runs on a 25% downscale then upsamples
    the illumination map back, keeping speed and memory manageable."""
    if strength < 0.01:
        return img.copy()
    H, W = img.shape[:2]
    scale  = min(1.0, 600.0 / max(H, W))
    sw, sh = max(1, int(W * scale)), max(1, int(H * scale))
    small  = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    f      = small.astype(np.float32) + 1.0
    result = np.zeros_like(f)
    adj_sigma = max(3.0, sigma * scale)
    for c in range(3):
        blur            = cv2.GaussianBlur(f[:, :, c], (0, 0), adj_sigma)
        r               = np.log10(f[:, :, c]) - np.log10(blur + 1.0)
        result[:, :, c] = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
    result_full = cv2.resize(result.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1.0 - strength, result_full, strength, 0)


def apply_gradient_norm(img, strength):
    """Gradient-aware lighting normalisation for tubelight/strip-light panels.

    Log-domain background subtraction, per-channel:
      1. Convert each channel to log space
      2. Large Gaussian blur isolates the low-frequency illumination gradient
      3. Subtract the blur — removes the gradient, keeps cast shadows
      4. Convert back to linear space, re-normalise

    Key design decisions:
      - Per-channel, not luminance-only — removes chrominance gradients too
      - NO Sobel edge blend — that was reintroducing the gradient at edges
      - Log subtraction inherently preserves high-frequency content (shadows,
        groove edges) because those are local variations, not global trend
      - Sigma = 25% of longest axis — large enough to span the full tubelight
        gradient, which is typically a smooth ramp across the whole panel
      - Downscaled blur for speed, upsampled back to full res"""
    if strength < 0.01:
        return img.copy()
    H, W   = img.shape[:2]
    L      = max(H, W)
    sigma  = L * 0.25   # large — covers full panel gradient

    # Downscale factor for the blur computation
    ds     = min(1.0, 600.0 / L)
    sw, sh = max(1, int(W * ds)), max(1, int(H * ds))

    result = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch     = img[:, :, c].astype(np.float32)
        log_ch = np.log1p(ch)

        # Estimate illumination on downscale, upsample back
        log_small  = cv2.resize(log_ch, (sw, sh), interpolation=cv2.INTER_AREA)
        blur_small = cv2.GaussianBlur(log_small, (0, 0), sigma * ds)
        illum      = cv2.resize(blur_small, (W, H), interpolation=cv2.INTER_LINEAR)

        # Subtract gradient in log domain
        corrected  = log_ch - illum

        # Re-centre: shift so mean matches original mean (preserves overall tone)
        corrected  = corrected - corrected.mean() + log_ch.mean()

        # Back to linear
        corrected  = np.expm1(corrected)
        corrected  = np.clip(corrected, 0, 255)
        result[:, :, c] = corrected

    result = result.astype(np.uint8)
    return cv2.addWeighted(img, 1.0 - strength, result, strength, 0)


def apply_flash_correct(img, strength):
    """Fast flash/hotspot correction via quotient image (linear space division).
    Use for camera flash. For tubelight gradients use gradient_norm instead."""
    if strength < 0.01:
        return img.copy()
    H, W  = img.shape[:2]
    sigma = max(H, W) * 0.20
    f     = img.astype(np.float32) + 1.0
    result= np.zeros_like(f)
    for c in range(3):
        illum           = cv2.GaussianBlur(f[:, :, c], (0, 0), sigma)
        quotient        = f[:, :, c] / (illum + 1e-6)
        result[:, :, c] = cv2.normalize(quotient, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.addWeighted(img, 1.0 - strength, result.astype(np.uint8), strength, 0)



def apply_shadow_lift(img, lift):
    if abs(lift) < 0.01:
        return img.copy()
    lut = np.arange(256, dtype=np.float32)
    lut = lut + lift * (1.0 - lut / 255.0) ** 2 * 255.0
    return cv2.LUT(img, np.clip(lut, 0, 255).astype(np.uint8))

def apply_contrast(img, contrast):
    if abs(contrast - 1.0) < 0.01:
        return img.copy()
    lut = np.clip(
        (np.arange(256, dtype=np.float32) - 128.0) * contrast + 128.0,
        0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

def process(img, p):
    out = apply_gamma(img, p["gamma"])
    out = apply_clahe(out, p["clahe_clip"], p["clahe_tile"])
    out = apply_shadow_lift(out, p["shadow_lift"])
    out = apply_gradient_norm(out, p["gradient_norm"])   # tubelight gradient
    out = apply_flash_correct(out, p["flash_correct"])   # camera flash
    out = apply_retinex(out, p["retinex_sigma"], p["retinex_strength"])
    out = apply_contrast(out, p["contrast"])
    return out


# ── GUI ───────────────────────────────────────────────────────────────────────

PREVIEW_MAX = 560   # longest axis in px — two panels + controls fit in 1920


class LiveEditor:
    def __init__(self, root, image_path):
        self.root        = root
        self.image_path  = image_path
        self.source_bgr  = cv2.imread(image_path)
        if self.source_bgr is None:
            messagebox.showerror("Error", f"Cannot open: {image_path}")
            root.destroy()
            return

        # Crop state — (x1,y1,x2,y2) in FULL-RES pixels, or None
        self._crop      = None
        # Rubber-band drag state
        self._drag_start = None
        self._rect_id    = None

        self.root.title(f"Live Preprocessor — {os.path.basename(image_path)}")
        self.root.configure(bg="#1a1a1a")
        self._compute_thumb_size(self.source_bgr)
        self._build_ui()
        self._redraw_orig()
        self._update()

    # ── Thumbnail sizing ──────────────────────────────────────────────────────

    def _compute_thumb_size(self, img):
        h, w = img.shape[:2]
        scale      = min(PREVIEW_MAX / w, PREVIEW_MAX / h)
        self._tw   = max(1, int(w * scale))
        self._th   = max(1, int(h * scale))
        self._scale = scale   # full-res → thumb

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Status bar — bottom first so it anchors correctly
        self.status_var = tk.StringVar(value="Draw a crop box on Original, or adjust sliders")
        tk.Label(self.root, textvariable=self.status_var, bg="#111", fg="#555",
                 font=("Consolas", 8), anchor="w"
                 ).pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=2)

        outer = tk.Frame(self.root, bg="#1a1a1a")
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # ── Original (Canvas so we can draw the crop rubber-band) ────────
        col_a = tk.Frame(outer, bg="#1a1a1a")
        col_a.pack(side=tk.LEFT, anchor="n", padx=(0, 8))

        hdr_a = tk.Frame(col_a, bg="#1a1a1a")
        hdr_a.pack(fill=tk.X, pady=(0, 4))
        tk.Label(hdr_a, text="Original  (drag to crop)",
                 bg="#1a1a1a", fg="#777",
                 font=("Consolas", 10)).pack(side=tk.LEFT)
        tk.Button(hdr_a, text="✕ Clear crop",
                  bg="#3a2020", fg="#c08080",
                  font=("Consolas", 8), relief="flat",
                  padx=6, pady=2, cursor="hand2",
                  command=self._clear_crop
                  ).pack(side=tk.RIGHT)

        self.canvas_orig = tk.Canvas(col_a, bg="#111",
                                     width=self._tw, height=self._th,
                                     highlightthickness=1,
                                     highlightbackground="#333",
                                     cursor="crosshair")
        self.canvas_orig.pack()

        # Drag bindings
        self.canvas_orig.bind("<ButtonPress-1>",   self._drag_start_cb)
        self.canvas_orig.bind("<B1-Motion>",        self._drag_move_cb)
        self.canvas_orig.bind("<ButtonRelease-1>",  self._drag_end_cb)

        # ── Processed ─────────────────────────────────────────────────────
        col_b = tk.Frame(outer, bg="#1a1a1a")
        col_b.pack(side=tk.LEFT, anchor="n", padx=(0, 12))
        tk.Label(col_b, text="Processed",
                 bg="#1a1a1a", fg="#e0c97f",
                 font=("Consolas", 10)).pack(pady=(0, 4))
        self.canvas_proc = tk.Label(col_b, bg="#111", relief="flat",
                                    width=self._tw, height=self._th)
        self.canvas_proc.pack()

        # ── Scrollable controls ───────────────────────────────────────────
        col_c = tk.Frame(outer, bg="#1a1a1a")
        col_c.pack(side=tk.LEFT, anchor="n", fill=tk.Y)

        ctrl_cv = tk.Canvas(col_c, bg="#1a1a1a", highlightthickness=0,
                            width=370, height=self._th + 24)
        sb = ttk.Scrollbar(col_c, orient=tk.VERTICAL, command=ctrl_cv.yview)
        ctrl_cv.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(ctrl_cv, bg="#1a1a1a")
        ctrl_cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: ctrl_cv.configure(
                       scrollregion=ctrl_cv.bbox("all")))
        ctrl_cv.bind_all(
            "<MouseWheel>",
            lambda e: ctrl_cv.yview_scroll(int(-1 * e.delta / 120), "units"))

        # ── Crop info label (updates when crop is set) ────────────────────
        self.crop_lbl_var = tk.StringVar(value="No crop set")
        tk.Label(inner, textvariable=self.crop_lbl_var,
                 bg="#1a1a1a", fg="#7aafcf",
                 font=("Consolas", 8), anchor="w"
                 ).grid(row=0, column=0, columnspan=3,
                        sticky="ew", padx=4, pady=(4, 8))

        # ── Sliders ───────────────────────────────────────────────────────
        SLIDERS = [
            ("gamma",            "Gamma",           0.4,   3.0,  1.0,  "── Tone"),
            ("clahe_clip",       "CLAHE clip",      0.5,  10.0,  2.5,  None),
            ("clahe_tile",       "CLAHE tile",      2,    32,    8,    None),
            ("shadow_lift",      "Shadow lift",     0.0,   1.0,  0.0,  None),
            ("gradient_norm",    "Gradient norm",   0.0,   1.0,  0.0,  "── Lighting correction"),
            ("flash_correct",    "Flash correct",   0.0,   1.0,  0.0,  None),
            ("retinex_sigma",    "Retinex sigma",   5,   200,   80,    None),
            ("retinex_strength", "Retinex str",     0.0,   1.0,  0.0,  None),
            ("contrast",         "Contrast",        0.5,   2.0,  1.0,  "── Output"),
        ]
        TIPS = {
            "gamma":            "Raise above 1.0 for dark granite",
            "clahe_clip":       "Higher = more aggressive local shadow lift",
            "clahe_tile":       "Smaller = tighter neighbourhood",
            "shadow_lift":      "Quadratic lift — protects highlights",
            "gradient_norm":    "TUBELIGHT — log-domain subtraction, Sobel edge preservation",
            "flash_correct":    "CAMERA FLASH — quotient correction for hotspot gradients",
            "retinex_sigma":    "Illumination blur radius (slower on large images)",
            "retinex_strength": "Blend — start 0, raise slowly",
            "contrast":         "Mid-point contrast stretch",
        }

        self.vars = {}
        row = 1  # row 0 used by crop label

        for key, label, lo, hi, default, group in SLIDERS:
            if group:
                tk.Label(inner, text=group, bg="#1a1a1a", fg="#555",
                         font=("Consolas", 8), anchor="w"
                         ).grid(row=row, column=0, columnspan=3,
                                sticky="ew", pady=(12, 3), padx=4)
                row += 1

            tk.Label(inner, text=label, bg="#1a1a1a", fg="#ccc",
                     font=("Consolas", 9), width=14, anchor="w"
                     ).grid(row=row, column=0, sticky="w", padx=4)

            var = tk.DoubleVar(value=default)
            self.vars[key] = var

            ttk.Scale(inner, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, length=210,
                      command=lambda _e: self._update()
                      ).grid(row=row, column=1, padx=4)

            val_lbl = tk.Label(inner, bg="#1a1a1a", fg="#e0c97f",
                               font=("Consolas", 9), width=5, anchor="w")
            val_lbl.grid(row=row, column=2, sticky="w")
            var.trace_add("write",
                          lambda *a, v=var, lbl=val_lbl:
                          lbl.configure(text=f"{v.get():.2f}"))
            row += 1

            tk.Label(inner, text=TIPS[key], bg="#1a1a1a", fg="#444",
                     font=("Consolas", 7), anchor="w"
                     ).grid(row=row, column=1, columnspan=2,
                            sticky="w", pady=(0, 4))
            row += 1

        # Divider + buttons
        tk.Frame(inner, bg="#333", height=1
                 ).grid(row=row, column=0, columnspan=3,
                        sticky="ew", pady=(10, 8), padx=4)
        row += 1

        bf = tk.Frame(inner, bg="#1a1a1a")
        bf.grid(row=row, column=0, columnspan=3, sticky="w", padx=4)
        bs = {"font": ("Consolas", 9), "relief": "flat",
              "padx": 10, "pady": 5, "cursor": "hand2"}
        tk.Button(bf, text="[R] Reset",
                  bg="#333",    fg="#aaa",
                  command=self._reset, **bs).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bf, text="[C] Clear crop",
                  bg="#3a2020", fg="#c08080",
                  command=self._clear_crop, **bs).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bf, text="[P] Params",
                  bg="#333",    fg="#aaa",
                  command=self._print_params, **bs).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bf, text="[S] Save",
                  bg="#2a4a2a", fg="#8fbc8f",
                  command=self._save, **bs).pack(side=tk.LEFT)

        self.root.bind("<r>", lambda e: self._reset())
        self.root.bind("<c>", lambda e: self._clear_crop())
        self.root.bind("<p>", lambda e: self._print_params())
        self.root.bind("<s>", lambda e: self._save())

    # ── Crop rubber-band ──────────────────────────────────────────────────────

    def _drag_start_cb(self, e):
        self._drag_start = (e.x, e.y)
        if self._rect_id:
            self.canvas_orig.delete(self._rect_id)
            self._rect_id = None

    def _drag_move_cb(self, e):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        if self._rect_id:
            self.canvas_orig.delete(self._rect_id)
        self._rect_id = self.canvas_orig.create_rectangle(
            x0, y0, e.x, e.y,
            outline="#e0c97f", width=2, dash=(4, 3))

    def _drag_end_cb(self, e):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = e.x, e.y
        self._drag_start = None

        # Need at least 10×10 px on the thumbnail to be meaningful
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return

        # Map thumbnail coords → full-res coords
        sx0 = int(min(x0, x1) / self._scale)
        sy0 = int(min(y0, y1) / self._scale)
        sx1 = int(max(x0, x1) / self._scale)
        sy1 = int(max(y0, y1) / self._scale)

        H, W = self.source_bgr.shape[:2]
        sx0 = max(0, min(sx0, W - 1))
        sy0 = max(0, min(sy0, H - 1))
        sx1 = max(sx0 + 1, min(sx1, W))
        sy1 = max(sy0 + 1, min(sy1, H))

        self._crop = (sx0, sy0, sx1, sy1)
        self.crop_lbl_var.set(
            f"Crop  x:{sx0}–{sx1}  y:{sy0}–{sy1}  "
            f"({sx1-sx0}×{sy1-sy0} px full-res)")
        self._redraw_orig()
        self._update()

    def _clear_crop(self):
        self._crop = None
        if self._rect_id:
            self.canvas_orig.delete(self._rect_id)
            self._rect_id = None
        self.crop_lbl_var.set("No crop set")
        self._compute_thumb_size(self.source_bgr)
        self._redraw_orig()
        self._update()

    # ── Working image (cropped or full) ──────────────────────────────────────

    def _working_img(self):
        if self._crop:
            x0, y0, x1, y1 = self._crop
            return self.source_bgr[y0:y1, x0:x1]
        return self.source_bgr

    # ── Draw original panel (shows full image with dimmed crop overlay) ───────

    def _redraw_orig(self):
        src = self.source_bgr
        H, W = src.shape[:2]
        scale = min(PREVIEW_MAX / W, PREVIEW_MAX / H)
        tw = max(1, int(W * scale))
        th = max(1, int(H * scale))
        self._scale = scale

        thumb = cv2.resize(src, (tw, th))

        if self._crop:
            x0, y0, x1, y1 = self._crop
            # Dim the non-crop area
            overlay = thumb.copy()
            overlay[:, :] = (overlay * 0.35).astype(np.uint8)
            tx0 = int(x0 * scale); ty0 = int(y0 * scale)
            tx1 = int(x1 * scale); ty1 = int(y1 * scale)
            overlay[ty0:ty1, tx0:tx1] = thumb[ty0:ty1, tx0:tx1]
            # Gold border
            cv2.rectangle(overlay, (tx0, ty0), (tx1-1, ty1-1),
                          (30, 200, 224), 2)
            thumb = overlay

        tk_img = self._to_tk(thumb)
        self.canvas_orig.configure(width=tw, height=th)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(0, 0, anchor="nw", image=tk_img)
        self.canvas_orig._tk = tk_img   # keep reference

    # ── Update processed panel ────────────────────────────────────────────────

    def _update(self, *_):
        p    = self._get_params()
        work = self._working_img()
        proc = process(work, p)

        # Size processed thumbnail to match PREVIEW_MAX on the working region
        h, w = proc.shape[:2]
        scale = min(PREVIEW_MAX / w, PREVIEW_MAX / h)
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        thumb = cv2.resize(proc, (tw, th))

        tk_img = self._to_tk(thumb)
        self.canvas_proc.configure(image=tk_img, width=tw, height=th)
        self.canvas_proc._tk = tk_img

        crop_txt = (f"  crop:{self._crop[2]-self._crop[0]}×"
                    f"{self._crop[3]-self._crop[1]}px"
                    if self._crop else "  full image")
        self.status_var.set(
            f"γ={p['gamma']:.2f}  clip={p['clahe_clip']:.1f}  "
            f"tile={int(p['clahe_tile'])}  lift={p['shadow_lift']:.2f}  "
            f"ret_σ={p['retinex_sigma']:.0f}  "
            f"ret_str={p['retinex_strength']:.2f}  "
            f"contrast={p['contrast']:.2f}"
            + crop_txt)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _get_params(self):
        return {k: v.get() for k, v in self.vars.items()}

    def _reset(self):
        for k, v in {"gamma": 1.0, "clahe_clip": 2.5, "clahe_tile": 8,
                     "shadow_lift": 0.0, "gradient_norm": 0.0,
                     "flash_correct": 0.0, "retinex_sigma": 80.0,
                     "retinex_strength": 0.0, "contrast": 1.0}.items():
            self.vars[k].set(v)
        self._update()

    def _print_params(self):
        p = self._get_params()
        print("\n── Preprocessing params ──")
        for k, v in p.items():
            print(f"  {k:<20} = {v:.3f}")
        if self._crop:
            x0, y0, x1, y1 = self._crop
            print(f"  crop                 = {x0},{y0},{x1},{y1}  "
                  f"({x1-x0}×{y1-y0} px)")
        print()

    def _save(self):
        p    = self._get_params()
        work = self._working_img()
        proc = process(work, p)

        base, ext = os.path.splitext(self.image_path)
        suffix = "_preprocessed"
        if self._crop:
            x0, y0, x1, y1 = self._crop
            suffix += f"_crop{x0}_{y0}_{x1}_{y1}"
        default_name = os.path.basename(f"{base}{suffix}{ext}")

        out = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=ext,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")])
        if out:
            cv2.imwrite(out, proc)
            self.status_var.set(f"Saved → {out}")
            print(f"[Saved] {out}")

    @staticmethod
    def _to_tk(bgr):
        return ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Live preprocessor with visual crop for motif docs")
    ap.add_argument("--image", required=True, help="Input image path")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print(f"[Error] Not found: {args.image}")
        sys.exit(1)

    root = tk.Tk()
    root.resizable(True, True)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TScale", background="#1a1a1a", troughcolor="#2a2a2a",
                    sliderlength=14, sliderrelief="flat")
    style.configure("TScrollbar", background="#2a2a2a", troughcolor="#1a1a1a",
                    arrowcolor="#555")

    LiveEditor(root, args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
