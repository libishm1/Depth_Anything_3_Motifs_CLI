"""
preprocess_live.py — Live image preprocessing for motif documentation
Usage: py preprocess_live.py --image 1.jpg

Layout: Original | Processed | Scrollable controls
Keyboard: R = reset  P = print params  S = save
"""

import argparse
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ── Image processing ─────────────────────────────────────────────────────────

def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 0.01:
        return img.copy()
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def apply_clahe(img, clip, tile):
    tile = max(1, int(tile))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def apply_retinex(img, sigma, strength):
    if strength < 0.01:
        return img.copy()
    f = img.astype(np.float32) + 1.0
    result = np.zeros_like(f)
    for c in range(3):
        blur = cv2.GaussianBlur(f[:, :, c], (0, 0), sigma)
        r = np.log10(f[:, :, c]) - np.log10(blur + 1.0)
        result[:, :, c] = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
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
    lut = np.clip((np.arange(256, dtype=np.float32) - 128.0) * contrast + 128.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)

def process(img, p):
    out = apply_gamma(img, p["gamma"])
    out = apply_clahe(out, p["clahe_clip"], p["clahe_tile"])
    out = apply_shadow_lift(out, p["shadow_lift"])
    out = apply_retinex(out, p["retinex_sigma"], p["retinex_strength"])
    out = apply_contrast(out, p["contrast"])
    return out


# ── GUI ───────────────────────────────────────────────────────────────────────

PREVIEW_MAX = 560   # max px on longest axis — two images + controls fit in 1920


class LiveEditor:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.source_bgr = cv2.imread(image_path)
        if self.source_bgr is None:
            messagebox.showerror("Error", f"Cannot open: {image_path}")
            root.destroy()
            return
        self.root.title(f"Live Preprocessor — {os.path.basename(image_path)}")
        self.root.configure(bg="#1a1a1a")
        self._build_ui()
        self._update()

    def _build_ui(self):
        # Status bar — pin to bottom first
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, bg="#111", fg="#555",
                 font=("Consolas", 8), anchor="w"
                 ).pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=2)

        # Outer row
        outer = tk.Frame(self.root, bg="#1a1a1a")
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Thumbnail size
        h, w = self.source_bgr.shape[:2]
        scale = min(PREVIEW_MAX / w, PREVIEW_MAX / h)
        self._tw = max(1, int(w * scale))
        self._th = max(1, int(h * scale))

        # ── Original ──────────────────────────────────────────────────────
        col_a = tk.Frame(outer, bg="#1a1a1a")
        col_a.pack(side=tk.LEFT, anchor="n", padx=(0, 8))
        tk.Label(col_a, text="Original", bg="#1a1a1a", fg="#777",
                 font=("Consolas", 10)).pack(pady=(0, 4))
        self.canvas_orig = tk.Label(col_a, bg="#111", relief="flat",
                                    width=self._tw, height=self._th)
        self.canvas_orig.pack()
        thumb = cv2.resize(self.source_bgr, (self._tw, self._th))
        self._orig_tk = self._to_tk(thumb)
        self.canvas_orig.configure(image=self._orig_tk)

        # ── Processed ─────────────────────────────────────────────────────
        col_b = tk.Frame(outer, bg="#1a1a1a")
        col_b.pack(side=tk.LEFT, anchor="n", padx=(0, 12))
        tk.Label(col_b, text="Processed", bg="#1a1a1a", fg="#e0c97f",
                 font=("Consolas", 10)).pack(pady=(0, 4))
        self.canvas_proc = tk.Label(col_b, bg="#111", relief="flat",
                                    width=self._tw, height=self._th)
        self.canvas_proc.pack()

        # ── Controls (scrollable) ─────────────────────────────────────────
        col_c = tk.Frame(outer, bg="#1a1a1a")
        col_c.pack(side=tk.LEFT, anchor="n", fill=tk.Y)

        scroll_h = self._th + 24   # match image height
        ctrl_canvas = tk.Canvas(col_c, bg="#1a1a1a", highlightthickness=0,
                                width=370, height=scroll_h)
        sb = ttk.Scrollbar(col_c, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(ctrl_canvas, bg="#1a1a1a")
        ctrl_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: ctrl_canvas.configure(
                       scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.bind_all(
            "<MouseWheel>",
            lambda e: ctrl_canvas.yview_scroll(int(-1 * e.delta / 120), "units"))

        # ── Slider definitions ────────────────────────────────────────────
        SLIDERS = [
            # (key, label, min, max, default, group_label or None)
            ("gamma",            "Gamma",           0.4,   3.0,  1.0,  "── Tone"),
            ("clahe_clip",       "CLAHE clip",      0.5,  10.0,  2.5,  None),
            ("clahe_tile",       "CLAHE tile",      2,    32,    8,    None),
            ("shadow_lift",      "Shadow lift",     0.0,   1.0,  0.0,  None),
            ("retinex_sigma",    "Retinex sigma",   5,   200,   80,    "── Retinex"),
            ("retinex_strength", "Retinex str",     0.0,   1.0,  0.0,  None),
            ("contrast",         "Contrast",        0.5,   2.0,  1.0,  "── Output"),
        ]
        TIPS = {
            "gamma":            "Raise above 1.0 for dark granite",
            "clahe_clip":       "Higher = more aggressive local shadow lift",
            "clahe_tile":       "Smaller = tighter neighbourhood",
            "shadow_lift":      "Quadratic lift — protects highlights",
            "retinex_sigma":    "Illumination blur radius",
            "retinex_strength": "Blend — start 0, raise slowly",
            "contrast":         "Mid-point contrast stretch",
        }

        self.vars = {}
        row = 0

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
                      command=lambda _e, k=key: self._update()
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

        # Divider
        tk.Frame(inner, bg="#333", height=1
                 ).grid(row=row, column=0, columnspan=3,
                        sticky="ew", pady=(10, 8), padx=4)
        row += 1

        # Buttons
        bf = tk.Frame(inner, bg="#1a1a1a")
        bf.grid(row=row, column=0, columnspan=3, sticky="w", padx=4)
        bs = {"font": ("Consolas", 9), "relief": "flat",
              "padx": 10, "pady": 5, "cursor": "hand2"}
        tk.Button(bf, text="[R] Reset",   bg="#333",    fg="#aaa",
                  command=self._reset, **bs).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bf, text="[P] Params",  bg="#333",    fg="#aaa",
                  command=self._print_params, **bs).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(bf, text="[S] Save",    bg="#2a4a2a", fg="#8fbc8f",
                  command=self._save, **bs).pack(side=tk.LEFT)

        self.root.bind("<r>", lambda e: self._reset())
        self.root.bind("<p>", lambda e: self._print_params())
        self.root.bind("<s>", lambda e: self._save())

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _get_params(self):
        return {k: v.get() for k, v in self.vars.items()}

    def _update(self, *_):
        p = self._get_params()
        processed = process(self.source_bgr, p)
        thumb = cv2.resize(processed, (self._tw, self._th))
        tk_img = self._to_tk(thumb)
        self.canvas_proc.configure(image=tk_img,
                                   width=self._tw, height=self._th)
        self.canvas_proc._tk = tk_img
        self.status_var.set(
            f"γ={p['gamma']:.2f}  clip={p['clahe_clip']:.1f}  "
            f"tile={int(p['clahe_tile'])}  lift={p['shadow_lift']:.2f}  "
            f"ret_σ={p['retinex_sigma']:.0f}  ret_str={p['retinex_strength']:.2f}  "
            f"contrast={p['contrast']:.2f}")

    def _reset(self):
        defaults = {"gamma": 1.0, "clahe_clip": 2.5, "clahe_tile": 8,
                    "shadow_lift": 0.0, "retinex_sigma": 80.0,
                    "retinex_strength": 0.0, "contrast": 1.0}
        for k, v in defaults.items():
            self.vars[k].set(v)
        self._update()

    def _print_params(self):
        p = self._get_params()
        print("\n── Preprocessing params ──")
        for k, v in p.items():
            print(f"  {k:<20} = {v:.3f}")
        print()

    def _save(self):
        p = self._get_params()
        processed = process(self.source_bgr, p)
        base, ext = os.path.splitext(self.image_path)
        out = filedialog.asksaveasfilename(
            initialfile=os.path.basename(f"{base}_preprocessed{ext}"),
            defaultextension=ext,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")])
        if out:
            cv2.imwrite(out, processed)
            self.status_var.set(f"Saved → {out}")
            print(f"[Saved] {out}")

    @staticmethod
    def _to_tk(bgr):
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    if not os.path.isfile(args.image):
        print(f"[Error] Not found: {args.image}"); sys.exit(1)

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
