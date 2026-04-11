"""
DA3 Motifs — Interactive Processor
One image at a time. Tune parameters, preview mesh in Open3D, save only the
final heightfield STL when satisfied.

Controls per image:
  [p] preview mesh in Open3D
  [e] edit parameters
  [d] show depth map + mask (matplotlib)
  [s] save final STL and move to next
  [r] re-run depth inference (same image)
  [n] skip this image (no save)
  [q] quit

Usage:
  python motifs_interactive.py
  python motifs_interactive.py --input ./my_photos --output ./finals
  python motifs_interactive.py --model base        # faster / less VRAM
  python motifs_interactive.py --image one.jpg     # single image

Preprocessing:
  Run preprocess_live.py first for live CLAHE / gamma / shadow correction.
  This script auto-detects a *_preprocessed.* sidecar and offers to use it.

Install:
  pip install "torch>=2" torchvision open3d opencv-python pillow matplotlib
  pip install --force-reinstall "numpy==2.0.2" "pillow<12"
  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ── Output directory (override --output default) ─────────────────────────────
OUTPUT_DIR = "stl_output"

MODEL_MAP = {
    "nested": "depth-anything/DA3NESTED-GIANT-LARGE",
    "large":  "depth-anything/DA3-Large",
    "base":   "depth-anything/DA3-BASE",
}

# ── Default parameters (all editable per-image) ─────────────────────────────
DEFAULT_PARAMS = {
    "target_w":       1400,    # resize width for processing
    "far_pct":        85,      # depth percentile for foreground mask
    "erode_px":       6,       # extra erosion on mask boundary
    "invert":         True,    # invert depth (near=high relief)
    "use_center_roi": True,    # normalise from centre region only
    "detail_boost":   0.25,    # high-pass sharpening (0 = off)
    "relief_mm":      10,      # relief height in mm
    "base_mm":        3,       # base thickness in mm
    "px_mm":          2,       # pixel size in mm (controls output physical size)
    "z_exag":          24.0,   # z exaggeration multiplier
    "mesh_blur_sigma": 1.5,    # Gaussian blur on depth before mesh (0=off, reduces striations)
    "aspect_crop":     True,   # crop to mask bounding box
    "crop_pad_px":     20,     # padding around bounding-box crop in px
}


# ── Depth inference ──────────────────────────────────────────────────────────

def run_depth(model, image_path):
    pred  = model.inference([image_path])
    depth = pred.depth[0].astype(np.float32)

    # DA3 may return depth rotated 90 degrees for portrait images.
    # Detect by comparing aspect ratios; rotate clockwise (k=3) to correct.
    src_img        = cv2.imread(image_path)
    src_h, src_w   = src_img.shape[:2]
    d_h,   d_w     = depth.shape[:2]
    if (src_h > src_w) != (d_h > d_w):
        depth = np.rot90(depth, k=3)   # 90 degrees clockwise
        print(f"  [INFO] Depth rotated 90 CW to match source "
              f"(source {src_w}x{src_h}, depth was {d_w}x{d_h})")
    return depth



# ── Bounding-box crop ────────────────────────────────────────────────────────

def crop_to_mask_bbox(arrays, mask, pad_px=20):
    """Crop a list of 2-D arrays to the tight bounding box of mask (uint8 0/255)."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return arrays, None
    x, y, w, h = cv2.boundingRect(coords)
    H, W = mask.shape[:2]
    x1 = max(0, x - pad_px);  y1 = max(0, y - pad_px)
    x2 = min(W, x + w + pad_px);  y2 = min(H, y + h + pad_px)
    return [a[y1:y2, x1:x2] for a in arrays], (y1, y2, x1, x2)


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_path, depth, p):
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    w2 = p["target_w"]
    h2 = int(H * w2 / W)

    rgb2 = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
    d2   = cv2.resize(depth, (w2, h2), interpolation=cv2.INTER_LINEAR)

    # Robust normalise
    p1, p99 = np.percentile(d2, 1), np.percentile(d2, 99)
    d2 = np.clip((d2 - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    # Depth-based foreground mask
    far  = np.percentile(d2, p["far_pct"])
    mask = (d2 < far).astype(np.uint8) * 255

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Keep largest connected component
    num, labels = cv2.connectedComponents(mask)
    if num > 1:
        areas = [(labels == i).sum() for i in range(1, num)]
        keep  = 1 + int(np.argmax(areas))
        mask  = (labels == keep).astype(np.uint8) * 255

    mask_erode = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    d2_smooth = cv2.bilateralFilter(d2.astype(np.float32), d=9, sigmaColor=0.05, sigmaSpace=9)
    d2_smooth = cv2.GaussianBlur(d2_smooth, (5, 5), sigmaX=1.0)

    coverage = mask.sum() / (255.0 * mask.size)
    if coverage > 0.70:
        print(f"  [WARN] Mask coverage {coverage:.0%} -- try lowering far_pct to 45-65.")

    if p.get("aspect_crop", True):
        pad = int(p.get("crop_pad_px", 20))
        cropped, bbox = crop_to_mask_bbox([rgb2, d2_smooth, mask, mask_erode], mask, pad_px=pad)
        if bbox is not None:
            rgb2, d2_smooth, mask, mask_erode = cropped
            y1, y2, x1, x2 = bbox
            print(f"  Cropped to mask bbox: {x2-x1} x {y2-y1} px  (pad={pad})")

    return rgb2, d2_smooth, mask, mask_erode


# ── Heightfield mesh (final output only) ────────────────────────────────────

def build_heightfield(d2_smooth, mask_erode, p):
    """Returns an o3d.TriangleMesh. No files written here."""
    mask = (mask_erode > 0).astype(np.uint8)

    if p["erode_px"] > 0:
        k    = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * p["erode_px"] + 1, 2 * p["erode_px"] + 1)
        )
        mask = cv2.erode(mask, k, iterations=1)
    m = mask.astype(bool)

    if m.sum() < 500:
        raise RuntimeError("Mask too small — try reducing erode_px or far_pct.")

    d = d2_smooth.astype(np.float32).copy()

    norm_region = m
    if p["use_center_roi"]:
        h, w  = d.shape
        y0, y1 = int(0.15 * h), int(0.85 * h)
        x0, x1 = int(0.15 * w), int(0.85 * w)
        center = np.zeros_like(m, dtype=bool)
        center[y0:y1, x0:x1] = True
        nr = m & center
        norm_region = nr if nr.sum() >= 1000 else m

    vals    = d[norm_region]
    p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
    dn      = np.clip((d - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    if p["invert"]:
        dn = 1.0 - dn

    blur_sigma = p.get("mesh_blur_sigma", 0.0)
    if blur_sigma > 0:
        dn = cv2.GaussianBlur(dn, (0, 0), sigmaX=float(blur_sigma))

    if p["detail_boost"] > 0:
        blur = cv2.GaussianBlur(dn, (0, 0), sigmaX=3.0)
        dn   = np.clip(dn + p["detail_boost"] * (dn - blur), 0.0, 1.0)

    H_mm     = p["base_mm"] + (p["relief_mm"] * p["z_exag"]) * dn
    H_mm[~m] = 0.0

    h, w   = H_mm.shape
    X      = (np.arange(w) - (w - 1) / 2) * (p["px_mm"] / 1000)
    Y      = (np.arange(h) - (h - 1) / 2) * (p["px_mm"] / 1000)
    XX, YY = np.meshgrid(X, -Y)
    ZZ     = H_mm / 1000

    verts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)

    idx  = np.arange(h * w).reshape(h, w)
    a    = idx[:-1, :-1].reshape(-1)
    b    = idx[:-1,  1:].reshape(-1)
    c    = idx[ 1:, :-1].reshape(-1)
    d_   = idx[ 1:,  1:].reshape(-1)
    good = (m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]).reshape(-1)

    faces = np.vstack([
        np.stack([a, c, b],  axis=1)[good],
        np.stack([b, c, d_], axis=1)[good],
    ])

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.compute_vertex_normals()
    return mesh


# ── Depth + mask viewer (matplotlib) ─────────────────────────────────────────

def show_depth_view(image_path, depth, d2_smooth, mask, mask_erode):
    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    d_norm = depth - depth.min()
    d_norm = d_norm / (d_norm.max() + 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(os.path.basename(image_path), fontsize=11)

    axes[0].imshow(rgb);                              axes[0].set_title("Original");     axes[0].axis("off")
    axes[1].imshow(d_norm, cmap="Spectral");          axes[1].set_title("Raw depth");    axes[1].axis("off")
    axes[2].imshow(mask,   cmap="gray");              axes[2].set_title("Mask");         axes[2].axis("off")
    axes[3].imshow(d2_smooth, cmap="viridis");        axes[3].set_title("Smooth depth"); axes[3].axis("off")

    plt.tight_layout()
    plt.show()


# ── Open3D mesh preview ───────────────────────────────────────────────────────

def preview_mesh(mesh, title="Relief mesh preview"):
    V = np.asarray(mesh.vertices)
    z_range = (V[:, 2].max() - V[:, 2].min()) * 1000
    print(f"    Mesh: {len(V)} vertices, Z range {z_range:.1f} mm")
    print("    (Close the Open3D window to continue)")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1100, height=700)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.light_on             = True
    opt.mesh_show_back_face  = True
    opt.mesh_color_option    = o3d.visualization.MeshColorOption.Normal

    vis.run()
    vis.destroy_window()


# ── Parameter editor ─────────────────────────────────────────────────────────

PARAM_HELP = {
    "target_w":       "Resize width for processing (px)",
    "far_pct":        "Depth percentile for foreground mask (50–95)",
    "erode_px":       "Extra mask erosion in px — removes halo edges (0–20)",
    "invert":         "Invert depth so near surfaces are raised (True/False)",
    "use_center_roi": "Normalise from centre region only, avoids border skew (True/False)",
    "detail_boost":   "High-pass sharpening added back (0.0 = off, 0.25 default, max ~0.6)",
    "relief_mm":      "Relief height in mm (1–40)",
    "base_mm":        "Base plate thickness in mm (0–10)",
    "px_mm":          "Physical pixel size in mm — controls output dimensions (0.2–2.0)",
    "z_exag":          "Z exaggeration multiplier (1-30, 24 = default)",
    "mesh_blur_sigma": "Gaussian blur on depth before mesh (0=off, 1-3 reduces striations)",
    "aspect_crop":     "Crop to mask bounding box -- preserves rectangular panels (True/False)",
    "crop_pad_px":     "Padding around bounding-box crop in px (0-60)",
}

def edit_params(params):
    p = dict(params)
    print("\n  Current parameters:")
    keys = list(PARAM_HELP.keys())
    for i, k in enumerate(keys):
        print(f"    [{i}] {k:18s} = {p[k]:<10}  # {PARAM_HELP[k]}")

    print("\n  Enter param number to edit (blank = done, 'r' = reset defaults):")
    while True:
        raw = input("  > ").strip()
        if raw == "":
            break
        if raw == "r":
            p = dict(DEFAULT_PARAMS)
            print("  Reset to defaults.")
            continue
        try:
            idx = int(raw)
            k   = keys[idx]
            cur = p[k]
            new_raw = input(f"    {k} [{cur}] → ").strip()
            if new_raw == "":
                continue
            if isinstance(cur, bool):
                p[k] = new_raw.lower() in ("true", "1", "yes", "y")
            elif isinstance(cur, int):
                p[k] = int(new_raw)
            else:
                p[k] = float(new_raw)
            print(f"    {k} = {p[k]}")
        except (ValueError, IndexError):
            print("  Invalid — enter a number from the list.")
    return p


# ── Main interactive loop ────────────────────────────────────────────────────

def process_image(model, image_path, out_dir, params):
    """Full interactive cycle for one image. Returns True if saved, False if skipped."""
    name = os.path.splitext(os.path.basename(image_path))[0]
    p    = dict(params)

    print(f"\n{'═'*60}")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"{'═'*60}")

    # Run depth inference once (re-run only on [r])
    print("  Running depth inference...")
    depth = run_depth(model, image_path)
    print(f"  Depth shape: {depth.shape}  min/max: {depth.min():.3f}/{depth.max():.3f}")

    mesh      = None
    preproc   = None   # cached preprocessing result

    def refresh_preproc():
        nonlocal preproc
        preproc = preprocess(image_path, depth, p)

    refresh_preproc()

    while True:
        print(f"\n  [{name}]  [p]review  [e]dit params  [d]epth view  [s]ave  [r]e-run depth  [n]ext  [q]uit")
        cmd = input("  > ").strip().lower()

        if cmd == "q":
            print("  Quitting.")
            sys.exit(0)

        elif cmd == "n":
            print(f"  Skipped: {name}")
            return False

        elif cmd == "d":
            rgb2, d2_smooth, mask, mask_erode = preproc
            show_depth_view(image_path, depth, d2_smooth, mask, mask_erode)

        elif cmd == "e":
            p = edit_params(p)
            refresh_preproc()
            mesh = None  # invalidate cached mesh
            print("  Parameters updated. Run [p] to rebuild mesh.")

        elif cmd == "r":
            print("  Re-running depth inference...")
            depth = run_depth(model, image_path)
            refresh_preproc()
            mesh  = None
            print("  Done.")

        elif cmd == "p":
            print("  Building heightfield mesh...")
            try:
                _, d2_smooth, _, mask_erode = preproc
                mesh = build_heightfield(d2_smooth, mask_erode, p)
                preview_mesh(mesh, title=name)
            except Exception as e:
                print(f"  ERROR: {e}")
                mesh = None

        elif cmd == "s":
            if mesh is None:
                print("  No mesh yet — run [p] first to build and preview.")
                continue
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{name}.stl")
            o3d.io.write_triangle_mesh(out_path, mesh)
            V       = np.asarray(mesh.vertices)
            z_range = (V[:, 2].max() - V[:, 2].min()) * 1000
            print(f"  Saved: {out_path}")
            print(f"  Z range: {z_range:.1f} mm  |  Vertices: {len(V)}")
            return True

        else:
            print("  Unknown command.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./motifs/raw",    help="Folder of input images")
    parser.add_argument("--output", default=OUTPUT_DIR,        help="Folder for saved STL files")
    parser.add_argument("--model",  default="nested",
                        choices=list(MODEL_MAP.keys()),
                        help="nested = DA3NESTED-GIANT-LARGE | base = DA3BASE (faster)")
    parser.add_argument("--image",  default=None,              help="Process a single image file")
    parser.add_argument("--skip",   default=0, type=int,       help="Skip first N images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from depth_anything_3.api import DepthAnything3
    model_id = MODEL_MAP[args.model]
    print(f"Loading {model_id}...")
    model = DepthAnything3.from_pretrained(model_id).to(device)
    print("Model ready.\n")

    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted([
            p for ext in SUPPORTED_EXT
            for p in glob.glob(os.path.join(args.input, f"*{ext}"))
        ])[args.skip:]

    if not image_paths:
        print(f"No images found in: {args.input}")
        return

    total   = len(image_paths)
    saved   = 0
    skipped = 0
    params  = dict(DEFAULT_PARAMS)

    print(f"Found {total} image(s). Starting...\n")
    print("Tip: after editing params on one image, they carry over to the next.\n")

    for i, img_path in enumerate(image_paths):
        print(f"\n[Image {i + 1 + args.skip}/{total + args.skip}]")
        result = process_image(model, img_path, args.output, params)
        if result:
            saved += 1
        else:
            skipped += 1

    print(f"\n{'─'*50}")
    print(f"Session complete — saved: {saved}  skipped: {skipped}")


if __name__ == "__main__":
    main()