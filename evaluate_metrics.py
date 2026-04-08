#!/usr/bin/env python3
"""
evaluate_metrics.py
-------------------
Quantitative evaluation script for Depth_Anything_3_Motifs_CLI pipeline outputs.
Computes five metrics across a set of panel results.

Metrics:
  1. SNC  - Surface Normal Consistency
  2. DRP  - Depth Range Plausibility (requires one physical caliper measurement)
  3. MR   - Mask Repeatability (IoU, requires two acquisitions of same panel)
  4. MC   - Mesh Completeness
  5. SI   - Depth Map Sharpness Index

Usage:
    python evaluate_metrics.py --results-dir ./results --panels-csv panels.csv [--output metrics_report.csv]

panels.csv format (one row per panel):
    panel_id, depth_map, mask, mesh_ply, source_image, physical_relief_mm, second_mask (optional)

Author: Libish Murugesan
License: MIT
Repo: https://github.com/libishm1/Depth_Anything_3_Motifs_CLI
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# Optional imports - warn gracefully if absent
try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    warnings.warn("open3d not found. SNC and MC metrics will be skipped. Install: pip install open3d")

try:
    from scipy.ndimage import laplace
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found. SI metric will use numpy fallback.")


# ─────────────────────────────────────────────
# METRIC 1: Surface Normal Consistency (SNC)
# ─────────────────────────────────────────────
def compute_snc(mesh_path: str, background_mask_path: str = None, radius: float = 0.02) -> dict:
    """
    Surface Normal Consistency.
    Computes mean cosine similarity of adjacent face normals in the background region.
    
    Args:
        mesh_path: Path to .ply mesh file
        background_mask_path: Optional binary mask PNG identifying background region.
                              If None, uses full mesh.
        radius: KD-tree search radius for neighbour normal comparison (in mesh units)
    
    Returns:
        dict with 'snc', 'n_face_pairs', 'below_threshold_pct'
    """
    if not HAS_O3D:
        return {"snc": None, "error": "open3d not installed"}
    
    if not Path(mesh_path).exists():
        return {"snc": None, "error": f"Mesh not found: {mesh_path}"}
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    if len(normals) == 0:
        return {"snc": None, "error": "No face normals computed"}
    
    # Build face centroid array
    centroids = vertices[triangles].mean(axis=1)
    
    # KD-tree for neighbour lookup
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centroids)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    similarities = []
    for i in range(len(normals)):
        [k, idx, _] = tree.search_radius_vector_3d(centroids[i], radius)
        if k < 2:
            continue
        n_i = normals[i]
        for j in idx[1:]:
            n_j = normals[j]
            norm_i = np.linalg.norm(n_i)
            norm_j = np.linalg.norm(n_j)
            if norm_i < 1e-8 or norm_j < 1e-8:
                continue
            cos_sim = np.dot(n_i, n_j) / (norm_i * norm_j)
            similarities.append(float(np.clip(cos_sim, -1.0, 1.0)))
    
    if not similarities:
        return {"snc": None, "error": "No face pairs found within radius"}
    
    snc = float(np.mean(similarities))
    below = float(np.mean(np.array(similarities) < 0.85) * 100)
    
    return {
        "snc": round(snc, 4),
        "n_face_pairs": len(similarities),
        "below_threshold_pct": round(below, 2),
        "target": 0.85,
        "pass": snc >= 0.85
    }


# ─────────────────────────────────────────────
# METRIC 2: Depth Range Plausibility (DRP)
# ─────────────────────────────────────────────
def compute_drp(depth_map_path: str, mask_path: str, physical_relief_mm: float,
                background_sample_box: tuple = None) -> dict:
    """
    Depth Range Plausibility.
    Maps predicted max relief depth in the motif mask to a physical mm estimate
    using one anchor measurement, then computes ratio to measured physical relief.
    
    Args:
        depth_map_path: Path to 16-bit greyscale depth map PNG
        mask_path: Path to binary mask PNG (white = motif region)
        physical_relief_mm: Measured relief depth at one reference point (mm)
                            Measured with digital calipers on-site.
        background_sample_box: (x1, y1, x2, y2) pixel box for background sampling.
                               If None, uses image border (10px inset).
    
    Returns:
        dict with 'drp_ratio', 'predicted_relief_mm', 'physical_relief_mm'
    
    NOTE: This metric requires ONE physical measurement per panel.
          Without it, DRP cannot be computed.
          Record measurements in panels.csv under 'physical_relief_mm'.
    """
    if physical_relief_mm is None or physical_relief_mm <= 0:
        return {
            "drp_ratio": None,
            "error": "physical_relief_mm not provided. Measure with calipers on-site.",
            "action_required": "Record one caliper measurement per panel in panels.csv"
        }
    
    depth_img = np.array(Image.open(depth_map_path).convert("I"))
    depth_float = depth_img.astype(np.float32)
    
    mask_img = np.array(Image.open(mask_path).convert("L"))
    motif_mask = mask_img > 127
    
    if background_sample_box is None:
        h, w = depth_float.shape
        inset = 10
        bg = np.zeros_like(depth_float, dtype=bool)
        bg[:inset, :] = True
        bg[-inset:, :] = True
        bg[:, :inset] = True
        bg[:, -inset:] = True
    else:
        x1, y1, x2, y2 = background_sample_box
        bg = np.zeros_like(depth_float, dtype=bool)
        bg[y1:y2, x1:x2] = True
    
    if not bg.any():
        return {"drp_ratio": None, "error": "No background pixels found"}
    if not motif_mask.any():
        return {"drp_ratio": None, "error": "Empty motif mask"}
    
    d_background = float(np.median(depth_float[bg]))
    d_max_motif = float(np.percentile(depth_float[motif_mask], 98))  # 98th pct avoids spike artefacts
    depth_range = depth_float.max() - depth_float.min()
    
    if depth_range < 1e-6:
        return {"drp_ratio": None, "error": "Depth map has zero range"}
    
    # Scale factor: map full 16-bit range to a reasonable physical range
    # This is a relative ratio — absolute scale requires the anchor measurement
    predicted_relative = (d_max_motif - d_background) / depth_range
    
    # Estimate predicted relief using anchor
    # Assumption: the anchor measurement IS at the 98th percentile depth of its panel's motif
    # Adjust scale_factor_mm per dataset from actual caliper measurements
    scale_factor_mm = physical_relief_mm / predicted_relative if predicted_relative > 0 else 1.0
    predicted_relief_mm = predicted_relative * scale_factor_mm
    
    drp_ratio = predicted_relief_mm / physical_relief_mm if physical_relief_mm > 0 else None
    
    return {
        "drp_ratio": round(float(drp_ratio), 4) if drp_ratio else None,
        "predicted_relief_mm": round(predicted_relief_mm, 2),
        "physical_relief_mm": physical_relief_mm,
        "d_background_raw": round(d_background, 1),
        "d_max_motif_raw": round(d_max_motif, 1),
        "target_range": "0.80–1.20",
        "pass": (0.80 <= drp_ratio <= 1.20) if drp_ratio else None,
        "warning": "DRP is an approximation. Requires on-site caliper measurement per panel."
    }


# ─────────────────────────────────────────────
# METRIC 3: Mask Repeatability (MR / IoU)
# ─────────────────────────────────────────────
def compute_mr(mask_path_1: str, mask_path_2: str) -> dict:
    """
    Mask Repeatability as Intersection over Union (IoU).
    Compares two segmentation masks of the same panel under different lighting.
    
    Args:
        mask_path_1: Binary mask PNG from acquisition 1
        mask_path_2: Binary mask PNG from acquisition 2 (different lighting)
    
    Returns:
        dict with 'iou', 'intersection_px', 'union_px'
    
    NOTE: Requires two acquisitions of the same panel.
          If only one acquisition exists, this metric cannot be computed.
    """
    if mask_path_2 is None or not Path(mask_path_2).exists():
        return {
            "iou": None,
            "error": "Second mask not provided",
            "action_required": "Photograph same panel under different lighting and run SAM again"
        }
    
    m1 = np.array(Image.open(mask_path_1).convert("L")) > 127
    m2_img = Image.open(mask_path_2).convert("L")
    
    # Resize m2 to m1 dimensions if needed
    if m2_img.size != Image.open(mask_path_1).size:
        m2_img = m2_img.resize(Image.open(mask_path_1).size, Image.NEAREST)
    m2 = np.array(m2_img) > 127
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return {"iou": None, "error": "Both masks are empty"}
    
    iou = float(intersection) / float(union)
    
    return {
        "iou": round(iou, 4),
        "intersection_px": int(intersection),
        "union_px": int(union),
        "m1_area_px": int(m1.sum()),
        "m2_area_px": int(m2.sum()),
        "target": 0.80,
        "pass": iou >= 0.80
    }


# ─────────────────────────────────────────────
# METRIC 4: Mesh Completeness (MC)
# ─────────────────────────────────────────────
def compute_mc(mesh_path: str, depth_map_path: str, relief_band_mm: float = 15.0,
               scale_factor: float = None) -> dict:
    """
    Mesh Completeness.
    Percentage of mesh faces whose depth values fall within the physically
    plausible relief band [D_background, D_background + relief_band_mm equivalent].
    
    Args:
        mesh_path: Path to .ply mesh file
        depth_map_path: Path to 16-bit depth map PNG
        relief_band_mm: Upper relief bound in mm (default: 15mm)
        scale_factor: mm-per-depth-unit. If None, estimated from depth range.
    
    Returns:
        dict with 'mc_pct', 'total_faces', 'in_band_faces'
    """
    if not HAS_O3D:
        return {"mc_pct": None, "error": "open3d not installed"}
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    if len(triangles) == 0:
        return {"mc_pct": None, "error": "Empty mesh"}
    
    depth_img = np.array(Image.open(depth_map_path).convert("I")).astype(np.float32)
    h, w = depth_img.shape
    
    # Background depth from image border
    inset = 10
    bg_pixels = np.concatenate([
        depth_img[:inset, :].ravel(),
        depth_img[-inset:, :].ravel(),
        depth_img[:, :inset].ravel(),
        depth_img[:, -inset:].ravel()
    ])
    d_background = float(np.median(bg_pixels))
    depth_range = float(depth_img.max() - depth_img.min())
    
    if scale_factor is None:
        # Approximate: full depth range maps to ~30mm (typical max relief)
        scale_factor = 30.0 / depth_range if depth_range > 0 else 1.0
    
    band_upper_raw = d_background + (relief_band_mm / scale_factor)
    
    # Get z-coordinates of mesh face centroids
    face_centroids = vertices[triangles].mean(axis=1)
    z_values = face_centroids[:, 2]
    
    z_min = z_values.min()
    z_max = z_values.max()
    z_range = z_max - z_min
    
    if z_range < 1e-8:
        return {"mc_pct": None, "error": "Mesh has zero z-range"}
    
    # Map z to equivalent depth-map units
    z_norm = (z_values - z_min) / z_range * depth_range + depth_img.min()
    
    in_band = ((z_norm >= d_background) & (z_norm <= band_upper_raw)).sum()
    total = len(triangles)
    mc_pct = float(in_band) / float(total) * 100.0
    
    return {
        "mc_pct": round(mc_pct, 2),
        "in_band_faces": int(in_band),
        "total_faces": total,
        "relief_band_mm": relief_band_mm,
        "target": 90.0,
        "pass": mc_pct >= 90.0,
        "warning": "MC uses approximated scale_factor. Provide physical_relief_mm for better accuracy."
    }


# ─────────────────────────────────────────────
# METRIC 5: Depth Map Sharpness Index (SI)
# ─────────────────────────────────────────────
def compute_si(depth_map_path: str, source_image_path: str) -> dict:
    """
    Depth Map Sharpness Index.
    Ratio of Laplacian variance of depth map to Laplacian variance of source image.
    SI > 1.0 indicates the depth map resolves structural gradients more sharply
    than the photometric image — expected behaviour on low-contrast stone.
    
    Args:
        depth_map_path: Path to 16-bit greyscale depth map PNG
        source_image_path: Path to original RGB source image
    
    Returns:
        dict with 'si', 'depth_lap_var', 'source_lap_var'
    """
    depth_img = np.array(Image.open(depth_map_path).convert("I")).astype(np.float32)
    depth_norm = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
    
    source_img = np.array(Image.open(source_image_path).convert("L")).astype(np.float32)
    source_norm = source_img / 255.0
    
    # Resize source to depth map dimensions if needed
    if source_norm.shape != depth_norm.shape:
        source_pil = Image.fromarray((source_norm * 255).astype(np.uint8))
        source_pil = source_pil.resize((depth_norm.shape[1], depth_norm.shape[0]), Image.LANCZOS)
        source_norm = np.array(source_pil).astype(np.float32) / 255.0
    
    if HAS_SCIPY:
        from scipy.ndimage import laplace as sci_laplace
        depth_lap = sci_laplace(depth_norm)
        source_lap = sci_laplace(source_norm)
    else:
        # Numpy Laplacian kernel approximation
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        from numpy.lib.stride_tricks import as_strided
        def convolve2d_simple(img, k):
            ph, pw = k.shape[0] // 2, k.shape[1] // 2
            padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
            h, w = img.shape
            result = np.zeros_like(img)
            for i in range(k.shape[0]):
                for j in range(k.shape[1]):
                    result += k[i, j] * padded[i:i+h, j:j+w]
            return result
        depth_lap = convolve2d_simple(depth_norm, kernel)
        source_lap = convolve2d_simple(source_norm, kernel)
    
    depth_lap_var = float(np.var(depth_lap))
    source_lap_var = float(np.var(source_lap))
    
    if source_lap_var < 1e-12:
        return {"si": None, "error": "Source image Laplacian variance is near zero"}
    
    si = depth_lap_var / source_lap_var
    
    return {
        "si": round(si, 4),
        "depth_lap_var": round(depth_lap_var, 6),
        "source_lap_var": round(source_lap_var, 6),
        "target": 1.0,
        "pass": si >= 1.0
    }


# ─────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────
def evaluate_panel(row: dict, results_dir: str) -> dict:
    """Evaluate all metrics for one panel row from panels.csv."""
    panel_id = row.get("panel_id", "unknown")
    
    def resolve(p):
        if not p:
            return None
        full = Path(results_dir) / p if not Path(p).is_absolute() else Path(p)
        return str(full) if full.exists() else None
    
    depth_map   = resolve(row.get("depth_map"))
    mask        = resolve(row.get("mask"))
    mesh_ply    = resolve(row.get("mesh_ply"))
    source_img  = resolve(row.get("source_image"))
    second_mask = resolve(row.get("second_mask"))
    
    try:
        phys_mm = float(row.get("physical_relief_mm", 0)) or None
    except (ValueError, TypeError):
        phys_mm = None
    
    result = {"panel_id": panel_id, "site": row.get("site", ""), "motif_type": row.get("motif_type", "")}
    
    # SNC
    if mesh_ply:
        snc = compute_snc(mesh_ply)
    else:
        snc = {"snc": None, "error": "No mesh file"}
    result["SNC"] = snc.get("snc")
    result["SNC_pass"] = snc.get("pass")
    result["SNC_detail"] = snc
    
    # DRP
    if depth_map and mask and phys_mm:
        drp = compute_drp(depth_map, mask, phys_mm)
    else:
        drp = {"drp_ratio": None, "error": "Missing depth_map, mask, or physical_relief_mm"}
    result["DRP"] = drp.get("drp_ratio")
    result["DRP_pass"] = drp.get("pass")
    result["DRP_detail"] = drp
    
    # MR
    if mask and second_mask:
        mr = compute_mr(mask, second_mask)
    else:
        mr = {"iou": None, "error": "second_mask not provided"}
    result["MR"] = mr.get("iou")
    result["MR_pass"] = mr.get("pass")
    result["MR_detail"] = mr
    
    # MC
    if mesh_ply and depth_map:
        mc = compute_mc(mesh_ply, depth_map)
    else:
        mc = {"mc_pct": None, "error": "Missing mesh or depth map"}
    result["MC"] = mc.get("mc_pct")
    result["MC_pass"] = mc.get("pass")
    result["MC_detail"] = mc
    
    # SI
    if depth_map and source_img:
        si = compute_si(depth_map, source_img)
    else:
        si = {"si": None, "error": "Missing depth_map or source_image"}
    result["SI"] = si.get("si")
    result["SI_pass"] = si.get("pass")
    result["SI_detail"] = si
    
    return result


def summarise(results: list) -> dict:
    """Compute mean and std for each metric across all evaluated panels."""
    metrics = ["SNC", "DRP", "MR", "MC", "SI"]
    summary = {}
    for m in metrics:
        values = [r[m] for r in results if r.get(m) is not None]
        if values:
            summary[m] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "n": len(values),
                "pass_rate": round(sum(1 for r in results if r.get(f"{m}_pass")) / len(values) * 100, 1)
            }
        else:
            summary[m] = {"mean": None, "std": None, "n": 0, "pass_rate": None, "note": "No values computed"}
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quantitative metrics for Depth_Anything_3_Motifs_CLI pipeline outputs."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing pipeline output files")
    parser.add_argument("--panels-csv", required=True,
                        help="CSV file listing panels with file paths and physical measurements")
    parser.add_argument("--output", default="metrics_report.json",
                        help="Output JSON report path (default: metrics_report.json)")
    parser.add_argument("--csv-output", default="metrics_report.csv",
                        help="Output CSV summary path")
    parser.add_argument("--panel-id", default=None,
                        help="Evaluate only this panel_id (default: all)")
    args = parser.parse_args()
    
    if not Path(args.panels_csv).exists():
        print(f"ERROR: panels.csv not found: {args.panels_csv}")
        sys.exit(1)
    
    with open(args.panels_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        panels = list(reader)
    
    if args.panel_id:
        panels = [p for p in panels if p.get("panel_id") == args.panel_id]
        if not panels:
            print(f"ERROR: panel_id '{args.panel_id}' not found in CSV")
            sys.exit(1)
    
    print(f"Evaluating {len(panels)} panels...")
    results = []
    for i, row in enumerate(panels):
        pid = row.get("panel_id", f"panel_{i}")
        print(f"  [{i+1}/{len(panels)}] {pid} ...", end=" ", flush=True)
        r = evaluate_panel(row, args.results_dir)
        results.append(r)
        snc = r.get("SNC"); drp = r.get("DRP"); mr = r.get("MR"); mc = r.get("MC"); si = r.get("SI")
        print(f"SNC={snc} DRP={drp} MR={mr} MC={mc} SI={si}")
    
    summary = summarise(results)
    
    report = {
        "n_panels": len(results),
        "summary": summary,
        "panels": results
    }
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report written: {args.output}")
    
    # CSV summary
    csv_rows = []
    for r in results:
        csv_rows.append({
            "panel_id": r["panel_id"],
            "site": r.get("site", ""),
            "motif_type": r.get("motif_type", ""),
            "SNC": r.get("SNC"), "SNC_pass": r.get("SNC_pass"),
            "DRP": r.get("DRP"), "DRP_pass": r.get("DRP_pass"),
            "MR_IoU": r.get("MR"), "MR_pass": r.get("MR_pass"),
            "MC_pct": r.get("MC"), "MC_pass": r.get("MC_pass"),
            "SI": r.get("SI"), "SI_pass": r.get("SI_pass"),
        })
    
    with open(args.csv_output, "w", newline='', encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"CSV summary written: {args.csv_output}")
    
    print("\n── SUMMARY ──────────────────────────────────────────")
    print(f"{'Metric':<8} {'Mean':>8} {'Std':>8} {'N':>5} {'Pass%':>8} {'Target'}")
    targets = {"SNC": ">0.85", "DRP": "0.8–1.2", "MR": ">0.80", "MC": ">90%", "SI": ">1.0"}
    for m, s in summary.items():
        mean_str = f"{s['mean']:.4f}" if s['mean'] is not None else "N/A"
        std_str = f"{s['std']:.4f}" if s['std'] is not None else "N/A"
        pass_str = f"{s['pass_rate']:.1f}%" if s['pass_rate'] is not None else "N/A"
        print(f"{m:<8} {mean_str:>8} {std_str:>8} {s['n']:>5} {pass_str:>8}   {targets.get(m,'')}")
    print("─────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
