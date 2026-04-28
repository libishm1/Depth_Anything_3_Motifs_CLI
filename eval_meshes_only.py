r"""
eval_meshes_only.py
-------------------
Mesh-only quantitative evaluation for the Depth_Anything_3_Motifs_CLI corpus.

Reproduces the per-panel validation table reported in Section 6.6 of the
accompanying paper, computed against the consolidated 35-panel mesh corpus
released at https://doi.org/10.5281/zenodo.19846595.

Design goals
------------
- Run on consumer laptops (32 GB RAM tested) without crashing on large STLs.
- Stream meshes one at a time; never load two simultaneously.
- Decimate aggressively (target 100,000 triangles by default) BEFORE
  computing any metric. Very large meshes (>5M triangles) are pre-decimated
  via numpy-based vertex clustering to keep peak memory bounded, then
  passed to Open3D for quadric edge-collapse decimation in 4x stages.
- Write results row-by-row to CSV so partial runs are preserved.

Computed metrics
----------------
- SNC (surface normal consistency) on the decimated mesh
- Relief amplitude: z-range, 1st-to-99th-percentile spread, std
- Bounding box and aspect ratios
- Mesh integrity: vertex/face count, watertight, winding consistency

Usage
-----
1. Download the consolidated mesh corpus from Zenodo:
   https://doi.org/10.5281/zenodo.19846595

2. Extract the three site zip files into local folders:
     thanjavur_meshes/
     kalamangalam_meshes/
     namakkal_meshes/

3. Run the evaluator on all three folders:
     python eval_meshes_only.py \
       --input ./thanjavur_meshes ./kalamangalam_meshes ./namakkal_meshes \
       --output ./eval_results

4. Compare ./eval_results/mesh_metrics.csv against the released
   mesh_metrics_updated.csv (same DOI). The two should match within
   floating-point rounding for SNC and within +/-0.001 for relief
   amplitude.

For a quick smoke test on a single folder:
     python eval_meshes_only.py --input ./thanjavur_meshes --output ./test --limit 3

Dependencies
------------
See requirements_validation.txt:
    trimesh>=4.0
    numpy>=1.24
    open3d>=0.18

Open3D is required for quadric decimation. Install with:
    pip install -r requirements_validation.txt

Hardware requirements
---------------------
Tested on: NVIDIA Quadro RTX 4000 (8 GB VRAM), 32 GB system RAM, Windows 11.
The script is CPU-only; the GPU is not used. Open3D's quadric decimation
is the memory-hungry step. For meshes <1 GB, 16 GB system RAM is sufficient.
For meshes >1 GB, 32 GB is recommended.

Reproducibility notes
---------------------
- Runtime: approximately 4-6 hours wall time for the full 35-panel corpus
  on the tested hardware. Median per-panel time is ~150 seconds.
- Determinism: vertex clustering uses a fixed-seed numpy RNG (seed=42) for
  the SNC pair sampling step. Quadric decimation in Open3D is deterministic.
  Two runs against the same input should produce bit-identical CSV output.

License: MIT (see LICENSE in the repository root).
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except ImportError:
    print("ERROR: trimesh not installed. Run: pip install trimesh numpy")
    sys.exit(1)

# Optional: open3d for faster decimation. Falls back to trimesh if absent.
try:
    import open3d as o3d
    HAVE_OPEN3D = True
except ImportError:
    HAVE_OPEN3D = False


MESH_EXTS = {".stl", ".ply", ".obj"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_meshes(input_dirs: List[Path], skip_substrings: List[str]) -> List[Tuple[str, Path]]:
    """Return list of (site, path). Site is the first dir under any of the input_dirs."""
    panels = []
    for root in input_dirs:
        if not root.exists():
            print(f"  WARN: {root} does not exist, skipping")
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in MESH_EXTS:
                continue
            name_lower = p.name.lower()
            if any(skip in name_lower for skip in skip_substrings):
                continue
            # Site = first relative dir component, or the root name itself if mesh sits at root.
            try:
                rel = p.relative_to(root)
                if len(rel.parts) > 1:
                    site = rel.parts[0]
                else:
                    site = root.name
            except ValueError:
                site = root.name
            panels.append((site, p))
    return panels


# ---------------------------------------------------------------------------
# Loading and decimation
# ---------------------------------------------------------------------------

def load_and_decimate(mesh_path: Path, target_tris: int, max_load_bytes: int) -> Tuple[Optional[trimesh.Trimesh], Dict]:
    """
    Load mesh and decimate to target_tris if larger.
    Returns (mesh, info) or (None, info) on failure.
    """
    info = {
        "size_bytes": mesh_path.stat().st_size,
        "skipped_too_large": False,
        "load_error": None,
        "original_tris": 0,
        "decimated_tris": 0,
        "decimation_method": "none",
    }

    if info["size_bytes"] > max_load_bytes:
        info["skipped_too_large"] = True
        info["load_error"] = f"file size {info['size_bytes']/1e9:.2f} GB exceeds --max-mesh-bytes"
        return None, info

    try:
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    except Exception as e:
        info["load_error"] = f"trimesh load failed: {type(e).__name__}: {e}"
        return None, info

    if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        info["load_error"] = "mesh empty or invalid"
        return None, info

    info["original_tris"] = int(len(mesh.faces))

    # Decimate if too dense.
    if info["original_tris"] > target_tris:
        if not HAVE_OPEN3D:
            info["load_error"] = ("open3d required for valid decimation. "
                                  "Install with: pip install open3d. "
                                  "Random face-subsampling fallback removed because it destroys "
                                  "mesh adjacency and breaks SNC.")
            return None, info

        decimated = _open3d_decimate_progressive(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_tris,
            info,
        )
        if decimated is None:
            return None, info

        new_verts, new_faces = decimated
        # Free the original mesh memory before building the decimated trimesh.
        del mesh
        gc.collect()
        mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

    # IMPORTANT: build adjacency and clean up degenerate faces. process=True does this.
    # We do it once at the end so it runs on the small decimated mesh, not the huge original.
    try:
        mesh.process(validate=True)
    except Exception as e:
        info["load_error"] = f"mesh.process failed: {type(e).__name__}: {e}"
        return None, info

    info["decimated_tris"] = int(len(mesh.faces))
    return mesh, info


def _vertex_cluster_decimate(verts: np.ndarray, faces: np.ndarray,
                             target_tris: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-bounded mesh simplification by vertex clustering on a regular 3D grid.

    Bins vertices into a uniform grid with bin size chosen so the resulting
    coarsened mesh has roughly target_tris triangles. All vertices in one bin
    collapse to the bin centroid; faces with collapsed-vertex repeats are dropped.

    This runs entirely in numpy and never copies the full mesh, so it works on
    23M-triangle inputs that crash Open3D's quadric decimation.

    Quality is lower than quadric decimation (geometric features are clamped to
    grid resolution), but it's the only stable preprocessing for meshes that
    won't fit Open3D's working memory.
    """
    # Pick bin size: grid cells along the longest axis. We want ~ target_tris faces
    # remaining, which empirically corresponds to ~sqrt(target_tris/2) cells along
    # the dominant axis for a heightfield-like mesh.
    bbox = verts.max(axis=0) - verts.min(axis=0)
    longest = float(bbox.max())
    n_cells = max(int(np.sqrt(target_tris / 2)) * 2, 64)
    bin_size = longest / n_cells

    # Discretise vertices to bin indices.
    origin = verts.min(axis=0)
    bin_idx = np.floor((verts - origin) / bin_size).astype(np.int64)
    # Pack 3D bin index into single int64 key.
    # Use generous offsets to handle negative coordinates after floor.
    max_bin = int(bin_idx.max()) + 1
    keys = (bin_idx[:, 0].astype(np.int64) * max_bin * max_bin
            + bin_idx[:, 1].astype(np.int64) * max_bin
            + bin_idx[:, 2].astype(np.int64))

    # Map each vertex to its cluster id (0..n_clusters-1).
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_clusters = len(unique_keys)

    # Cluster centroids: mean of vertices falling into each cluster.
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    new_verts = np.zeros((n_clusters, 3), dtype=np.float64)
    np.add.at(new_verts, inverse, verts)
    new_verts /= counts[:, None]

    # Remap face indices through the cluster map.
    new_faces = inverse[faces]

    # Drop degenerate faces: any face where two or three corners hit the same cluster.
    a, b, c = new_faces[:, 0], new_faces[:, 1], new_faces[:, 2]
    keep = (a != b) & (b != c) & (a != c)
    new_faces = new_faces[keep]

    # Free the index arrays we no longer need.
    del bin_idx, keys, inverse, counts
    gc.collect()

    return new_verts, new_faces.astype(np.int32)


def _open3d_decimate_progressive(verts: np.ndarray, faces: np.ndarray,
                                 target_tris: int, info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Progressive quadric decimation with memory-bounded preconditioning.

    For very large meshes (>5M triangles), runs vertex clustering in numpy first
    to bring the mesh under Open3D's working-memory ceiling, then does quadric
    decimation in 4x stages.
    """
    n_orig = len(faces)
    method_log = []

    # Pre-decimation step for meshes too big for Open3D to load comfortably.
    # 5M triangles is the empirical ceiling for ~32 GB RAM with other processes running.
    PREDECIM_THRESHOLD = 5_000_000
    PREDECIM_TARGET = 1_500_000  # well under the threshold for the next stage

    if n_orig > PREDECIM_THRESHOLD:
        try:
            verts, faces = _vertex_cluster_decimate(verts, faces, PREDECIM_TARGET)
            method_log.append(f"vertex_cluster:{n_orig}->{len(faces)}")
            gc.collect()
        except MemoryError:
            info["load_error"] = ("vertex clustering hit MemoryError on input mesh. "
                                  "Try --target-tris 25000 or process this mesh on a higher-RAM machine.")
            return None
        except Exception as e:
            info["load_error"] = f"vertex clustering failed: {type(e).__name__}: {e}"
            return None

        if len(faces) <= target_tris:
            # Pre-decimation already brought us under the target; skip Open3D entirely.
            info["decimation_method"] = "+".join(method_log)
            return verts, faces

    # Stage plan: 4x reduction per stage on whatever we have now.
    n_now = len(faces)
    stages = []
    cur = n_now
    while cur > target_tris * 4:
        cur = cur // 4
        stages.append(cur)
    stages.append(target_tris)

    try:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        # Free the numpy copies; o3d has its own.
        del verts, faces
        gc.collect()

        for stage_target in stages:
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=int(stage_target))
            method_log.append(f"o3d_quadric:{stage_target}")
            gc.collect()

        out_verts = np.asarray(o3d_mesh.vertices)
        out_faces = np.asarray(o3d_mesh.triangles)
        del o3d_mesh
        gc.collect()

        info["decimation_method"] = "+".join(method_log)
        return out_verts, out_faces

    except MemoryError:
        info["load_error"] = ("open3d quadric decimation hit MemoryError after pre-decimation. "
                              f"Working size at failure: {n_now} triangles. "
                              "Try a smaller --target-tris (e.g. 50000 or 25000).")
        return None
    except Exception as e:
        info["load_error"] = f"open3d quadric decimation failed: {type(e).__name__}: {e}"
        return None




# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_snc_mesh(mesh: trimesh.Trimesh, max_pairs: int = 200_000) -> Dict:
    """
    Surface Normal Consistency from triangle adjacency on the decimated mesh.
    Mean cosine similarity between neighboring face normals.
    Caps work at max_pairs to keep memory bounded.
    """
    try:
        face_normals = mesh.face_normals  # property; computed lazily
        adj = mesh.face_adjacency  # (M, 2) array of face index pairs
    except Exception as e:
        return {"snc_mesh": None, "snc_n_pairs": 0, "snc_below_85_pct": None,
                "snc_error": f"{type(e).__name__}: {e}"}

    if adj is None or len(adj) == 0:
        return {"snc_mesh": None, "snc_n_pairs": 0, "snc_below_85_pct": None,
                "snc_error": "no face adjacency"}

    if len(adj) > max_pairs:
        idx = np.random.default_rng(42).choice(len(adj), size=max_pairs, replace=False)
        adj = adj[idx]

    n1 = face_normals[adj[:, 0]]
    n2 = face_normals[adj[:, 1]]
    cos = np.einsum("ij,ij->i", n1, n2)
    cos = np.clip(cos, -1.0, 1.0)

    return {
        "snc_mesh": round(float(np.mean(cos)), 4),
        "snc_n_pairs": int(len(cos)),
        "snc_below_85_pct": round(float(np.mean(cos < 0.85) * 100), 2),
        "snc_error": None,
    }


def compute_relief_stats(mesh: trimesh.Trimesh) -> Dict:
    """
    Relief amplitude statistics in mesh units (typically mm if pipeline preserved scale,
    otherwise relative). Reports z-range and percentile spreads.
    """
    try:
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            return {"relief_range": None, "relief_p1_p99": None, "relief_std": None,
                    "bbox_x": None, "bbox_y": None, "bbox_z": None,
                    "aspect_xy": None, "aspect_zx": None}
        z = verts[:, 2]
        x = verts[:, 0]
        y = verts[:, 1]
        bbox_x = float(x.max() - x.min())
        bbox_y = float(y.max() - y.min())
        bbox_z = float(z.max() - z.min())
        return {
            "relief_range": round(float(z.max() - z.min()), 4),
            "relief_p1_p99": round(float(np.percentile(z, 99) - np.percentile(z, 1)), 4),
            "relief_std": round(float(np.std(z)), 4),
            "bbox_x": round(bbox_x, 4),
            "bbox_y": round(bbox_y, 4),
            "bbox_z": round(bbox_z, 4),
            "aspect_xy": round(bbox_x / bbox_y, 4) if bbox_y > 0 else None,
            "aspect_zx": round(bbox_z / max(bbox_x, bbox_y), 4) if max(bbox_x, bbox_y) > 0 else None,
        }
    except Exception as e:
        return {"relief_range": None, "relief_p1_p99": None, "relief_std": None,
                "bbox_x": None, "bbox_y": None, "bbox_z": None,
                "aspect_xy": None, "aspect_zx": None,
                "relief_error": f"{type(e).__name__}: {e}"}


def compute_mesh_health(mesh: trimesh.Trimesh) -> Dict:
    """Basic mesh diagnostics."""
    try:
        return {
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "euler": int(mesh.euler_number) if hasattr(mesh, "euler_number") else None,
            "n_verts": int(len(mesh.vertices)),
            "n_faces": int(len(mesh.faces)),
        }
    except Exception as e:
        return {"is_watertight": None, "is_winding_consistent": None, "euler": None,
                "n_verts": int(len(mesh.vertices)) if hasattr(mesh, "vertices") else 0,
                "n_faces": int(len(mesh.faces)) if hasattr(mesh, "faces") else 0,
                "health_error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Per-panel evaluation
# ---------------------------------------------------------------------------

def evaluate_panel(site: str, mesh_path: Path, target_tris: int, max_load_bytes: int) -> Dict:
    panel_id = mesh_path.stem
    row: Dict = {
        "panel_id": panel_id,
        "site": site,
        "mesh_path": str(mesh_path),
    }
    t0 = time.time()
    mesh, load_info = load_and_decimate(mesh_path, target_tris, max_load_bytes)
    row.update(load_info)

    if mesh is None:
        row["snc_mesh"] = None
        row["relief_range"] = None
        row["eval_seconds"] = round(time.time() - t0, 2)
        row["status"] = "skipped" if load_info.get("skipped_too_large") else "load_failed"
        return row

    row.update(compute_snc_mesh(mesh))
    row.update(compute_relief_stats(mesh))
    row.update(compute_mesh_health(mesh))

    row["eval_seconds"] = round(time.time() - t0, 2)
    row["status"] = "ok"

    # Free this mesh before returning. The caller will gc.
    del mesh
    return row


# ---------------------------------------------------------------------------
# CSV row writer
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "panel_id", "site", "mesh_path", "size_bytes",
    "original_tris", "decimated_tris", "decimation_method",
    "snc_mesh", "snc_n_pairs", "snc_below_85_pct", "snc_error",
    "relief_range", "relief_p1_p99", "relief_std",
    "bbox_x", "bbox_y", "bbox_z", "aspect_xy", "aspect_zx",
    "is_watertight", "is_winding_consistent", "euler", "n_verts", "n_faces",
    "skipped_too_large", "load_error", "eval_seconds", "status",
]


def write_csv_header(csv_path: Path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()


def append_csv_row(csv_path: Path, row: Dict):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Site-level summary
# ---------------------------------------------------------------------------

def summarise_by_site(rows: List[Dict]) -> Dict:
    by_site: Dict[str, List[Dict]] = {}
    for r in rows:
        by_site.setdefault(r["site"], []).append(r)

    summary = {}
    for site, rs in by_site.items():
        ok_rows = [r for r in rs if r.get("status") == "ok"]
        snc_vals = [r["snc_mesh"] for r in ok_rows if r.get("snc_mesh") is not None]
        relief_vals = [r["relief_range"] for r in ok_rows if r.get("relief_range") is not None]
        summary[site] = {
            "n_total": len(rs),
            "n_ok": len(ok_rows),
            "n_skipped_too_large": sum(1 for r in rs if r.get("skipped_too_large")),
            "n_load_failed": sum(1 for r in rs if r.get("status") == "load_failed"),
            "snc_median": round(float(np.median(snc_vals)), 4) if snc_vals else None,
            "snc_mean": round(float(np.mean(snc_vals)), 4) if snc_vals else None,
            "snc_min": round(float(np.min(snc_vals)), 4) if snc_vals else None,
            "snc_max": round(float(np.max(snc_vals)), 4) if snc_vals else None,
            "relief_range_median": round(float(np.median(relief_vals)), 4) if relief_vals else None,
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Mesh-only quantitative evaluator (laptop-safe).")
    ap.add_argument("--input", nargs="+", required=True,
                    help="One or more input folders. Recursive search.")
    ap.add_argument("--output", required=True, help="Output folder for CSV + JSON.")
    ap.add_argument("--target-tris", type=int, default=100_000,
                    help="Decimate to this triangle count if larger. Default 100000.")
    ap.add_argument("--max-mesh-bytes", type=float, default=3e9,
                    help="Skip meshes larger than this. Default 3 GB.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N meshes (for testing). 0 = no limit.")
    ap.add_argument("--skip-substring", nargs="*",
                    default=["scaled", "_decim", "_lite", "_repaired"],
                    help="Skip filenames containing any of these substrings.")
    args = ap.parse_args()

    input_dirs = [Path(d).resolve() for d in args.input]
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mesh_metrics.csv"
    json_summary_path = output_dir / "mesh_metrics_summary.json"
    log_path = output_dir / "run_log.txt"

    print(f"Backend: {'open3d (preferred)' if HAVE_OPEN3D else 'trimesh + face-subsample fallback'}")
    print(f"Input dirs: {[str(d) for d in input_dirs]}")
    print(f"Output dir: {output_dir}")
    print(f"Decimation target: {args.target_tris:,} triangles")
    print(f"Max mesh size: {args.max_mesh_bytes/1e9:.2f} GB")
    print()

    panels = discover_meshes(input_dirs, skip_substrings=[s.lower() for s in args.skip_substring])
    if args.limit > 0:
        panels = panels[:args.limit]

    if not panels:
        print("No meshes found. Check --input paths.")
        return

    print(f"Discovered {len(panels)} meshes across {len(set(p[0] for p in panels))} sites")
    print(f"Sites: {sorted(set(p[0] for p in panels))}")
    print()

    write_csv_header(csv_path)
    rows: List[Dict] = []

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"Run start: {time.ctime()}\n")
        logf.write(f"Total panels: {len(panels)}\n\n")
        logf.flush()

        for i, (site, mp) in enumerate(panels, 1):
            print(f"[{i}/{len(panels)}] {site}/{mp.name} ({mp.stat().st_size/1e6:.1f} MB) ... ", end="", flush=True)
            try:
                row = evaluate_panel(site, mp, args.target_tris, int(args.max_mesh_bytes))
            except Exception as e:
                row = {
                    "panel_id": mp.stem, "site": site, "mesh_path": str(mp),
                    "status": "exception", "load_error": f"{type(e).__name__}: {e}",
                }
                logf.write(f"EXCEPTION on {mp}:\n{traceback.format_exc()}\n\n")
                logf.flush()

            append_csv_row(csv_path, row)
            rows.append(row)

            status = row.get("status", "?")
            snc = row.get("snc_mesh")
            relief = row.get("relief_range")
            print(f"{status}", end="")
            if snc is not None:
                print(f"  SNC={snc:.3f}  relief={relief}", end="")
            print(f"  [{row.get('eval_seconds', '?')}s]")

            # Hard memory cleanup between panels.
            gc.collect()

        logf.write(f"\nRun end: {time.ctime()}\n")

    # Site-level summary.
    summary = summarise_by_site(rows)
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_panels_total": len(rows),
            "n_panels_ok": sum(1 for r in rows if r.get("status") == "ok"),
            "n_panels_skipped_size": sum(1 for r in rows if r.get("skipped_too_large")),
            "n_panels_load_failed": sum(1 for r in rows if r.get("status") == "load_failed"),
            "by_site": summary,
        }, f, indent=2)

    print()
    print(f"=== Done ===")
    print(f"Per-panel CSV: {csv_path}")
    print(f"Site summary:  {json_summary_path}")
    print(f"Run log:       {log_path}")
    print()
    print("Site summary preview:")
    for site, s in summary.items():
        print(f"  {site}: ok={s['n_ok']}/{s['n_total']}  "
              f"SNC_median={s['snc_median']}  "
              f"relief_median={s['relief_range_median']}  "
              f"skipped(too_large)={s['n_skipped_too_large']}")


if __name__ == "__main__":
    main()
