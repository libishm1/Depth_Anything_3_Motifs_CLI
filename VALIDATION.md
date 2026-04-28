# Reproducing the Section 6.6 Mesh Validation

This section explains how to independently verify the mesh-only quantitative
evaluation table reported in Section 6.6 of the accompanying paper.

## What the validation reports

Per-panel metrics across 35 unique reconstructed mesh panels:

- **SNC** (Surface Normal Consistency) on decimated meshes — median 0.985 across the corpus.
- **Relief amplitude** (1st-to-99th-percentile spread of the Z coordinate distribution, in normalised mesh units) — median 0.293 across the corpus.
- **Mesh integrity diagnostics**: vertex/face count, watertightness, winding consistency.
- **Decimation provenance**: original triangle count, decimation method used per panel.

Mask-based metrics (mesh completeness, mask repeatability), the depth-map sharpness index, and metric-depth plausibility (DRP) are not computed by this script. These require per-panel binary masks and raw single-channel depth maps, which were not exported in the first acquisition. They are reserved for the second field acquisition with caliper-measured ground truth, as documented in Section 7.5 of the paper.

## Required inputs

The mesh corpus is hosted on Zenodo at:

> https://doi.org/10.5281/zenodo.19846595

Download the three site zip files and `mesh_metrics_updated.csv` from that record.

## Setup

```bash
git clone https://github.com/libishm1/Depth_Anything_3_Motifs_CLI.git
cd Depth_Anything_3_Motifs_CLI
pip install -r requirements_validation.txt
```

Tested on Python 3.10. The validation script is CPU-only.

## Run

Extract the three zip files into folders matching their archive names:

```
working_directory/
├── thanjavur_meshes/
├── kalamangalam_meshes/
└── namakkal_meshes/
```

Run the evaluator:

```bash
python eval_meshes_only.py \
  --input ./thanjavur_meshes ./kalamangalam_meshes ./namakkal_meshes \
  --output ./eval_results
```

Expected runtime: approximately 4 to 6 hours wall time on consumer hardware (32 GB RAM, Python 3.10, no GPU). Median time per panel is around 150 seconds; the largest meshes (1.0 to 1.8 GB) take 5 to 10 minutes each because of the vertex-clustering pre-decimation step.

For a quick functional smoke test on a single site:

```bash
python eval_meshes_only.py \
  --input ./thanjavur_meshes \
  --output ./test_run \
  --limit 3
```

## Comparing against the released CSV

After the run completes, compare `./eval_results/mesh_metrics.csv` against `mesh_metrics_updated.csv` from the Zenodo release. The two should match within floating-point rounding for SNC and within ±0.001 for relief amplitude. Decimation method strings will be identical because the staging logic is deterministic given a fixed numpy RNG seed.

If your reproduced CSV diverges meaningfully, please open an issue describing the panel(s) that differ. Hardware variation in Open3D's quadric decimation can produce tiny SNC differences (4th decimal place); larger differences usually indicate a different mesh input or a different Open3D version.

## Hardware notes

- Tested on: NVIDIA Quadro RTX 4000 (8 GB VRAM), 32 GB system RAM, Windows 11, CUDA 11.8, Python 3.10.
- The script does not use the GPU. Open3D's quadric decimation is CPU-only.
- For meshes under 1 GB, 16 GB system RAM is sufficient. For meshes over 1 GB (the larger Thanjavur and Namakkal STLs), 32 GB is recommended; the vertex-clustering pre-decimation step will OOM on 16 GB systems for the very largest panels.
- Closing other applications (browser, IDE) before running is advisable on 32 GB systems to maximise available memory.
