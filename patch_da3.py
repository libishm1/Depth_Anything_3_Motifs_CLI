"""
Run this once to patch DA3 so it stops requiring unused dependencies.
Usage: py patch_da3.py
"""
import sys
import importlib.util

# Find DA3 install location
import os, site

base = None
for sp in site.getsitepackages():
    candidate = os.path.join(sp, "depth_anything_3")
    if os.path.isdir(candidate):
        base = candidate
        break

if base is None:
    # also check user site
    candidate = os.path.join(site.getusersitepackages(), "depth_anything_3")
    if os.path.isdir(candidate):
        base = candidate

if base is None:
    print("ERROR: depth_anything_3 folder not found in site-packages.")
    sys.exit(1)

print(f"DA3 found at: {base}")

patches = {
    os.path.join(base, "utils", "export", "__init__.py"): [
        ("from depth_anything_3.utils.export.gs import",
         "# from depth_anything_3.utils.export.gs import"),
        ("from .gs import",
         "# from .gs import"),
        ("from .colmap import",
         "# from .colmap import"),
        ("from depth_anything_3.utils.export.colmap import",
         "# from depth_anything_3.utils.export.colmap import"),
    ],
    os.path.join(base, "api.py"): [
        ("from depth_anything_3.utils.pose_align import",
         "# from depth_anything_3.utils.pose_align import"),
    ],
}

for filepath, replacements in patches.items():
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        txt = f.read()
    for old, new in replacements:
        if old in txt:
            txt = txt.replace(old, new)
            print(f"  Patched: {old[:60]}...")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  Done: {filepath}")

print("\nPatch complete. Run your script now.")
