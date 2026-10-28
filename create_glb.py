#!/usr/bin/env python3
"""Export a standalone, statically textured GLB from a T2-Splat checkpoint."""
import argparse
from pathlib import Path

from utils.glb_export import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Model root, iteration directory, or checkpoint .pt")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration under a model root; -1 selects latest")
    parser.add_argument("--out", type=Path, required=True, help="Output .glb (all PNG atlases are embedded)")
    parser.add_argument("--color-mode", choices=("visible-mean", "dc"), default="visible-mean",
                        help="Bake SH over visible training cameras, or use DC only (CPU, lower fidelity)")
    parser.add_argument("--cameras", type=Path, help="Saved cameras.json; defaults to the model root")
    parser.add_argument("--holdout", type=int, default=8,
                        help="Exclude every Nth camera after sorting by img_name; 0 uses all cameras")
    parser.add_argument("--train-list", type=Path,
                        help="Explicit training img_name list, one per line; overrides --holdout")
    parser.add_argument("--visibility-max-size", type=int, default=1600,
                        help="Longest side for visibility before supersampling; no training images needed")
    parser.add_argument("--visibility-supersampling", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="CUDA device for visible-mean, e.g. cuda:0")
    parser.add_argument("--atlas-size", type=int, default=4096, help="Power-of-two PNG atlas width/height")
    parser.add_argument("--padding", type=int, default=2, help="Edge replication around each face tile")
    parser.add_argument("--mip-level", type=int, default=0,
                        help="Bake this native per-face mip level (0 preserves full trained resolution)")
    parser.add_argument("--chunk-faces", type=int, default=8192, help="Maximum texture batch size on CPU")
    parser.add_argument("--zup", action="store_true", help="Rotate Z-up input to glTF Y-up")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing GLB and export report")
    args = parser.parse_args()
    try:
        report = export_checkpoint(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        parser.exit(1, f"GLB export failed: {error}\n")
    print(f"Saved {args.out}: {report['faces']:,} faces, {report['atlas_count']} atlases, "
          f"{report['bytes'] / 1e6:.2f} MB", flush=True)
    print("Static RGB bake: view-dependent SH and native per-face LOD/blending are not preserved.")


if __name__ == "__main__":
    main()
