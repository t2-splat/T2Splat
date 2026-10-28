"""Resize mono-depth prior PNGs to the resolution train.py actually renders at.

`utils/camera_utils.loadCam` resizes the RGB image but loads `depth/*.png` at its
native resolution, so the priors must be pre-resized or the depth loss raises a
shape mismatch. This replicates the resolution rule in `loadCam`.

Usage:
    python utils/resize_depth_priors.py -s data/TNT/Truck -i images
    python utils/resize_depth_priors.py -s data/TNT/Truck -i images --resolution 1
"""
import argparse
import os
import shutil

import cv2
from PIL import Image


def target_resolution(orig_w, orig_h, resolution):
    if resolution in (1, 2, 4, 8):
        down = float(resolution)
    elif resolution == -1:
        down = orig_w / 1600 if orig_w > 1600 else 1.0
    else:
        down = orig_w / resolution
    return int(orig_w / down), int(orig_h / down)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("-i", "--images", default="images")
    p.add_argument("--depth_dir", default="depth")
    p.add_argument("--resolution", type=int, default=-1,
                   help="same semantics as train.py --resolution")
    p.add_argument("--backup_dir", default=None,
                   help="where to move the original full-res PNGs (default <depth_dir>_fullres)")
    args = p.parse_args()

    img_dir = os.path.join(args.source_path, args.images)
    depth_dir = os.path.join(args.source_path, args.depth_dir)
    backup = args.backup_dir or (depth_dir + "_fullres")

    names = sorted(os.listdir(depth_dir))
    if not names:
        raise SystemExit(f"no depth maps in {depth_dir}")

    os.makedirs(backup, exist_ok=True)
    resized = skipped = 0
    target = None
    for name in names:
        stem = os.path.splitext(name)[0]
        src_img = next((os.path.join(img_dir, stem + e)
                        for e in (".jpg", ".JPG", ".jpeg", ".png", ".PNG")
                        if os.path.exists(os.path.join(img_dir, stem + e))), None)
        if src_img is None:
            print(f"[skip] no matching image for {name}")
            skipped += 1
            continue

        with Image.open(src_img) as im:
            orig_w, orig_h = im.size
        target = target_resolution(orig_w, orig_h, args.resolution)

        path = os.path.join(depth_dir, name)
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            print(f"[skip] unreadable {name}")
            skipped += 1
            continue
        if (d.shape[1], d.shape[0]) == target:
            skipped += 1
            continue

        shutil.move(path, os.path.join(backup, name))
        cv2.imwrite(path, cv2.resize(d, target, interpolation=cv2.INTER_AREA))
        resized += 1

    print(f"resized {resized}, unchanged/skipped {skipped}, target={target}")
    print(f"originals kept in {backup}")


if __name__ == "__main__":
    main()
