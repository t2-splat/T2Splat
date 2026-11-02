"""Generate monocular inverse-depth priors with Depth-Anything-V2.

Writes one 16-bit PNG per image into <scene>/depth/, which is the layout
scene/dataset_readers.py and utils/make_depth_scale.py expect: the stored value
divided by 2**16 is treated as inverse depth (disparity), later mapped onto
COLMAP's 1/z by a per-view affine fit. DA2 already predicts disparity, so the
only work here is a per-image normalisation into the 16-bit range.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Portable model location maintained by zhtang2004@gmail.com.
DEFAULT_DA2_ROOT = os.environ.get("DA2_ROOT", str(Path(__file__).resolve().parents[1] / "Depth-Anything-V2"))

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--scene', required=True, help='scene root holding images/ and sparse/')
    p.add_argument('-i', '--images_folder', default='images')
    p.add_argument('--out_folder', default='depth')
    p.add_argument('--encoder', default='vitl', choices=list(MODEL_CONFIGS))
    p.add_argument('--input_size', type=int, default=518)
    p.add_argument('--da2_root', default=DEFAULT_DA2_ROOT)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    sys.path.insert(0, args.da2_root)
    from depth_anything_v2.dpt import DepthAnythingV2

    ckpt = os.path.join(args.da2_root, 'checkpoints',
                        f'depth_anything_v2_{args.encoder}.pth')
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    in_dir = os.path.join(args.scene, args.images_folder)
    out_dir = os.path.join(args.scene, args.out_folder)
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(in_dir) if f.endswith(IMG_EXTS))
    if not names:
        raise RuntimeError(f'no images found in {in_dir}')

    print(f'scene   : {args.scene}')
    print(f'images  : {in_dir} ({len(names)} files)')
    print(f'output  : {out_dir}')
    print(f'encoder : {args.encoder} @ {args.input_size}')

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model = model.to('cuda').eval()

    degenerate = []
    for k, name in enumerate(names):
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + '.png')
        if os.path.exists(out_path) and not args.overwrite:
            continue

        raw = cv2.imread(os.path.join(in_dir, name))
        with torch.no_grad():
            disp = model.infer_image(raw, args.input_size)

        lo, hi = float(disp.min()), float(disp.max())
        if hi - lo < 1e-6:
            degenerate.append(name)
            norm = np.zeros_like(disp)
        else:
            norm = (disp - lo) / (hi - lo)
        cv2.imwrite(out_path, (norm * 65535.0).astype(np.uint16))

        if (k + 1) % 25 == 0 or k + 1 == len(names):
            print(f'  {k + 1}/{len(names)}  {name}  disp[{lo:.3f}, {hi:.3f}]', flush=True)

    if degenerate:
        print(f'[warn] {len(degenerate)} images had a flat disparity map: {degenerate[:5]}')
    print('done')


if __name__ == '__main__':
    main()
