"""Use lingbot-depth to refine raw depth and generate PLY point cloud.

Usage:
    uv run python refine_ply.py output/color_20260311_104008.png output/depth_raw_20260311_104008.npy

Intrinsics are read from the camera if available, or you can pass them manually:
    uv run python refine_ply.py rgb.png depth.npy --fx 386 --fy 386 --cx 320 --cy 240
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from mdm.model.v2 import MDMModel


def main():
    parser = argparse.ArgumentParser(description="Refine depth with lingbot-depth and export PLY")
    parser.add_argument("rgb", help="Path to RGB image (PNG/JPG)")
    parser.add_argument("depth", help="Path to raw depth (.npy uint16 in mm)")
    parser.add_argument("--fx", type=float, default=386.0, help="Depth camera fx")
    parser.add_argument("--fy", type=float, default=386.0, help="Depth camera fy")
    parser.add_argument("--cx", type=float, default=320.0, help="Depth camera cx")
    parser.add_argument("--cy", type=float, default=240.0, help="Depth camera cy")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
                        help="HuggingFace model ID or local path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load RGB ---
    image_bgr = cv2.imread(args.rgb)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB: {args.rgb}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"RGB: {w}x{h}")

    image_tensor = torch.tensor(
        image_rgb / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # --- Load Depth ---
    depth_raw = np.load(args.depth)  # uint16, mm
    depth_m = depth_raw.astype(np.float32) / 1000.0  # convert to meters
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)

    # Resize depth to match RGB if needed
    dh, dw = depth_m.shape
    if (dh, dw) != (h, w):
        print(f"Resizing depth {dw}x{dh} -> {w}x{h} to match RGB")
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

    print(f"Depth range: {depth_m[depth_m > 0].min():.3f} - {depth_m.max():.3f} m")

    depth_tensor = torch.tensor(depth_m, dtype=torch.float32, device=device)

    # --- Intrinsics (normalized) ---
    intrinsics = np.array([
        [args.fx / w, 0.0,         args.cx / w],
        [0.0,         args.fy / h, args.cy / h],
        [0.0,         0.0,         1.0],
    ], dtype=np.float32)
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=device).unsqueeze(0)

    # --- Load model ---
    print(f"Loading model: {args.model}")
    t0 = time.time()
    model = MDMModel.from_pretrained(args.model).to(device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # --- Inference ---
    print("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        output = model.infer(image_tensor, depth_in=depth_tensor, intrinsics=intrinsics_tensor)
    print(f"Inference done in {time.time() - t0:.2f}s")

    depth_pred = output['depth'].squeeze().cpu().numpy()
    points_pred = output['points'].squeeze().cpu().numpy()  # [H, W, 3]

    print(f"Refined depth range: {depth_pred[depth_pred > 0].min():.3f} - {depth_pred.max():.3f} m")

    # --- Save PLY directly from model output ---
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    valid = np.isfinite(points_pred).all(axis=-1) & (points_pred[..., 2] > 0)
    verts = points_pred[valid]
    colors = image_rgb[valid]

    ply_path = out_dir / "pointcloud_refined.ply"
    pc = trimesh.PointCloud(verts, colors)
    pc.export(ply_path)
    print(f"Saved {ply_path} ({len(verts):,} points)")

    # Also save refined depth for reference
    np.save(out_dir / "depth_refined.npy", depth_pred)
    print(f"Saved {out_dir / 'depth_refined.npy'}")


if __name__ == "__main__":
    main()
