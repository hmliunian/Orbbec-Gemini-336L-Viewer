"""Test point cloud export."""
import time, traceback
from camera import OrbbecCamera

try:
    cam = OrbbecCamera(output_dir='output')
    cam.start()
    time.sleep(1)
    for i in range(10):
        ok = cam.get_frames()
        if ok and cam.depth_raw is not None:
            break
        time.sleep(0.2)

    print(f'depth_raw: dtype={cam.depth_raw.dtype}, shape={cam.depth_raw.shape}, min={cam.depth_raw.min()}, max={cam.depth_raw.max()}')
    print(f'depth_scale: {cam.depth_scale}')
    print(f'color_image: shape={cam.color_image.shape if cam.color_image is not None else None}')

    path = cam.save_ply()
    print(f'PLY saved: {path}')

    if path:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        print(f'Point count: {len(pcd.points)}')
        print(f'Has colors: {pcd.has_colors()}')
        pts = pcd.points
        import numpy as np
        arr = np.asarray(pts)
        print(f'Points range: x=[{arr[:,0].min():.3f}, {arr[:,0].max():.3f}], y=[{arr[:,1].min():.3f}, {arr[:,1].max():.3f}], z=[{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]')

    cam.stop()
except Exception:
    traceback.print_exc()
