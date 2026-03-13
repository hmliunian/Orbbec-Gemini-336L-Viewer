"""Visualize one grasp: object (blue) + gripper open (green) + gripper close (red)."""

import sys
import numpy as np
import open3d as o3d

grasp_dir = "/data/DISCOVERSE_v2_exp/Orbbec/output/grasp_pcds"
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

obj_pts = np.load(f"{grasp_dir}/object.npy")
open_pts = np.load(f"{grasp_dir}/grasp_{idx:03d}_gripper_open.npy")
close_pts = np.load(f"{grasp_dir}/grasp_{idx:03d}_gripper_close.npy")

print(f"Grasp {idx}")
print(f"  object:        {obj_pts.shape}")
print(f"  gripper_open:  {open_pts.shape}")
print(f"  gripper_close: {close_pts.shape}")

# Object — blue
pcd_obj = o3d.geometry.PointCloud()
pcd_obj.points = o3d.utility.Vector3dVector(obj_pts)
pcd_obj.paint_uniform_color([0.3, 0.5, 1.0])

# Gripper open — green
pcd_open = o3d.geometry.PointCloud()
pcd_open.points = o3d.utility.Vector3dVector(open_pts)
pcd_open.paint_uniform_color([0.2, 0.9, 0.3])

# Gripper close — red
pcd_close = o3d.geometry.PointCloud()
pcd_close.points = o3d.utility.Vector3dVector(close_pts)
pcd_close.paint_uniform_color([1.0, 0.2, 0.2])

# Coordinate frame for reference
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

o3d.visualization.draw_geometries(
    [pcd_obj, pcd_open, pcd_close, frame],
    window_name=f"Grasp {idx}",
    width=1280, height=720,
    point_show_normal=False,
)
