"""GraspGen wrapper — generate grasp poses from object point clouds."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Ensure third_party/GraspGen is importable
_GRASPGEN_ROOT = Path(__file__).resolve().parent / "third_party" / "GraspGen"
if str(_GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRASPGEN_ROOT))

# Default config path
_DEFAULT_CONFIG = _GRASPGEN_ROOT / "GraspGenModels" / "checkpoints" / "graspgen_g2.yml"

AVAILABLE_GRIPPERS = {
    "g2": "graspgen_g2.yml",
    "franka_panda": "graspgen_franka_panda.yml",
    "robotiq_2f_140": "graspgen_robotiq_2f_140.yml",
    "single_suction_cup_30mm": "graspgen_single_suction_cup_30mm.yml",
}


class GraspModel:
    """Wraps GraspGenSampler for grasp pose generation."""

    def __init__(self, gripper_name: str = "g2"):
        self.gripper_name = gripper_name
        self.sampler = None
        self._cfg = None
        self._collision_mesh = None

    @property
    def loaded(self) -> bool:
        return self.sampler is not None

    def load(self, gripper_name: Optional[str] = None) -> None:
        """Load GraspGen model for the specified gripper."""
        if gripper_name is not None:
            self.gripper_name = gripper_name
        if self.gripper_name not in AVAILABLE_GRIPPERS:
            raise ValueError(f"Unknown gripper: {self.gripper_name}. Choose from {list(AVAILABLE_GRIPPERS)}")

        config_path = _GRASPGEN_ROOT / "GraspGenModels" / "checkpoints" / AVAILABLE_GRIPPERS[self.gripper_name]
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info

        self._cfg = load_grasp_cfg(str(config_path))
        self.sampler = GraspGenSampler(self._cfg)

        # Load gripper collision mesh for scene collision filtering
        gripper_info = get_gripper_info(self.gripper_name)
        self._collision_mesh = gripper_info.collision_mesh

    def generate(
        self,
        point_cloud: np.ndarray,
        num_grasps: int = 200,
        topk: int = 20,
        threshold: float = 0.5,
    ) -> dict:
        """Generate grasp poses from an object point cloud (N, 3).

        Returns:
            dict with keys:
                poses: (K, 4, 4) np.ndarray — grasp poses as homogeneous transforms
                scores: (K,) np.ndarray — confidence scores
        """
        if not self.loaded:
            raise RuntimeError("Call load() first")

        from grasp_gen.grasp_server import GraspGenSampler

        grasps, confs = GraspGenSampler.run_inference(
            point_cloud,
            self.sampler,
            grasp_threshold=threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk,
        )

        if len(grasps) == 0:
            return {"poses": np.empty((0, 4, 4)), "scores": np.empty(0)}

        return {
            "poses": grasps.cpu().numpy(),
            "scores": confs.cpu().numpy(),
        }

    def filter_collisions(
        self,
        poses: np.ndarray,
        scores: np.ndarray,
        scene_pc: np.ndarray,
        obj_mask: np.ndarray,
        collision_threshold: float = 0.01,
        max_scene_points: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter grasps that collide with the scene (excluding the target object).

        Mirrors the logic in demo_scene_pc.py:
        1. Remove object points from scene point cloud
        2. Downsample scene for speed
        3. Call filter_colliding_grasps with gripper collision mesh

        Args:
            poses: (K, 4, 4) grasp poses
            scores: (K,) confidence scores
            scene_pc: (N, 3) full scene point cloud (flat)
            obj_mask: (N,) bool mask — True for object points
            collision_threshold: distance threshold in meters
            max_scene_points: max scene points for collision check

        Returns:
            filtered_poses: (M, 4, 4) collision-free grasp poses
            filtered_scores: (M,) corresponding scores
        """
        if self._collision_mesh is None:
            raise RuntimeError("Call load() first")
        if len(poses) == 0:
            return poses, scores

        from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

        # Remove object points — grasps near the object are expected
        scene_no_obj = scene_pc[~obj_mask]

        # Keep only finite points
        valid = np.isfinite(scene_no_obj).all(axis=-1) & (scene_no_obj[:, 2] > 0)
        scene_no_obj = scene_no_obj[valid]

        if len(scene_no_obj) == 0:
            print("[GraspGen] No scene points left after removing object — skipping collision filter")
            return poses, scores

        # Downsample for speed
        if len(scene_no_obj) > max_scene_points:
            idx = np.random.choice(len(scene_no_obj), max_scene_points, replace=False)
            scene_ds = scene_no_obj[idx]
            print(f"[GraspGen] Downsampled scene from {len(scene_no_obj)} to {len(scene_ds)} points")
        else:
            scene_ds = scene_no_obj
            print(f"[GraspGen] Scene has {len(scene_ds)} points (no downsampling needed)")

        # Ensure homogeneous coordinate
        poses_check = poses.copy()
        poses_check[:, 3, 3] = 1

        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_ds,
            grasp_poses=poses_check,
            gripper_collision_mesh=self._collision_mesh,
            collision_threshold=collision_threshold,
        )

        n_free = collision_free_mask.sum()
        print(f"[GraspGen] Collision filter: {n_free}/{len(poses)} grasps are collision-free")

        return poses[collision_free_mask], scores[collision_free_mask]


def extract_object_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    mask: np.ndarray,
    depth_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract object points using a 2D segmentation mask.

    Args:
        points: (H*W, 3) or (H, W, 3) full scene point cloud
        colors: (H*W, 3) or (H, W, 3) corresponding colors
        mask: (H, W) bool segmentation mask
        depth_shape: (H, W) shape of the depth/point map

    Returns:
        obj_points: (M, 3) filtered object points
        obj_colors: (M, 3) corresponding colors
    """
    h, w = depth_shape
    if points.ndim == 2:
        points = points.reshape(h, w, 3)
    if colors.ndim == 2:
        colors = colors.reshape(h, w, 3)

    # Resize mask to match depth resolution if needed
    if mask.shape != (h, w):
        import cv2
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    valid = mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    return points[valid], colors[valid]
