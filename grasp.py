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

# Default config path (franka panda)
_DEFAULT_CONFIG = _GRASPGEN_ROOT / "GraspGenModels" / "checkpoints" / "graspgen_franka_panda.yml"

AVAILABLE_GRIPPERS = {
    "franka_panda": "graspgen_franka_panda.yml",
    "robotiq_2f_140": "graspgen_robotiq_2f_140.yml",
    "single_suction_cup_30mm": "graspgen_single_suction_cup_30mm.yml",
}


class GraspModel:
    """Wraps GraspGenSampler for grasp pose generation."""

    def __init__(self, gripper_name: str = "franka_panda"):
        self.gripper_name = gripper_name
        self.sampler = None
        self._cfg = None

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

        self._cfg = load_grasp_cfg(str(config_path))
        self.sampler = GraspGenSampler(self._cfg)

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
