"""Orbbec Gemini 336L camera wrapper using pyorbbecsdk."""

import os
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

try:
    from pyorbbecsdk import (
        Config,
        OBFormat,
        OBSensorType,
        Pipeline,
    )

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class OrbbecCamera:
    """Wrapper around Orbbec camera pipeline for color + depth streaming."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pipeline: Optional[object] = None
        self.config: Optional[object] = None
        self.running = False

        # Latest frames (numpy arrays)
        self.color_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None
        self.depth_raw: Optional[np.ndarray] = None

        # Depth scale (mm -> m conversion factor)
        self.depth_scale: float = 1.0

        # Camera intrinsics & extrinsics (populated after pipeline start)
        self.depth_intrinsic: Optional[tuple] = None  # (fx, fy, cx, cy, w, h)
        self.color_intrinsic: Optional[tuple] = None  # (fx, fy, cx, cy, w, h)
        self.extrinsic_rot: Optional[np.ndarray] = None  # 3x3 rotation depth->color
        self.extrinsic_trans: Optional[np.ndarray] = None  # 3x1 translation (mm)

        # Recording state
        self.recording = False
        self.color_writer: Optional[cv2.VideoWriter] = None
        self.depth_writer: Optional[cv2.VideoWriter] = None

    def start(self) -> None:
        """Initialize and start the camera pipeline."""
        if not HAS_SDK:
            raise RuntimeError(
                "pyorbbecsdk is not installed or not available on this platform."
            )

        self.pipeline = Pipeline()
        self.config = Config()

        # Enable depth stream
        try:
            depth_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = depth_profiles.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)
        except Exception:
            pass

        # Enable color stream
        try:
            color_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            color_profile = color_profiles.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)
        except Exception:
            pass

        self.pipeline.start(self.config)
        self.running = True
        self._read_calibration()

    def _read_calibration(self) -> None:
        """Read intrinsics and extrinsics from the running pipeline."""
        # Depth intrinsics from stream profile
        try:
            profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            prof = profiles.get_default_video_stream_profile()
            intr = prof.get_intrinsic()
            if intr.fx > 0:
                self.depth_intrinsic = (
                    intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height
                )
        except Exception:
            pass

        # Color intrinsics from stream profile
        try:
            profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            prof = profiles.get_default_video_stream_profile()
            intr = prof.get_intrinsic()
            if intr.fx > 0:
                self.color_intrinsic = (
                    intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height
                )
        except Exception:
            pass

        # Extrinsics (depth -> color transform)
        try:
            params = self.pipeline.get_camera_param()
            ext = params.transform
            rot = np.array(ext.rot).reshape(3, 3)
            trans = np.array(ext.transform).reshape(3)
            # Only use if non-zero
            if np.any(rot != 0):
                self.extrinsic_rot = rot
                self.extrinsic_trans = trans
        except Exception:
            pass

    def stop(self) -> None:
        """Stop the camera pipeline and release resources."""
        self.stop_recording()
        if self.pipeline and self.running:
            self.pipeline.stop()
            self.running = False
        self.pipeline = None
        self.config = None

    def get_frames(self) -> bool:
        """Capture one frameset and update internal color/depth images.

        Returns True if at least one frame was obtained.
        """
        if not self.running or not self.pipeline:
            return False

        try:
            frames = self.pipeline.wait_for_frames(100)
        except Exception:
            return False

        if frames is None:
            return False

        got_frame = False

        # --- Color frame ---
        color_frame = frames.get_color_frame()
        if color_frame is not None:
            w = color_frame.get_width()
            h = color_frame.get_height()
            data = np.asarray(color_frame.get_data())

            fmt = color_frame.get_format()
            if fmt == OBFormat.MJPG:
                self.color_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == OBFormat.RGB:
                self.color_image = cv2.cvtColor(
                    data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR
                )
            elif fmt == OBFormat.YUYV:
                self.color_image = cv2.cvtColor(
                    data.reshape((h, w, 2)), cv2.COLOR_YUV2BGR_YUYV
                )
            else:
                try:
                    self.color_image = data.reshape((h, w, 3)).copy()
                except ValueError:
                    self.color_image = data.reshape((h, w, -1))[:, :, :3].copy()
            got_frame = True

        # --- Depth frame ---
        depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            self.depth_scale = depth_frame.get_depth_scale()
            data = np.asarray(depth_frame.get_data())
            if data.dtype == np.uint8:
                data = data.view(np.uint16)
            self.depth_raw = data.reshape((h, w)).astype(np.uint16).copy()

            # Colormap visualization
            depth_vis = cv2.normalize(
                self.depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            self.depth_image = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            got_frame = True

        if self.recording:
            self._write_recording_frame()

        return got_frame

    # ---- Snapshot ----

    def save_image(self, prefix: str = "") -> str:
        """Save current color and depth images. Returns the timestamp string."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []

        if self.color_image is not None:
            path = os.path.join(self.output_dir, f"{prefix}color_{ts}.png")
            cv2.imwrite(path, self.color_image)
            saved.append(path)

        if self.depth_image is not None:
            path = os.path.join(self.output_dir, f"{prefix}depth_{ts}.png")
            cv2.imwrite(path, self.depth_image)
            saved.append(path)

        if self.depth_raw is not None:
            path = os.path.join(self.output_dir, f"{prefix}depth_raw_{ts}.npy")
            np.save(path, self.depth_raw)
            saved.append(path)

        return ts if saved else ""

    # ---- Recording ----

    def start_recording(self) -> str:
        """Start recording color and depth video. Returns timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        if self.color_image is not None:
            h, w = self.color_image.shape[:2]
            path = os.path.join(self.output_dir, f"color_{ts}.avi")
            self.color_writer = cv2.VideoWriter(path, fourcc, 30, (w, h))

        if self.depth_image is not None:
            h, w = self.depth_image.shape[:2]
            path = os.path.join(self.output_dir, f"depth_{ts}.avi")
            self.depth_writer = cv2.VideoWriter(path, fourcc, 30, (w, h))

        self.recording = True
        return ts

    def stop_recording(self) -> None:
        """Stop recording and release video writers."""
        self.recording = False
        if self.color_writer:
            self.color_writer.release()
            self.color_writer = None
        if self.depth_writer:
            self.depth_writer.release()
            self.depth_writer = None

    def _write_recording_frame(self) -> None:
        if self.color_writer and self.color_image is not None:
            self.color_writer.write(self.color_image)
        if self.depth_writer and self.depth_image is not None:
            self.depth_writer.write(self.depth_image)

    # ---- Point Cloud ----

    def save_ply(self) -> str:
        """Generate and save a colored PLY point cloud.

        Uses the raw (unaligned) depth with the depth camera's own intrinsics,
        then projects each 3D point into the color camera via extrinsics to
        obtain per-point color.

        Returns the saved file path, or empty string on failure.
        """
        if self.depth_raw is None:
            return ""

        import open3d as o3d

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"pointcloud_{ts}.ply")

        depth = self.depth_raw.astype(np.float64)
        dh, dw = depth.shape

        # Depth intrinsics
        if self.depth_intrinsic and self.depth_intrinsic[0] > 0:
            fx_d, fy_d, cx_d, cy_d = self.depth_intrinsic[:4]
        else:
            fx_d = fy_d = 0.6 * dw
            cx_d, cy_d = dw / 2.0, dh / 2.0

        # Valid mask: non-zero depth, within reasonable range
        valid = (depth > 0) & (depth < 10000)
        vs, us = np.where(valid)
        zs = depth[valid]  # mm

        # Back-project to 3D in depth camera frame (meters)
        xs = (us - cx_d) * zs / fx_d
        ys = (vs - cy_d) * zs / fy_d
        points_d = np.stack([xs, ys, zs], axis=-1)  # (N, 3) in mm

        # Convert to meters
        points_m = points_d / 1000.0

        # Build point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_m)

        # Project to color for per-point coloring
        if (
            self.color_image is not None
            and self.color_intrinsic is not None
            and self.extrinsic_rot is not None
        ):
            fx_c, fy_c, cx_c, cy_c = self.color_intrinsic[:4]
            ch, cw = self.color_image.shape[:2]

            # Transform depth -> color camera frame (in mm)
            points_c = (self.extrinsic_rot @ points_d.T).T + self.extrinsic_trans
            # Project to color image plane
            u_c = (points_c[:, 0] * fx_c / points_c[:, 2] + cx_c).astype(np.int32)
            v_c = (points_c[:, 1] * fy_c / points_c[:, 2] + cy_c).astype(np.int32)
            # Mask for valid projections
            color_valid = (u_c >= 0) & (u_c < cw) & (v_c >= 0) & (v_c < ch)

            color_rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
            colors = np.ones((len(points_m), 3), dtype=np.float64) * 0.5
            colors[color_valid] = (
                color_rgb[v_c[color_valid], u_c[color_valid]].astype(np.float64) / 255.0
            )
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        o3d.io.write_point_cloud(path, pcd)
        return path
