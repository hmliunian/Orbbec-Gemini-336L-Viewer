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

        # Enable depth stream (default: 848x480 Y16 @ 30fps)
        try:
            depth_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = depth_profiles.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)
        except Exception:
            pass

        # Enable color stream (default: 1280x720 MJPG @ 30fps)
        try:
            color_profiles = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            color_profile = color_profiles.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)
        except Exception:
            pass  # Some devices may not have color sensor

        self.pipeline.start(self.config)
        self.running = True

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
                # Assume BGR
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
            # Y16 format: raw bytes need to be viewed as uint16 first
            if data.dtype == np.uint8:
                data = data.view(np.uint16)
            self.depth_raw = data.reshape((h, w)).astype(np.uint16).copy()

            # Colormap visualization
            depth_vis = cv2.normalize(
                self.depth_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            self.depth_image = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            got_frame = True

        # Write to video if recording
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
        """Generate and save a PLY point cloud from current depth (+ color).

        Returns the saved file path, or empty string on failure.
        """
        if self.depth_raw is None:
            return ""

        import open3d as o3d

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"pointcloud_{ts}.ply")

        h, w = self.depth_raw.shape

        # Try to read intrinsics from the pipeline; fall back to approximate values
        try:
            params = self.pipeline.get_camera_param()
            dp = params.depth_intrinsic
            fx, fy, cx, cy = dp.fx, dp.fy, dp.cx, dp.cy
        except Exception:
            fx = fy = 0.6 * w  # rough estimate
            cx, cy = w / 2.0, h / 2.0

        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # depth_scale: raw depth values are in mm, Open3D needs the divisor to get meters
        depth_o3d = o3d.geometry.Image(self.depth_raw.astype(np.uint16))

        if self.color_image is not None:
            color_resized = cv2.resize(self.color_image, (w, h))
            color_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)
            color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=1000.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsic,
                depth_scale=1000.0,
                depth_trunc=5.0,
            )

        o3d.io.write_point_cloud(path, pcd)
        return path
