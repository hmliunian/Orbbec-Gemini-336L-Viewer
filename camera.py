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

        # Depth refinement model (lazy-loaded)
        self._refine_model = None
        self._refine_device = None

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

    def _get_refine_model(self):
        """Lazy-load the lingbot-depth model on first use."""
        if self._refine_model is None:
            import torch
            from mdm.model.v2 import MDMModel

            self._refine_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._refine_model = MDMModel.from_pretrained(
                "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
            ).to(self._refine_device)
        return self._refine_model, self._refine_device

    def save_ply(self) -> str:
        """Generate and save a colored PLY point cloud using lingbot-depth.

        Feeds RGB + raw depth + intrinsics to lingbot-depth, then saves
        the model's point cloud output directly as PLY.

        Returns the saved file path, or empty string on failure.
        """
        if self.depth_raw is None or self.color_image is None:
            return ""

        import torch
        import trimesh

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"pointcloud_{ts}.ply")

        model, device = self._get_refine_model()

        # Prepare RGB: BGR -> RGB, normalize to [0,1], shape [1,3,H,W]
        image_rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        image_tensor = torch.tensor(
            image_rgb / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1).unsqueeze(0)

        # Prepare depth: uint16 mm -> float32 meters, resize to match RGB
        depth_m = self.depth_raw.astype(np.float32) / 1000.0
        dh, dw = depth_m.shape
        if (dh, dw) != (h, w):
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_tensor = torch.tensor(depth_m, dtype=torch.float32, device=device)

        # Prepare normalized intrinsics
        if self.depth_intrinsic and self.depth_intrinsic[0] > 0:
            fx, fy, cx, cy = self.depth_intrinsic[:4]
        else:
            fx = fy = 0.6 * w
            cx, cy = w / 2.0, h / 2.0

        intrinsics = np.array([
            [fx / w, 0.0,    cx / w],
            [0.0,    fy / h, cy / h],
            [0.0,    0.0,    1.0],
        ], dtype=np.float32)
        intrinsics_tensor = torch.tensor(
            intrinsics, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model.infer(
                image_tensor, depth_in=depth_tensor, intrinsics=intrinsics_tensor
            )

        points_pred = output['points'].squeeze().cpu().numpy()  # [H, W, 3]

        # Save PLY directly from model output
        valid = np.isfinite(points_pred).all(axis=-1) & (points_pred[..., 2] > 0)
        verts = points_pred[valid]
        colors = image_rgb[valid]

        pc = trimesh.PointCloud(verts, colors)
        pc.export(path)
        return path
