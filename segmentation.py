"""SAM3 segmentation wrapper — text prompt detection + click-to-select."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Ensure third_party/sam3 is importable
_SAM3_ROOT = str(Path(__file__).resolve().parent / "third_party" / "sam3")
if _SAM3_ROOT not in sys.path:
    sys.path.insert(0, _SAM3_ROOT)

_SAM3_CHECKPOINT = str(Path(__file__).resolve().parent / "third_party" / "sam3" / "checkpoints" / "sam3.pt")


class SegmentationModel:
    """Wraps SAM3 image model for text-based detection and click selection."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._state: Optional[dict] = None
        # cached results after detection
        self._masks: Optional[torch.Tensor] = None  # (N, 1, H, W) bool
        self._scores: Optional[torch.Tensor] = None  # (N,)
        self._boxes: Optional[torch.Tensor] = None  # (N, 4)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load SAM3 image model (heavy — call once)."""
        if self.loaded:
            return
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(device=self.device, checkpoint_path=_SAM3_CHECKPOINT)
        self.processor = Sam3Processor(self.model, device=self.device, confidence_threshold=0.3)

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Set the working image (H, W, 3) uint8 RGB."""
        pil_img = Image.fromarray(image_rgb)
        self._state = self.processor.set_image(pil_img)
        self._masks = None
        self._scores = None
        self._boxes = None

    def detect_all(self, text_prompt: str = "object") -> dict:
        """Run text-prompted detection. Returns masks/boxes/scores."""
        if self._state is None:
            raise RuntimeError("Call set_image first")
        self.processor.reset_all_prompts(self._state)
        output = self.processor.set_text_prompt(prompt=text_prompt, state=self._state)
        self._masks = output["masks"]  # (N, 1, H, W) bool
        self._scores = output["scores"]  # (N,)
        self._boxes = output["boxes"]  # (N, 4) xyxy
        return {
            "masks": self._masks,
            "scores": self._scores,
            "boxes": self._boxes,
            "count": len(self._scores),
        }

    def select_by_click(self, x: int, y: int) -> Optional[np.ndarray]:
        """Given a pixel coordinate, return the mask (H, W) bool that contains it.

        If multiple masks contain the point, pick the one with highest score.
        Returns None if no mask contains the point.
        """
        if self._masks is None or self._masks.numel() == 0:
            return None

        masks_hw = self._masks.squeeze(1)  # (N, H, W)
        n = masks_hw.shape[0]
        best_idx = -1
        best_score = -1.0

        for i in range(n):
            if masks_hw[i, y, x].item():
                score = self._scores[i].item()
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_idx < 0:
            return None

        return masks_hw[best_idx].cpu().numpy()

    def get_colored_overlay(self, image_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Overlay all detected masks on the image with distinct colors."""
        if self._masks is None or self._masks.numel() == 0:
            return image_rgb.copy()

        overlay = image_rgb.astype(np.float32).copy()
        masks_hw = self._masks.squeeze(1).cpu().numpy()  # (N, H, W)

        palette = np.array([
            [66, 133, 244],   # blue
            [234, 67, 53],    # red
            [251, 188, 4],    # yellow
            [52, 168, 83],    # green
            [171, 71, 188],   # purple
            [255, 112, 67],   # orange
        ], dtype=np.float32)

        for i in range(masks_hw.shape[0]):
            color = palette[i % len(palette)]
            mask = masks_hw[i].astype(bool)
            overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

        return overlay.astype(np.uint8)

    def get_selected_overlay(
        self, image_rgb: np.ndarray, selected_mask: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay a single selected mask with a highlight color."""
        overlay = image_rgb.astype(np.float32).copy()
        color = np.array([0, 200, 255], dtype=np.float32)  # cyan
        mask = selected_mask.astype(bool)
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
        return overlay.astype(np.uint8)
