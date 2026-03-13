"""Orbbec RGB-D Pipeline — Camera → Segment → Grasp Generation GUI."""

import os
import threading
import time
from enum import IntEnum
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import customtkinter as ctk

from camera import OrbbecCamera

# ---------- Theme ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CANVAS_W, CANVAS_H = 640, 480
FPS_INTERVAL = 0.5


# ---------- Pipeline steps ----------
class Step(IntEnum):
    CONNECT = 0
    CAPTURE = 1
    SEGMENT = 2
    GRASP = 3


STEP_LABELS = ["Connect", "Capture", "Segment", "Generate Grasps"]
STEP_ICONS = ["\u25cb", "\u25cb", "\u25cb", "\u25cb"]  # ○
STEP_DONE = "\u25cf"  # ●


# ---------- Colors ----------
C_BG = "#1a1a2e"
C_SIDEBAR = "#16213e"
C_CARD = "#0f3460"
C_ACCENT = "#00adb5"
C_ACCENT_DIM = "#007a80"
C_TEXT = "#e2e2e2"
C_TEXT_DIM = "#8a8a9a"
C_SUCCESS = "#00c897"
C_WARNING = "#ff6b35"
C_DANGER = "#e94560"
C_STEP_ACTIVE = "#00adb5"
C_STEP_DONE = "#00c897"
C_STEP_PENDING = "#3a3a5c"


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Orbbec Grasp Pipeline")
        self.geometry("1440x820")
        self.minsize(1200, 700)
        self.configure(fg_color=C_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- State ---
        self.camera = OrbbecCamera(output_dir="output")
        self.connected = False
        self.streaming = False
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0

        # Pipeline state
        self.current_step = Step.CONNECT
        self._frozen_color: Optional[np.ndarray] = None  # RGB uint8
        self._frozen_depth_raw: Optional[np.ndarray] = None
        self._frozen_depth_vis: Optional[np.ndarray] = None

        # Segmentation state
        self._seg_model = None
        self._detected = False
        self._click_points: list[tuple[int, int, int]] = []  # (x, y, label)
        self._selected_mask: Optional[np.ndarray] = None
        self._text_prompt = "object"

        # Grasp state
        self._grasp_model = None
        self._grasp_results: Optional[dict] = None

        self._build_ui()

    # ================================================================ UI BUILD
    def _build_ui(self) -> None:
        # Main grid: sidebar | content
        self.grid_columnconfigure(0, weight=0, minsize=220)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content()

    # ---- Sidebar ----
    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, fg_color=C_SIDEBAR, corner_radius=0, width=220)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # Logo / title area
        title_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=16, pady=(24, 8))
        ctk.CTkLabel(
            title_frame, text="GRASP", font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_ACCENT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_frame, text="PIPELINE", font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_frame, text="Orbbec RGB-D", font=ctk.CTkFont(size=11),
            text_color=C_TEXT_DIM,
        ).pack(anchor="w", pady=(2, 0))

        # Divider
        ctk.CTkFrame(sidebar, fg_color=C_ACCENT_DIM, height=1).pack(fill="x", padx=16, pady=16)

        # Step indicators
        self.step_frames: list[ctk.CTkFrame] = []
        self.step_icons: list[ctk.CTkLabel] = []
        self.step_labels: list[ctk.CTkLabel] = []
        self.step_lines: list[ctk.CTkFrame] = []

        for i, label in enumerate(STEP_LABELS):
            row = ctk.CTkFrame(sidebar, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)

            icon_lbl = ctk.CTkLabel(
                row, text=STEP_ICONS[i], width=28,
                font=ctk.CTkFont(size=16), text_color=C_STEP_PENDING,
            )
            icon_lbl.pack(side="left", padx=(0, 8))

            txt_lbl = ctk.CTkLabel(
                row, text=label, font=ctk.CTkFont(size=13),
                text_color=C_TEXT_DIM, anchor="w",
            )
            txt_lbl.pack(side="left", fill="x", expand=True)

            self.step_frames.append(row)
            self.step_icons.append(icon_lbl)
            self.step_labels.append(txt_lbl)

            # Connector line between steps
            if i < len(STEP_LABELS) - 1:
                line = ctk.CTkFrame(sidebar, fg_color=C_STEP_PENDING, height=16, width=2)
                line.pack(padx=(30, 0), anchor="w")
                self.step_lines.append(line)

        # Divider
        ctk.CTkFrame(sidebar, fg_color=C_ACCENT_DIM, height=1).pack(fill="x", padx=16, pady=16)

        # --- Parameters panel ---
        param_label = ctk.CTkLabel(
            sidebar, text="Parameters", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        )
        param_label.pack(anchor="w", padx=16)

        # Text prompt
        ctk.CTkLabel(sidebar, text="Text prompt", font=ctk.CTkFont(size=11),
                      text_color=C_TEXT_DIM).pack(anchor="w", padx=16, pady=(8, 2))
        self.entry_prompt = ctk.CTkEntry(sidebar, height=28, fg_color=C_CARD, border_width=0)
        self.entry_prompt.insert(0, "object")
        self.entry_prompt.pack(fill="x", padx=16)

        # Gripper selector
        ctk.CTkLabel(sidebar, text="Gripper", font=ctk.CTkFont(size=11),
                      text_color=C_TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 2))
        self.combo_gripper = ctk.CTkComboBox(
            sidebar, values=["g2", "franka_panda", "robotiq_2f_140", "single_suction_cup_30mm"],
            height=28, fg_color=C_CARD, border_width=0, dropdown_fg_color=C_CARD,
        )
        self.combo_gripper.set("robotiq_2f_140")
        self.combo_gripper.pack(fill="x", padx=16)

        # Num grasps
        ctk.CTkLabel(sidebar, text="Num grasps", font=ctk.CTkFont(size=11),
                      text_color=C_TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 2))
        self.slider_grasps = ctk.CTkSlider(
            sidebar, from_=50, to=500, number_of_steps=9,
            fg_color=C_CARD, progress_color=C_ACCENT, button_color=C_ACCENT,
        )
        self.slider_grasps.set(200)
        self.slider_grasps.pack(fill="x", padx=16)
        self.lbl_grasps_val = ctk.CTkLabel(sidebar, text="200", font=ctk.CTkFont(size=11),
                                            text_color=C_TEXT_DIM)
        self.lbl_grasps_val.pack(anchor="w", padx=16)
        self.slider_grasps.configure(command=lambda v: self.lbl_grasps_val.configure(text=str(int(v))))

        # Confidence threshold
        ctk.CTkLabel(sidebar, text="Confidence threshold", font=ctk.CTkFont(size=11),
                      text_color=C_TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 2))
        self.slider_conf = ctk.CTkSlider(
            sidebar, from_=0.0, to=1.0, number_of_steps=20,
            fg_color=C_CARD, progress_color=C_ACCENT, button_color=C_ACCENT,
        )
        self.slider_conf.set(0.5)
        self.slider_conf.pack(fill="x", padx=16)
        self.lbl_conf_val = ctk.CTkLabel(sidebar, text="0.50", font=ctk.CTkFont(size=11),
                                          text_color=C_TEXT_DIM)
        self.lbl_conf_val.pack(anchor="w", padx=16)
        self.slider_conf.configure(command=lambda v: self.lbl_conf_val.configure(text=f"{v:.2f}"))

        # Spacer
        ctk.CTkFrame(sidebar, fg_color="transparent").pack(fill="both", expand=True)

        # FPS display at bottom
        self.lbl_fps = ctk.CTkLabel(
            sidebar, text="FPS: --", font=ctk.CTkFont(size=11), text_color=C_TEXT_DIM,
        )
        self.lbl_fps.pack(anchor="w", padx=16, pady=(0, 16))

    # ---- Content area ----
    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=0)
        content.grid_rowconfigure(0, weight=0)
        content.grid_rowconfigure(1, weight=1)
        content.grid_rowconfigure(2, weight=0)
        content.grid_columnconfigure(0, weight=1)

        # --- Top action bar ---
        topbar = ctk.CTkFrame(content, fg_color=C_SIDEBAR, height=52, corner_radius=0)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)

        self.btn_connect = ctk.CTkButton(
            topbar, text="Connect", width=110, height=32, corner_radius=6,
            fg_color=C_ACCENT, hover_color=C_ACCENT_DIM, command=self._toggle_connect,
        )
        self.btn_connect.pack(side="left", padx=(16, 8), pady=10)

        self.btn_capture = ctk.CTkButton(
            topbar, text="Capture", width=110, height=32, corner_radius=6,
            fg_color=C_CARD, hover_color=C_ACCENT_DIM, command=self._capture_frame,
            state="disabled",
        )
        self.btn_capture.pack(side="left", padx=4, pady=10)

        self.btn_detect = ctk.CTkButton(
            topbar, text="Detect Objects", width=130, height=32, corner_radius=6,
            fg_color=C_CARD, hover_color=C_ACCENT_DIM, command=self._run_detection,
            state="disabled",
        )
        self.btn_detect.pack(side="left", padx=4, pady=10)

        self.btn_grasp = ctk.CTkButton(
            topbar, text="Generate Grasps", width=140, height=32, corner_radius=6,
            fg_color=C_CARD, hover_color=C_ACCENT_DIM, command=self._run_grasp_gen,
            state="disabled",
        )
        self.btn_grasp.pack(side="left", padx=4, pady=10)

        # Right side: snapshot + record + export
        self.btn_ply = ctk.CTkButton(
            topbar, text="Export PLY", width=100, height=32, corner_radius=6,
            fg_color=C_CARD, hover_color=C_ACCENT_DIM, command=self._export_ply,
            state="disabled",
        )
        self.btn_ply.pack(side="right", padx=(4, 16), pady=10)

        self.btn_record = ctk.CTkButton(
            topbar, text="Record", width=90, height=32, corner_radius=6,
            fg_color=C_CARD, hover_color=C_ACCENT_DIM, command=self._toggle_record,
            state="disabled",
        )
        self.btn_record.pack(side="right", padx=4, pady=10)

        self.btn_snap = ctk.CTkButton(
            topbar, text="Snapshot", width=90, height=32, corner_radius=6,
            fg_color=C_CARD, hover_color=C_ACCENT_DIM, command=self._snapshot,
            state="disabled",
        )
        self.btn_snap.pack(side="right", padx=4, pady=10)

        self.btn_reset = ctk.CTkButton(
            topbar, text="Reset", width=70, height=32, corner_radius=6,
            fg_color=C_DANGER, hover_color="#c0392b", command=self._reset_pipeline,
            state="disabled",
        )
        self.btn_reset.pack(side="right", padx=4, pady=10)

        # --- Main viewport ---
        viewport = ctk.CTkFrame(content, fg_color=C_BG)
        viewport.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        viewport.grid_columnconfigure(0, weight=1)
        viewport.grid_columnconfigure(1, weight=1)
        viewport.grid_rowconfigure(0, weight=0)
        viewport.grid_rowconfigure(1, weight=1)

        # Headers
        ctk.CTkLabel(
            viewport, text="COLOR", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_ACCENT,
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(4, 0))
        ctk.CTkLabel(
            viewport, text="DEPTH", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C_ACCENT,
        ).grid(row=0, column=1, sticky="w", padx=8, pady=(4, 0))

        # Image cards
        left_card = ctk.CTkFrame(viewport, fg_color=C_CARD, corner_radius=8)
        left_card.grid(row=1, column=0, sticky="nsew", padx=(4, 4), pady=4)
        self.color_label = ctk.CTkLabel(left_card, text="No camera connected", text_color=C_TEXT_DIM)
        self.color_label.pack(fill="both", expand=True, padx=4, pady=4)
        # Bind click on color image (must bind to all internal widgets of CTkLabel)
        self._bind_click_recursive(self.color_label)

        right_card = ctk.CTkFrame(viewport, fg_color=C_CARD, corner_radius=8)
        right_card.grid(row=1, column=1, sticky="nsew", padx=(4, 4), pady=4)
        self.depth_label = ctk.CTkLabel(right_card, text="No camera connected", text_color=C_TEXT_DIM)
        self.depth_label.pack(fill="both", expand=True, padx=4, pady=4)

        # --- Status bar ---
        statusbar = ctk.CTkFrame(content, fg_color=C_SIDEBAR, height=32, corner_radius=0)
        statusbar.grid(row=2, column=0, sticky="ew")
        statusbar.grid_propagate(False)

        self.status_var = ctk.StringVar(value="Ready — press Connect to start")
        self.lbl_status = ctk.CTkLabel(
            statusbar, textvariable=self.status_var, anchor="w",
            font=ctk.CTkFont(size=11), text_color=C_TEXT_DIM,
        )
        self.lbl_status.pack(side="left", padx=16, fill="x", expand=True)

        self._update_steps()

    # ================================================================ STEP TRACKER
    def _update_steps(self) -> None:
        for i in range(len(STEP_LABELS)):
            if i < self.current_step:
                self.step_icons[i].configure(text=STEP_DONE, text_color=C_STEP_DONE)
                self.step_labels[i].configure(text_color=C_STEP_DONE)
            elif i == self.current_step:
                self.step_icons[i].configure(text="\u25c9", text_color=C_STEP_ACTIVE)  # ◉
                self.step_labels[i].configure(text_color=C_TEXT, font=ctk.CTkFont(size=13, weight="bold"))
            else:
                self.step_icons[i].configure(text="\u25cb", text_color=C_STEP_PENDING)
                self.step_labels[i].configure(text_color=C_TEXT_DIM, font=ctk.CTkFont(size=13))

        for i, line in enumerate(self.step_lines):
            line.configure(fg_color=C_STEP_DONE if i < self.current_step else C_STEP_PENDING)

    def _advance_step(self, step: Step) -> None:
        self.current_step = step
        self._update_steps()

    # ================================================================ CONNECTION
    def _toggle_connect(self) -> None:
        if not self.connected:
            self._connect()
        else:
            self._disconnect()

    def _connect(self) -> None:
        self.status_var.set("Connecting to camera...")
        self.btn_connect.configure(state="disabled")
        self.update_idletasks()

        def _do():
            try:
                self.camera.start()
                self.connected = True
                self.streaming = True
                self._stop_event.clear()
                self.after(0, self._on_connected)
                self._stream_loop()
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: self._on_connect_error(msg))

        threading.Thread(target=_do, daemon=True).start()

    def _on_connected(self) -> None:
        self.btn_connect.configure(text="Disconnect", state="normal", fg_color=C_DANGER)
        self.btn_capture.configure(state="normal")
        self.btn_snap.configure(state="normal")
        self.btn_record.configure(state="normal")
        self.btn_ply.configure(state="normal")
        self.btn_reset.configure(state="normal")
        self._advance_step(Step.CAPTURE)
        self.status_var.set("Connected — streaming live. Press Capture to freeze a frame.")

    def _on_connect_error(self, msg: str) -> None:
        self.btn_connect.configure(state="normal")
        self.status_var.set(f"Connection failed: {msg}")

    def _disconnect(self) -> None:
        self._stop_event.set()
        self.streaming = False
        self.camera.stop()
        self.connected = False
        self.btn_connect.configure(text="Connect", fg_color=C_ACCENT)
        self.btn_capture.configure(state="disabled")
        self.btn_detect.configure(state="disabled")
        self.btn_grasp.configure(state="disabled")
        self.btn_snap.configure(state="disabled")
        self.btn_record.configure(state="disabled")
        self.btn_ply.configure(state="disabled")
        self.btn_reset.configure(state="disabled")
        self._advance_step(Step.CONNECT)
        self.status_var.set("Disconnected.")

    # ================================================================ STREAMING
    def _stream_loop(self) -> None:
        while not self._stop_event.is_set():
            ok = self.camera.get_frames()
            if ok:
                self._frame_count += 1
                if self.streaming:
                    self.after(0, self._update_display)
            else:
                time.sleep(0.005)

    def _update_display(self) -> None:
        now = time.time()
        dt = now - self._fps_time
        if dt >= FPS_INTERVAL:
            self._fps = self._frame_count / dt
            self._frame_count = 0
            self._fps_time = now
            self.lbl_fps.configure(text=f"FPS: {self._fps:.1f}")

        if self.camera.color_image is not None:
            self._show_image(self.color_label, self.camera.color_image)
        if self.camera.depth_image is not None:
            self._show_image(self.depth_label, self.camera.depth_image)

    def _show_image(self, label: ctk.CTkLabel, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lw = label.winfo_width()
        lh = label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = CANVAS_W, CANVAS_H
        h, w = rgb.shape[:2]
        scale = min(lw / w, lh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 1 or new_h < 1:
            return
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=img)
        label.configure(image=photo, text="")
        label._photo = photo

    def _show_rgb_on_label(self, label: ctk.CTkLabel, rgb: np.ndarray) -> None:
        """Display an RGB numpy array on a label, fitted to label size."""
        lw = label.winfo_width()
        lh = label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = CANVAS_W, CANVAS_H
        h, w = rgb.shape[:2]
        scale = min(lw / w, lh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 1 or new_h < 1:
            return
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=img)
        label.configure(image=photo, text="")
        label._photo = photo
        # Store scale info for click coordinate mapping
        label._img_scale = scale
        label._img_offset_x = (lw - new_w) // 2
        label._img_offset_y = (lh - new_h) // 2
        label._img_w = new_w
        label._img_h = new_h
        label._orig_w = w
        label._orig_h = h

    # ================================================================ CAPTURE
    def _capture_frame(self) -> None:
        if self.camera.color_image is None:
            self.status_var.set("No frame available to capture.")
            return

        # Freeze current frame
        self._frozen_color = cv2.cvtColor(self.camera.color_image.copy(), cv2.COLOR_BGR2RGB)
        self._frozen_depth_raw = self.camera.depth_raw.copy() if self.camera.depth_raw is not None else None
        self._frozen_depth_vis = self.camera.depth_image.copy() if self.camera.depth_image is not None else None

        # Stop live display, show frozen frame
        self.streaming = False
        self._show_rgb_on_label(self.color_label, self._frozen_color)
        if self._frozen_depth_vis is not None:
            self._show_image(self.depth_label, self._frozen_depth_vis)

        # Reset segmentation state
        self._detected = False
        self._click_points.clear()
        self._selected_mask = None
        self._grasp_results = None

        self.btn_detect.configure(state="normal")
        self.btn_grasp.configure(state="disabled")
        self._advance_step(Step.SEGMENT)
        self.status_var.set("Frame captured. Press Detect Objects or adjust text prompt, then click on an object.")

    # ================================================================ SEGMENTATION
    def _run_detection(self) -> None:
        if self._frozen_color is None:
            return

        self._text_prompt = self.entry_prompt.get().strip() or "object"
        self.btn_detect.configure(state="disabled")
        self.status_var.set(f"Loading SAM3 & detecting \"{self._text_prompt}\"...")
        self.update_idletasks()

        def _do():
            try:
                from segmentation import SegmentationModel

                if self._seg_model is None:
                    self._seg_model = SegmentationModel()
                if not self._seg_model.loaded:
                    self._seg_model.load()

                self._seg_model.set_image(self._frozen_color)
                result = self._seg_model.detect_all(self._text_prompt)
                count = result["count"]
                self.after(0, lambda: self._on_detection_done(count))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: self._on_detection_error(msg))

        threading.Thread(target=_do, daemon=True).start()

    def _on_detection_done(self, count: int) -> None:
        self._detected = True
        self.btn_detect.configure(state="normal")
        if count == 0:
            self.status_var.set("No objects detected. Try a different text prompt.")
            return

        # Show overlay of all masks
        overlay = self._seg_model.get_colored_overlay(self._frozen_color)
        self._show_rgb_on_label(self.color_label, overlay)
        self.status_var.set(
            f"Detected {count} object(s). Left-click to select, right-click for negative point."
        )

    def _on_detection_error(self, msg: str) -> None:
        self.btn_detect.configure(state="normal")
        self.status_var.set(f"Detection failed: {msg}")

    # ---- Click handling ----
    def _bind_click_recursive(self, widget) -> None:
        """Bind click events to widget and all its children (needed for CTkLabel)."""
        widget.bind("<Button-1>", self._on_click_left)
        widget.bind("<Button-3>", self._on_click_right)
        for child in widget.winfo_children():
            self._bind_click_recursive(child)

    def _pixel_from_event(self, event) -> Optional[tuple[int, int]]:
        """Convert label click event to original image pixel coordinates."""
        label = self.color_label
        if not hasattr(label, "_img_scale"):
            return None

        # Use root (screen) coordinates to avoid child-widget offset issues
        lx = event.x_root - label.winfo_rootx()
        ly = event.y_root - label.winfo_rooty()

        # The image is centered in the label; compute offset
        lw = label.winfo_width()
        lh = label.winfo_height()
        img_w = label._img_w
        img_h = label._img_h
        off_x = (lw - img_w) // 2
        off_y = (lh - img_h) // 2

        ix = lx - off_x
        iy = ly - off_y
        if ix < 0 or iy < 0 or ix >= img_w or iy >= img_h:
            return None

        orig_x = int(ix / label._img_scale)
        orig_y = int(iy / label._img_scale)
        orig_x = min(orig_x, label._orig_w - 1)
        orig_y = min(orig_y, label._orig_h - 1)
        return orig_x, orig_y

    def _on_click_left(self, event) -> None:
        if not self._detected or self._frozen_color is None:
            return
        pt = self._pixel_from_event(event)
        if pt is None:
            return
        x, y = pt
        self._click_points.append((x, y, 1))
        self._handle_click_selection(x, y)

    def _on_click_right(self, event) -> None:
        if not self._detected or self._frozen_color is None:
            return
        pt = self._pixel_from_event(event)
        if pt is None:
            return
        x, y = pt
        self._click_points.append((x, y, 0))
        self._redraw_clicks()

    def _handle_click_selection(self, x: int, y: int) -> None:
        mask = self._seg_model.select_by_click(x, y)
        if mask is None:
            self.status_var.set(f"No object at ({x}, {y}). Try clicking on a detected region.")
            self._redraw_clicks()
            return

        self._selected_mask = mask
        overlay = self._seg_model.get_selected_overlay(self._frozen_color, mask)
        self._draw_click_markers(overlay)
        self._show_rgb_on_label(self.color_label, overlay)
        self.btn_grasp.configure(state="normal")
        self._advance_step(Step.GRASP)
        self.status_var.set("Object selected. Press Generate Grasps to compute grasp poses.")

    def _redraw_clicks(self) -> None:
        if self._selected_mask is not None:
            overlay = self._seg_model.get_selected_overlay(self._frozen_color, self._selected_mask)
        elif self._detected:
            overlay = self._seg_model.get_colored_overlay(self._frozen_color)
        else:
            overlay = self._frozen_color.copy()
        self._draw_click_markers(overlay)
        self._show_rgb_on_label(self.color_label, overlay)

    def _draw_click_markers(self, image: np.ndarray) -> None:
        for x, y, label in self._click_points:
            color = (0, 255, 120) if label == 1 else (255, 60, 60)
            cv2.circle(image, (x, y), 6, color, -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 1)

    # ================================================================ GRASP GENERATION
    def _run_grasp_gen(self) -> None:
        if self._selected_mask is None or self._frozen_color is None:
            return

        self.btn_grasp.configure(state="disabled")
        gripper = self.combo_gripper.get()
        num_grasps = int(self.slider_grasps.get())
        threshold = self.slider_conf.get()
        self.status_var.set(f"Loading GraspGen ({gripper}) & generating {num_grasps} grasps...")
        self.update_idletasks()

        def _do():
            try:
                import torch
                from grasp import GraspModel, extract_object_pointcloud

                # Generate point cloud using camera's refine model (lingbot-depth)
                points, colors = self._build_pointcloud()

                # Release lingbot-depth to free VRAM before loading GraspGen
                self.camera._refine_model = None
                self.camera._refine_device = None

                # Release SAM3 to free VRAM
                if self._seg_model is not None:
                    self._seg_model.model = None
                    self._seg_model.processor = None
                    self._seg_model._state = None

                torch.cuda.empty_cache()

                # Now load GraspGen with freed VRAM
                if self._grasp_model is None:
                    self._grasp_model = GraspModel(gripper_name=gripper)
                    self._grasp_model.load()
                elif self._grasp_model.gripper_name != gripper:
                    self._grasp_model.load(gripper_name=gripper)
                if points is None:
                    self.after(0, lambda: self._on_grasp_error("Failed to build point cloud"))
                    return

                # Use point cloud resolution (matches color image, not raw depth)
                h, w = points.shape[:2]
                obj_pts, obj_colors = extract_object_pointcloud(
                    points, colors, self._selected_mask, (h, w),
                )

                if len(obj_pts) < 100:
                    self.after(0, lambda: self._on_grasp_error(
                        f"Too few points ({len(obj_pts)}). Try selecting a larger object."
                    ))
                    return

                # Downsample for GraspGen inference only, keep full cloud for visualization
                max_pts = 4096
                grasp_pts = obj_pts
                if len(obj_pts) > max_pts:
                    idx = np.random.choice(len(obj_pts), max_pts, replace=False)
                    grasp_pts = obj_pts[idx]
                print(f"[GraspGen] Object points: {len(obj_pts)}, grasp input: {len(grasp_pts)}")

                # Build scene collision data once
                h, w = points.shape[:2]
                scene_flat = points.reshape(-1, 3)
                mask_resized = self._selected_mask
                if mask_resized.shape != (h, w):
                    mask_resized = cv2.resize(
                        mask_resized.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                obj_mask_flat = mask_resized.reshape(-1)

                # Loop generate + filter until we have 50 collision-free grasps
                TARGET_GRASPS = 50
                MAX_ROUNDS = 15
                all_poses = []
                all_scores = []

                for round_i in range(MAX_ROUNDS):
                    result = self._grasp_model.generate(
                        grasp_pts, num_grasps=num_grasps, topk=num_grasps, threshold=threshold,
                    )
                    if len(result["poses"]) > 0:
                        filtered_poses, filtered_scores = self._grasp_model.filter_collisions(
                            poses=result["poses"],
                            scores=result["scores"],
                            scene_pc=scene_flat,
                            obj_mask=obj_mask_flat,
                        )
                        if len(filtered_poses) > 0:
                            all_poses.append(filtered_poses)
                            all_scores.append(filtered_scores)

                    collected = sum(len(p) for p in all_poses)
                    print(f"[GraspGen] Round {round_i+1}: collected {collected}/{TARGET_GRASPS} collision-free grasps")
                    if collected >= TARGET_GRASPS:
                        break

                if len(all_poses) > 0:
                    all_poses = np.concatenate(all_poses, axis=0)[:TARGET_GRASPS]
                    all_scores = np.concatenate(all_scores, axis=0)[:TARGET_GRASPS]
                else:
                    all_poses = np.empty((0, 4, 4))
                    all_scores = np.empty(0)

                # Put back into result for downstream visualization
                result["poses"] = all_poses
                result["scores"] = all_scores

                # --- Save point clouds for each collision-free grasp ---
                import trimesh

                gripper_open_mesh = trimesh.load("/home/xuran-yao/Desktop/g2_gripper/gripper_down.obj")
                gripper_close_mesh = trimesh.load("/home/xuran-yao/Desktop/g2_gripper/g2_close.obj")

                gripper_open_pts = gripper_open_mesh.sample(1024)
                gripper_close_pts = gripper_close_mesh.sample(1024)
                tip_offset_local = np.array([0.0, 0.0, 0.019])

                if len(obj_pts) > 4096:
                    save_idx = np.random.choice(len(obj_pts), 4096, replace=False)
                    obj_pts_save = obj_pts[save_idx]
                else:
                    obj_pts_save = obj_pts

                save_dir = os.path.join(self.camera.output_dir, "grasp_pcds")
                os.makedirs(save_dir, exist_ok=True)

                # Save one object point cloud + 50 gripper open/close
                np.save(os.path.join(save_dir, "object.npy"), obj_pts_save)

                poses_save = all_poses.copy()
                if len(poses_save) > 0:
                    poses_save[:, 3, 3] = 1
                for i, (pose, score) in enumerate(zip(poses_save, all_scores)):
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    tip_world = R @ tip_offset_local
                    open_world = (R @ gripper_open_pts.T).T + t + tip_world
                    close_world = (R @ gripper_close_pts.T).T + t + tip_world

                    np.save(os.path.join(save_dir, f"grasp_{i:03d}_gripper_open.npy"), open_world)
                    np.save(os.path.join(save_dir, f"grasp_{i:03d}_gripper_close.npy"), close_world)

                print(f"[GraspGen] Saved {len(poses_save)} grasps + 1 object to {save_dir}")

                result["obj_points"] = obj_pts
                result["obj_colors"] = obj_colors
                result["scene_points"] = points.reshape(-1, 3)
                result["scene_colors"] = colors.reshape(-1, 3)
                self.after(0, lambda: self._on_grasp_done(result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                msg = str(e)
                self.after(0, lambda: self._on_grasp_error(msg))

        threading.Thread(target=_do, daemon=True).start()

    def _build_pointcloud(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build full scene point cloud from frozen frame using lingbot-depth."""
        import torch

        model, device = self.camera._get_refine_model()
        image_rgb = self._frozen_color
        h, w = image_rgb.shape[:2]

        image_tensor = torch.tensor(
            image_rgb / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1).unsqueeze(0)

        depth_m = self._frozen_depth_raw.astype(np.float32) / 1000.0
        dh, dw = depth_m.shape
        if (dh, dw) != (h, w):
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth_tensor = torch.tensor(depth_m, dtype=torch.float32, device=device)

        if self.camera.depth_intrinsic and self.camera.depth_intrinsic[0] > 0:
            fx, fy, cx, cy = self.camera.depth_intrinsic[:4]
        else:
            fx = fy = 0.6 * w
            cx, cy = w / 2.0, h / 2.0

        intrinsics = np.array([
            [fx / w, 0.0, cx / w],
            [0.0, fy / h, cy / h],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            output = model.infer(image_tensor, depth_in=depth_tensor, intrinsics=intrinsics_tensor)

        points = output['points'].squeeze().cpu().numpy()  # (H, W, 3)
        return points, image_rgb

    def _on_grasp_done(self, result: dict) -> None:
        self._grasp_results = result
        n = len(result["scores"])
        self.btn_grasp.configure(state="normal")

        if n == 0:
            self.status_var.set("No grasps found. Try adjusting parameters.")
            return

        best = result["scores"].max()
        self.status_var.set(f"Generated {n} grasp(s). Best confidence: {best:.3f}. Opening 3D viewer...")

        # Launch 3D visualization in a separate thread
        threading.Thread(target=self._visualize_grasps_3d, daemon=True).start()

    def _on_grasp_error(self, msg: str) -> None:
        self.btn_grasp.configure(state="normal")
        self.status_var.set(f"Grasp generation failed: {msg}")

    def _visualize_grasps_3d(self) -> None:
        """Visualize grasps using GraspGen's meshcat visualizer with gripper models."""
        from grasp_gen.utils.meshcat_utils import (
            create_visualizer,
            get_color_from_score,
            visualize_grasp,
            visualize_pointcloud,
        )

        result = self._grasp_results
        if result is None:
            return

        obj_pts = result["obj_points"]
        obj_colors = result["obj_colors"]
        scene_pts = result["scene_points"]
        scene_colors = result["scene_colors"]
        poses = result["poses"]
        scores = result["scores"]
        gripper = self.combo_gripper.get()

        try:
            vis = create_visualizer(clear=True)
        except Exception as e:
            self.after(0, lambda: self.status_var.set(
                f"meshcat connection failed: {e}. Run 'meshcat-server' in a separate terminal first."
            ))
            return

        # Filter valid scene points for visualization
        valid = np.isfinite(scene_pts).all(axis=-1) & (scene_pts[:, 2] > 0)
        visualize_pointcloud(vis, "scene/full", scene_pts[valid], color=scene_colors[valid])

        # Highlight object point cloud
        visualize_pointcloud(vis, "scene/object", obj_pts, color=obj_colors)

        # Draw each grasp with gripper shape, colored by confidence
        for i, (pose, score) in enumerate(zip(poses, scores)):
            color = get_color_from_score(float(score), use_255_scale=True).astype(int).tolist()
            visualize_grasp(vis, f"grasps/{i:03d}", pose, color=color, gripper_name=gripper)

        self.after(0, lambda: self.status_var.set(
            f"Meshcat visualization ready — {len(poses)} grasp(s). Open the meshcat URL in your browser."
        ))

    # ================================================================ UTILITIES
    def _snapshot(self) -> None:
        ts = self.camera.save_image()
        if ts:
            self.status_var.set(f"Snapshot saved ({ts}) -> {self.camera.output_dir}/")
        else:
            self.status_var.set("Snapshot failed — no frames available.")

    def _toggle_record(self) -> None:
        if not self.camera.recording:
            ts = self.camera.start_recording()
            self.btn_record.configure(text="Stop Rec", fg_color=C_DANGER)
            self.status_var.set(f"Recording started ({ts})...")
        else:
            self.camera.stop_recording()
            self.btn_record.configure(text="Record", fg_color=C_CARD)
            self.status_var.set(f"Recording saved -> {self.camera.output_dir}/")

    def _export_ply(self) -> None:
        self.btn_ply.configure(state="disabled")
        self.status_var.set("Generating point cloud...")
        self.update_idletasks()

        def _do():
            path = self.camera.save_ply()
            if path:
                self.after(0, lambda: self._on_ply_done(f"Point cloud saved -> {path}"))
            else:
                self.after(0, lambda: self._on_ply_done("PLY export failed."))

        threading.Thread(target=_do, daemon=True).start()

    def _on_ply_done(self, msg: str) -> None:
        self.status_var.set(msg)
        if self.connected:
            self.btn_ply.configure(state="normal")

    def _reset_pipeline(self) -> None:
        """Reset to live streaming mode, freeing GPU models."""
        import torch

        self._frozen_color = None
        self._frozen_depth_raw = None
        self._frozen_depth_vis = None
        self._detected = False
        self._click_points.clear()
        self._selected_mask = None
        self._grasp_results = None

        # Release GraspGen to free VRAM for next segmentation round
        if self._grasp_model is not None:
            self._grasp_model.sampler = None
            self._grasp_model = None

        torch.cuda.empty_cache()

        if self.connected:
            self.streaming = True
            self.btn_detect.configure(state="disabled")
            self.btn_grasp.configure(state="disabled")
            self._advance_step(Step.CAPTURE)
            self.status_var.set("Reset — streaming live. Press Capture to freeze a frame.")
        else:
            self._advance_step(Step.CONNECT)

    # ================================================================ CLEANUP
    def _on_close(self) -> None:
        self._stop_event.set()
        self.streaming = False
        try:
            self.camera.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
