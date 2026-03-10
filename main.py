"""Orbbec Gemini 336L Visualization GUI — main entry point."""

import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageTk

import customtkinter as ctk

from camera import OrbbecCamera

# ---------- Theme ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CANVAS_W, CANVAS_H = 640, 480
FPS_INTERVAL = 0.5  # seconds between FPS updates


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Orbbec Gemini 336L Viewer")
        self.geometry("1360x700")
        self.minsize(1100, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.camera = OrbbecCamera(output_dir="output")
        self.connected = False
        self.streaming = False
        self._stop_event = threading.Event()

        # FPS tracking
        self._frame_count = 0
        self._fps_time = time.time()
        self._fps = 0.0

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # --- Top bar ---
        top = ctk.CTkFrame(self, height=40)
        top.pack(fill="x", padx=10, pady=(10, 5))

        self.btn_connect = ctk.CTkButton(
            top, text="Connect", width=120, command=self._toggle_connect
        )
        self.btn_connect.pack(side="left", padx=5)

        self.lbl_fps = ctk.CTkLabel(top, text="FPS: --", width=90)
        self.lbl_fps.pack(side="right", padx=10)

        # --- Canvas area ---
        canvas_frame = ctk.CTkFrame(self)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: Color
        left = ctk.CTkFrame(canvas_frame)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        ctk.CTkLabel(left, text="Color", font=ctk.CTkFont(size=14, weight="bold")).pack(
            pady=(5, 2)
        )
        self.color_label = ctk.CTkLabel(left, text="")
        self.color_label.pack(fill="both", expand=True, padx=5, pady=5)

        # Right: Depth
        right = ctk.CTkFrame(canvas_frame)
        right.pack(side="right", fill="both", expand=True, padx=(5, 0))
        ctk.CTkLabel(
            right, text="Depth", font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(5, 2))
        self.depth_label = ctk.CTkLabel(right, text="")
        self.depth_label.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Bottom controls ---
        bottom = ctk.CTkFrame(self, height=50)
        bottom.pack(fill="x", padx=10, pady=(5, 5))

        self.btn_snap = ctk.CTkButton(
            bottom,
            text="\U0001f4f7  Snapshot",
            width=140,
            command=self._snapshot,
            state="disabled",
        )
        self.btn_snap.pack(side="left", padx=8)

        self.btn_record = ctk.CTkButton(
            bottom,
            text="\U0001f3a5  Record",
            width=140,
            command=self._toggle_record,
            state="disabled",
        )
        self.btn_record.pack(side="left", padx=8)

        self.btn_ply = ctk.CTkButton(
            bottom,
            text="\U0001f4e6  Export PLY",
            width=140,
            command=self._export_ply,
            state="disabled",
        )
        self.btn_ply.pack(side="left", padx=8)

        # --- Status bar ---
        self.status_var = ctk.StringVar(value="Ready \u2014 press Connect to start.")
        self.lbl_status = ctk.CTkLabel(
            self, textvariable=self.status_var, anchor="w", font=ctk.CTkFont(size=12)
        )
        self.lbl_status.pack(fill="x", padx=12, pady=(0, 8))

    # --------------------------------------------------------- Connection
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
        self.btn_connect.configure(text="Disconnect", state="normal")
        self.btn_snap.configure(state="normal")
        self.btn_record.configure(state="normal")
        self.btn_ply.configure(state="normal")
        self.status_var.set("Connected \u2014 streaming live.")

    def _on_connect_error(self, msg: str) -> None:
        self.btn_connect.configure(state="normal")
        self.status_var.set(f"Connection failed: {msg}")

    def _disconnect(self) -> None:
        self._stop_event.set()
        self.streaming = False
        self.camera.stop()
        self.connected = False

        self.btn_connect.configure(text="Connect")
        self.btn_snap.configure(state="disabled")
        self.btn_record.configure(state="disabled")
        self.btn_ply.configure(state="disabled")
        self.status_var.set("Disconnected.")

    # --------------------------------------------------------- Streaming
    def _stream_loop(self) -> None:
        """Background thread: continuously grab frames and push to GUI."""
        while not self._stop_event.is_set():
            ok = self.camera.get_frames()
            if ok:
                self._frame_count += 1
                self.after(0, self._update_display)
            else:
                time.sleep(0.005)

    def _update_display(self) -> None:
        """Called on the main thread to refresh canvas images."""
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
        """Resize BGR image to fit label and display it."""
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
        label._photo = photo  # prevent GC

    # --------------------------------------------------------- Actions
    def _snapshot(self) -> None:
        ts = self.camera.save_image()
        if ts:
            self.status_var.set(f"Snapshot saved  ({ts})  ->  {self.camera.output_dir}/")
        else:
            self.status_var.set("Snapshot failed \u2014 no frames available.")

    def _toggle_record(self) -> None:
        if not self.camera.recording:
            ts = self.camera.start_recording()
            self.btn_record.configure(
                text="\u23f9  Stop Recording", fg_color="#c0392b"
            )
            self.status_var.set(f"Recording started  ({ts}) ...")
        else:
            self.camera.stop_recording()
            self.btn_record.configure(
                text="\U0001f3a5  Record", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]
            )
            self.status_var.set(
                f"Recording saved  ->  {self.camera.output_dir}/"
            )

    def _export_ply(self) -> None:
        self.status_var.set("Generating point cloud...")
        self.update_idletasks()

        def _do():
            path = self.camera.save_ply()
            if path:
                self.after(
                    0, lambda: self.status_var.set(f"Point cloud saved -> {path}")
                )
            else:
                self.after(
                    0,
                    lambda: self.status_var.set(
                        "PLY export failed \u2014 no depth data available."
                    ),
                )

        threading.Thread(target=_do, daemon=True).start()

    # --------------------------------------------------------- Cleanup
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
