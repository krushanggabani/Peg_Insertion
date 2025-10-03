# recorder.py
import os
import subprocess
import pybullet as p
import numpy as np

# Optional deps
try:
    import imageio
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    import imageio_ffmpeg
    _HAS_IMAGEIO_FFMPEG = True
except Exception:
    _HAS_IMAGEIO_FFMPEG = False


def _has_system_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


class PBRecorder:
    """
    Unified recorder:
      - PRIMARY: PyBullet GUI MP4 (needs system ffmpeg in PATH).
      - FALLBACK: Manual MP4 via imageio-ffmpeg (uses camera frames).
      - GIF: imageio, via camera frames.

    Call:
      rec = PBRecorder()
      rec.start_video("logs/run.mp4", fps=60)     # starts GUI mp4 or fallback writer
      rec.start_gif("logs/run.gif", fps=20, size=(640,480), target=(0.5,0,0.6), distance=1.0, yaw=90, pitch=-35)
      ...
      # per loop:
      rec.grab_frame()    # feeds MP4 fallback + GIF
      ...
      rec.stop_video()
      rec.save_gif()
    """

    def __init__(self):
        # MP4 via PyBullet GUI
        self._video_logging_id = None
        # MP4 fallback via imageio-ffmpeg
        self._mp4_writer = None
        self._mp4_path = None
        self._mp4_fps = 30

        # GIF buffer
        self._gif_frames = []
        self._gif_path = None
        self._gif_fps = 20

        # camera
        self._size = (640, 480)
        self._view_matrix = None
        self._proj_matrix = None

    # ---------- MP4 ----------
    def start_video(self, path: str, fps: int = 60):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._mp4_path = path
        self._mp4_fps = fps

        # Prefer PyBullet GUI logger if possible
        can_gui_mp4 = p.isConnected(p.GUI) and _has_system_ffmpeg()
        if can_gui_mp4:
            self._video_logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, fileName=path)
            try:
                # helps GUI render per step
                p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            except Exception:
                pass
            print(f"[PBRecorder] ▶️ Started GUI MP4 recording to: {path}")
            return

        # Fallback: manual MP4 writer with imageio-ffmpeg
        if not (_HAS_IMAGEIO and _HAS_IMAGEIO_FFMPEG):
            print("[PBRecorder] ⚠️ No system ffmpeg found and imageio/imageio-ffmpeg not installed."
                  " MP4 fallback unavailable. Use GIF or install ffmpeg.")
            return

        self._mp4_writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        print(f"[PBRecorder] ▶️ Started fallback MP4 (imageio-ffmpeg) to: {path}")

    def stop_video(self):
        if self._video_logging_id is not None:
            p.stopStateLogging(self._video_logging_id)
            print("[PBRecorder] ⏹️ Stopped GUI MP4 recording.")
            self._video_logging_id = None

        if self._mp4_writer is not None:
            try:
                self._mp4_writer.close()
            except Exception:
                pass
            print("[PBRecorder] 💾 MP4 (fallback) written:", self._mp4_path)
            self._mp4_writer = None
            self._mp4_path = None

    # ---------- GIF ----------
    def start_gif(self, path: str, fps: int = 20, size=(640, 480),
                  view_matrix=None, proj_matrix=None,
                  fov_deg: float = 45.0, near: float = 0.01, far: float = 5.0,
                  target=(0.5, 0.0, 0.6), distance: float = 1.0, yaw: float = 90.0, pitch: float = -35.0):
        if not _HAS_IMAGEIO:
            raise RuntimeError("imageio not installed. `pip install imageio` to enable GIF export.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._gif_frames = []
        self._gif_path = path
        self._gif_fps = fps
        self._size = size

        # camera setup
        if view_matrix is None:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target, distance=distance, yaw=yaw, pitch=pitch, roll=0.0, upAxisIndex=2
            )
        if proj_matrix is None:
            aspect = float(size[0]) / float(size[1])
            proj_matrix = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=aspect, nearVal=near, farVal=far)

        self._view_matrix = view_matrix
        self._proj_matrix = proj_matrix
        print(f"[PBRecorder] 🧰 GIF capture primed (saving to {path} @ {fps} fps, {size[0]}x{size[1]}).")

    def save_gif(self, loop: int = 0):
        if not _HAS_IMAGEIO:
            print("[PBRecorder] imageio not available, cannot save GIF.")
            return
        if not self._gif_frames:
            print("[PBRecorder] No frames captured; GIF not written.")
            return
        imageio.mimsave(self._gif_path, self._gif_frames, fps=self._gif_fps, loop=loop)
        print(f"[PBRecorder] 💾 GIF written: {self._gif_path} ({len(self._gif_frames)} frames)")
        self._gif_frames = []
        self._gif_path = None

    # ---------- unified frame grab ----------
    def grab_frame(self):
        """
        Capture one frame from a stable camera and:
          - append to GIF buffer (if active)
          - stream to MP4 fallback writer (if active)
        (GUI MP4 logging doesn't need this; it records GUI automatically.)
        """
        need_camera = (self._gif_path is not None) or (self._mp4_writer is not None)
        if not need_camera:
            return

        # If no explicit camera, use current debug camera
        if (self._view_matrix is None) or (self._proj_matrix is None):
            w, h, view, proj, _, _ = p.getDebugVisualizerCamera()
            self._view_matrix, self._proj_matrix = view, proj
            self._size = (w, h)

        w, h = self._size
        _, _, px, _, _ = p.getCameraImage(
            width=w, height=h,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if p.isConnected(p.GUI) else p.ER_TINY_RENDERER,
        )
        frame = np.reshape(px, (h, w, 4))[:, :, :3]  # drop alpha

        if self._gif_path is not None:
            self._gif_frames.append(frame)

        if self._mp4_writer is not None:
            self._mp4_writer.append_data(frame)
