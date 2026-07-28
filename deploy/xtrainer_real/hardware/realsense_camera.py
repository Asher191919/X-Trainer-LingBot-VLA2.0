import threading

import numpy as np


class RealSenseCamera:
    """Minimal RealSense RGB camera wrapper for xtrainer runtime."""

    def __init__(self, serial: str, *, width: int = 640, height: int = 480, fps: int = 30):
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._pipeline = None
        self._connected = False
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._connected:
            return

        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        pipeline.start(config)

        # Warm up frames.
        for _ in range(10):
            pipeline.wait_for_frames(timeout_ms=1000)

        self._pipeline = pipeline
        self._connected = True

    def async_read(self, timeout_ms: int = 50) -> np.ndarray | None:
        if not self._connected or self._pipeline is None:
            return None
        with self._lock:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        bgr = np.asanyarray(color_frame.get_data())
        # Convert BGR -> RGB.
        return np.ascontiguousarray(bgr[..., ::-1])

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._connected = False
