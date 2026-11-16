"""mp4 output for co-sim camera views and report videos."""

import numpy as np


class Mp4Writer:
    """Streams RGB uint8 frames to an mp4 via OpenCV."""

    def __init__(self, out_path, fps):
        self.out_path = str(out_path)
        self.fps = float(fps)
        self.writer = None
        self.frame_count = 0

    def append_data(self, frame):
        import cv2

        frame = np.ascontiguousarray(frame)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Mp4Writer({self.out_path}): OpenCV could not open the video writer")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def write_mp4(out_path, frames, fps=10):
    if not frames:
        raise ValueError(f"write_mp4({out_path}): no frames")
    writer = Mp4Writer(out_path, fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
