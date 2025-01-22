import cv2
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import ffmpeg
import numpy as np
from .util import Frame, HAS_GPU

if TYPE_CHECKING:
    from .image import FaceSwap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MediaDirector(object):
    def __init__(
        self,
        swapper: "FaceSwap",
        in_filename: str,
        out_filename: str,
    ):
        self.swapper = swapper
        self.in_filename = in_filename
        self.out_filename = out_filename
        self.input_process = None
        self.output_process = None

        self.probe = ffmpeg.probe(self.in_filename)
        self.video_info = next(
            s for s in self.probe["streams"] if s["codec_type"] == "video"
        )
        self.width = int(self.video_info["width"])
        self.height = int(self.video_info["height"])
        rate = self.video_info["r_frame_rate"].split("/", 1)
        self.framerate = float(rate[0]) / float(rate[1])

        self.frame_buffer_lock: threading.Lock = threading.Lock()
        self._input_buffer = []
        self._output_buffer = []

    def append_output_frame(self, ix, frame):
        with self.frame_buffer_lock:
            self._output_buffer.append((ix, frame))

    def start_ffmpeg_input_process(self):
        logger.info("Starting ffmpeg input process")
        args = [
            "ffmpeg",
            "-i",
            self.in_filename,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:",
        ]
        self.input_process = subprocess.Popen(args, stdout=subprocess.PIPE)

    def start_ffmpeg_output_process(self, scale=1):
        logger.info("Starting ffmpeg output process")
        args = [
            "ffmpeg",
            "-framerate",  # FFMPEG expects this argument to be first
            str(self.framerate),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            "pipe:",
            "-i",
            self.in_filename,
            "-colorspace",
            "bt709",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "h264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-map",
            "0:v?",
            "-map",
            "1:a?",
            self.out_filename.replace(".webm", ".mp4"),
            "-y",
        ]
        self.output_process = subprocess.Popen(args, stdin=subprocess.PIPE)

    def read_frame(self):
        # Note: RGB24 == 3 bytes per pixel.
        frame_size = self.width * self.height * 3
        in_bytes = self.input_process.stdout.read(frame_size)
        if len(in_bytes) == 0:
            frame = None
        else:
            assert len(in_bytes) == frame_size
            frame = cv2.cvtColor(
                np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3]),
                cv2.COLOR_RGB2BGR,
            )
        return frame

    def get_frame(self):
        with self.frame_buffer_lock:
            if self._input_frame_buffer:
                return self._input_frame_buffer.pop(0)
            else:
                return None

    def process_frame(self, frame_no: int, frame: Frame):
        self.swapper.swap_face(frame)
        self.append_output_frame(frame_no, frame)

    def write_frame(self, frame):
        self.output_process.stdin.write(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8).tobytes()
        )

    def run(self):
        max_threads = 8 if HAS_GPU else 1
        self.start_ffmpeg_input_process()
        self.start_ffmpeg_output_process()
        frame = 0
        while True:
            frames = []
            for _ in range(max_threads):
                in_frame = self.read_frame()
                if in_frame is not None:
                    frames.append((frame, in_frame))
                    frame += 1
            if not frames:
                logger.info("End of input stream")
                break
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = []
                for ix, in_frame in frames:
                    future = executor.submit(
                        self.process_frame,
                        frame_no=ix,
                        frame=in_frame,
                    )
                    futures.append(future)
                for future_done in as_completed(futures):
                    future_done.result()

            with self.frame_buffer_lock:
                while self._output_buffer:
                    self._output_buffer.sort(key=lambda x: x[0])
                    _, out_frame = self._output_buffer.pop(0)
                    self.write_frame(out_frame)

        logger.info("Waiting for ffmpeg input process")
        self.input_process.wait()

        logger.info("Waiting for ffmpeg output process")
        self.output_process.stdin.close()
        self.output_process.wait()

        logger.info("Done")

    def run_single(self):
        start, frames, read, process, write = time.time(), 0, 0, 0, 0
        tic = time.time()
        self.start_ffmpeg_input_process()
        read += time.time() - tic

        tic = time.time()
        self.start_ffmpeg_output_process()
        write += time.time() - tic

        while True:
            frames += 1
            tic = time.time()
            frame = self.read_frame()
            read += time.time() - tic

            if frame is None:
                logger.info("End of input stream")
                break
            tic = time.time()
            self.swapper.swap_face(frame)
            process += time.time() - tic

            tic = time.time()
            self.write_frame(frame)
            write += time.time() - tic
            if (total := time.time() - start) > 10:
                print(
                    f"Frames: {frames}, Total: {total}, Read: {round(100*read/total)}, Process: {round(100*process/total)}, Write: {round(100*write/total)}, Elsewhere: {round(100*(total - read - process - write)/total)}"
                )
                start, frames, read, process, write = time.time(), 0, 0, 0, 0

        logger.info("Waiting for ffmpeg input process")
        self.input_process.wait()

        logger.info("Waiting for ffmpeg output process")
        self.output_process.stdin.close()
        self.output_process.wait()
