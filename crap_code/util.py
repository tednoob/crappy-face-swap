import cv2
import logging
import onnx
import onnxruntime
import numpy as np
from sys import platform
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Frame = np.ndarray[Any, Any]

# https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
PROVIDERS = (
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)
HAS_CUDA = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
HAS_METAL = "CoreMLExecutionProvider" in onnxruntime.get_available_providers()
HAS_GPU = HAS_CUDA or HAS_METAL


def normalize_path(path: str) -> str:
    """Converts a Windows path to a WSL path if necessary."""
    print("Platform:", platform, path)
    if ":" in path and platform.startswith("linux"):
        parts = path.split(":", 1)
        drive = parts[0].lower()
        drive_path = parts[1].replace("\\", "/")
        path = f"/mnt/{drive}{drive_path}"
    return path


def is_image(file: str) -> bool:
    """Checks if a file is an image."""
    return file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"))


def is_video(file: str) -> bool:
    """Checks if a file is a video."""
    return file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm")
    )


class OnnxWrapper:
    def __init__(self, model_path: str):

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 0
        self.model_path = model_path
        model = onnx.load(self.model_path)
        graph = model.graph
        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            so,
            providers=PROVIDERS,
            provider_options=None,
        )
        self.inputs = self.session.get_inputs()
        self.input_names = [i.name for i in self.inputs]
        self.outputs = self.session.get_outputs()
        self.output_names = [o.name for o in self.outputs]
        self.output_shape = self.outputs[0].shape
        self.input_shape = self.inputs[0].shape
        self.input_size = tuple(self.input_shape[2:4][::-1])


class OnnxInSwapper(OnnxWrapper):
    def __init__(self, model_path):
        super().__init__(model_path)
        assert self.output_names == ["output"], self.output_names
        assert self.input_names == ["target", "source"], self.input_names

    def get_blob(self, image: Frame, swapRB=True) -> Frame:
        # https://answers.opencv.org/question/208377/output-of-blobfromimage-function/
        return cv2.dnn.blobFromImage(
            image,
            1.0 / 255.0,
            self.input_size,
            (0, 0, 0),
            swapRB=swapRB,
        )

    def run(self, target, source, swapRB=True) -> Frame:
        session_inputs = {"target": self.get_blob(target, swapRB), "source": source}
        pred = self.session.run(
            self.output_names,
            session_inputs,
        )[0]
        image = pred.transpose((0, 2, 3, 1))[0]
        image = np.clip(255 * image, 0, 255).astype(np.uint8)[:, :, ::-1]
        return image


class OnnxGfpGan(OnnxWrapper):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        assert self.output_names == ["output"], self.output_names
        assert self.input_names == ["input"], self.input_names

    def normalize(self, frame, mean, std):
        """Normalize an uint8 image with mean and standard deviation.

        Taken from torchvision, see :class:`~torchvision.transforms.Normalize` for more details.

        Args:
            frame (numpy.ndarray): uint8 tensor image of size (H, W, C) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.

        Returns:
            numpy.ndarray: Normalized blob.
        """
        if frame.ndim < 3:
            raise ValueError(
                f"Expected frame to be an image of size (H, W, C). Got np.shape = {frame.shape}"
            )
        blob = (
            np.transpose(
                np.expand_dims(
                    cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (512, 512)),
                    axis=0,
                ),
                (0, 3, 1, 2),
            ).astype(np.float32)
            / 255.0
        )

        dtype = blob.dtype
        mean = np.asarray(mean, dtype=dtype)
        std = np.asarray(std, dtype=dtype)
        if (std == 0).any():
            raise ValueError(
                f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
            )
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1, 1)
        if std.ndim == 1:
            std = std.reshape(-1, 1, 1)
        return (blob - mean) / std

    def get_blob(self, image, swapRB=True):
        return self.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def run(self, input, swapRB=True) -> Frame:
        session_inputs = {"input": self.get_blob(input, swapRB)}
        pred = self.session.run(
            self.output_names,
            session_inputs,
        )[0]
        image = np.clip(pred.transpose((0, 2, 3, 1))[0], -1, 1)
        image = (255.0 / 2 * (image + 1)).astype(np.uint8)[:, :, ::-1]
        return image
