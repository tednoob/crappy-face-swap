import os
import sys
import cv2
import signal
import logging
import threading
from typing import Optional

import numpy as np
import onnxruntime
import onnx

from gfpgan.utils import GFPGANer
from onnx import numpy_helper
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils.face_align import estimate_norm, norm_crop
from .util import Frame, HAS_CUDA, PROVIDERS

import cProfile
import pstats


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaceSwap:
    def __init__(self, upscale: bool = True, profile: bool = False):
        self.profile = profile
        self.profiler = cProfile.Profile()
        signal.signal(signal.SIGINT, self.signal_handler)

        # Face analysis
        self.face_analyser = FaceAnalysis(
            name="buffalo_l", root="./", providers=PROVIDERS
        )
        self.face_analyser.prepare(ctx_id=0, det_thresh=0.5)

        # Corse face swap
        self.inswapper_model_file = "models/inswapper_128.onnx"
        model = onnx.load(self.inswapper_model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        self.session = onnxruntime.InferenceSession(
            self.inswapper_model_file,
            providers=PROVIDERS,
            provider_options=None,
        )
        self.inputs = self.session.get_inputs()
        self.input_names = [i.name for i in self.inputs]
        self.outputs = self.session.get_outputs()
        self.output_names = [o.name for o in self.outputs]
        assert self.output_names == ["output"]
        assert self.input_names == ["target", "source"]
        self.output_shape = self.outputs[0].shape
        self.input_shape = self.inputs[0].shape
        self.input_size = tuple(self.input_shape[2:4][::-1])
        self.fill_scale = 1.1

        # Upscale face
        self.scale_lock = threading.Lock()
        if upscale:
            self.upscaler = GFPGANer(
                model_path="models/GFPGANv1.4.pth",
                upscale=1,
                device="cuda" if HAS_CUDA else "cpu",
            )
        else:
            self.upscaler = None

    def signal_handler(self, signal, frame):
        self.print_stats()
        print("\nprogram exiting gracefully")
        sys.exit(0)

    def upscale(self, bgr_fake: Frame) -> Frame:
        if self.upscaler is not None:
            with self.scale_lock:
                _, restored_faces, _ = self.upscaler.enhance(
                    bgr_fake, has_aligned=True, paste_back=False
                )
                return restored_faces[0]
        return bgr_fake

    def warp(self, img: Frame, target_face: Face, target_img: Frame) -> Frame:
        M = estimate_norm(target_face.kps, img.shape[0])
        IM = cv2.invertAffineTransform(M)
        return cv2.warpAffine(
            img,
            IM,
            (target_img.shape[1], target_img.shape[0]),
            dst=target_img,
            borderMode=cv2.BORDER_TRANSPARENT,
            # borderValue=None,
        )

    def inswap(
        self, target_frame: Frame, target_face: Face, source_face: Face
    ) -> Frame:
        actual_image = norm_crop(target_frame, target_face.kps, self.input_size[0])

        # https://answers.opencv.org/question/208377/output-of-blobfromimage-function/
        blob = cv2.dnn.blobFromImage(
            actual_image,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        pred = self.session.run(
            self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent}
        )[0]

        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return bgr_fake

    def fill_face(self, target_face: Face, img_mask: Frame):
        # Face KPS is 5x2 array of x,y coordinates
        # Left eye, right eye, nose, left mouth, right mouth

        M = estimate_norm(target_face.kps, img_mask.shape[0])
        target2crop_kps = cv2.transform(target_face.kps.reshape(1, -1, 2), M)[0]
        target2crop_landmark_2d_106 = cv2.transform(
            target_face.landmark_2d_106.reshape(1, -1, 2), M
        )[0]

        hull_points = cv2.convexHull(target2crop_landmark_2d_106)

        if self.fill_scale != 1.0:
            center = np.mean(hull_points, axis=0)
            translated_points = hull_points - center
            scaled_points = translated_points * self.fill_scale
            hull_points = scaled_points + center
            hull_points[hull_points >= img_mask.shape[0]] = img_mask.shape[0] - 1
            hull_points[hull_points < 0] = 0
            target2crop_kps = target2crop_kps - center
            target2crop_kps = target2crop_kps * self.fill_scale
            target2crop_kps = target2crop_kps + center
            target2crop_kps[target2crop_kps >= img_mask.shape[0]] = (
                img_mask.shape[0] - 1
            )
            target2crop_kps[target2crop_kps < 0] = 0

        center_eye = np.mean(target2crop_kps[0:2], axis=0)
        center_mouth = np.mean(target2crop_kps[3:5], axis=0)
        radius = int(0.7 * self.fill_scale * np.linalg.norm(center_eye - center_mouth))

        pts = np.array([hull_points], dtype=np.int32)
        cv2.fillPoly(img_mask, pts, (1, 1, 1))
        cv2.circle(
            img_mask,
            (int(center_eye[0]), int(center_eye[1])),
            radius,
            (1, 1, 1),
            -1,
        )

    def combine(
        self,
        target_frame: Frame,
        target_face: Face,
        bgr_fake: Frame,
    ) -> Frame:
        """Operates in place on frame."""
        side = bgr_fake.shape[0]
        bgr_target = norm_crop(target_frame, target_face.kps, side)

        img_mask = np.full((side, side), 0, dtype=np.float32)
        self.fill_face(target_face, img_mask)
        blur = 2 * int(0.1 * side) + 1
        img_mask = cv2.GaussianBlur(img_mask, (blur, blur), 0)
        img_mask = np.reshape(img_mask, [side, side, 1])
        bgr_fake = img_mask * bgr_fake + (1 - img_mask) * bgr_target.astype(np.float32)
        self.warp(bgr_fake.astype(np.uint8), target_face, target_frame)

    def get_faces(self, img: Frame):
        return self.face_analyser.get(img)

    def get_face(self, image_path: str) -> Optional[Face]:
        if not os.path.exists(image_path):
            raise Exception(f"File {image_path} does not exist")
        if faces := self.get_faces(cv2.imread(image_path)):
            return faces[0]
        raise Exception("No face found")

    def swap_face(self, source_face: Face, frame: Frame) -> Frame:
        """Operates in place on frame."""
        if self.profile:
            self.profiler.enable()
        for target_face in self.get_faces(frame):
            bgr_fake = self.inswap(frame, target_face, source_face)
            bgr_fake = self.upscale(bgr_fake)
            self.combine(frame, target_face, bgr_fake)
        if self.profile:
            self.profiler.disable()

    def print_stats(self, num=20):
        if self.profile:
            try:
                stats = pstats.Stats(self.profiler).sort_stats("cumulative")
                stats.print_stats(num)
            except Exception:
                pass


class CameraSwap(FaceSwap):
    def __init__(self, camera_id: int, face_path: str, profile=True):
        super().__init__(upscale=False, profile=profile)
        self.fill_scale = 1.3
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(camera_id)
        self.source_face = self.get_face(face_path)

    def get_face(self, image_path: str) -> Optional[Face]:
        if not os.path.exists(image_path):
            raise Exception(f"File {image_path} does not exist")
        if faces := super().get_faces(cv2.imread(image_path)):
            return faces[0]
        raise Exception("No face found")

    def get_faces(self, img: Frame, max_num=0):
        bboxes, kpss = self.face_analyser.det_model.detect(
            img, max_num=max_num, metric="default"
        )
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            ret.append(face)
        return ret

    def fill_face(self, target_face: Face, img_mask: Frame):
        # Face KPS is 5x2 array of x,y coordinates
        # Left eye, right eye, nose, left mouth, right mouth

        M = estimate_norm(target_face.kps, img_mask.shape[0])
        target2crop_kps = cv2.transform(target_face.kps.reshape(1, -1, 2), M)[0]

        hull_points = cv2.convexHull(target2crop_kps)

        if self.fill_scale != 1.0:
            center = np.mean(hull_points, axis=0)
            translated_points = hull_points - center
            scaled_points = translated_points * self.fill_scale
            hull_points = scaled_points + center
            hull_points[hull_points >= img_mask.shape[0]] = img_mask.shape[0] - 1
            hull_points[hull_points < 0] = 0
            target2crop_kps = target2crop_kps - center
            target2crop_kps = target2crop_kps * self.fill_scale
            target2crop_kps = target2crop_kps + center
            target2crop_kps[target2crop_kps >= img_mask.shape[0]] = (
                img_mask.shape[0] - 1
            )
            target2crop_kps[target2crop_kps < 0] = 0

        center_eye = np.mean(target2crop_kps[0:2], axis=0)
        center_mouth = np.mean(target2crop_kps[3:5], axis=0)
        radius = int(0.7 * self.fill_scale * np.linalg.norm(center_eye - center_mouth))

        pts = np.array([hull_points], dtype=np.int32)
        cv2.fillPoly(img_mask, pts, (1, 1, 1))
        cv2.circle(
            img_mask,
            (int(center_eye[0]), int(center_eye[1])),
            radius,
            (1, 1, 1),
            -1,
        )

    def get_frame(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(self.camera_id)
        success, frame = self.camera.read()
        if success:
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            self.swap_face(source_face=self.source_face, frame=frame)
            return frame
        self.camera.release()
        self.camera = None
        return None
