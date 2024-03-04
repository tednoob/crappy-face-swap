import os
import sys
import cv2
import time
import signal
import logging
import threading
from typing import Optional

import numpy as np

from gfpgan.utils import GFPGANer
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils.face_align import estimate_norm, norm_crop
from crap_code.util import Frame, OnnxGfpGan, PROVIDERS, OnnxInSwapper

import cProfile
import pstats


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaceSwap:
    def __init__(self, upscale: bool = True, profile: bool = False):
        self.running = False
        self.profile = profile
        self.profiler = cProfile.Profile()
        signal.signal(signal.SIGINT, self.signal_handler)

        # Face analysis
        self.face_analyser = FaceAnalysis(
            name="buffalo_l", root="./", providers=PROVIDERS
        )
        self.face_analyser.prepare(ctx_id=0, det_thresh=0.5)

        # Corse face swap
        self.inswapper = OnnxInSwapper("models/inswapper_128.onnx")
        self.fill_scale = 1.1

        # Upscale face
        self.scale_lock = threading.Lock()
        if upscale:
            self.onnx_upscaler = OnnxGfpGan("models/GFPGANv1.4.onnx")
        else:
            self.upscaler = None

    def stop(self):
        self.running = False

    def signal_handler(self, *argv, **kwargs):
        self.stop()
        self.print_stats()
        print("\nprogram exiting gracefully")
        sys.exit(0)

    def upscale(self, bgr_fake: Frame) -> Frame:
        if self.onnx_upscaler is not None:
            return self.onnx_upscaler.run(input=bgr_fake)
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
        actual_image = norm_crop(
            target_frame, target_face.kps, self.inswapper.input_size[0]
        )
        return self.inswapper.run(target=actual_image, source=source_face.latent)

    def fill_face(self, target_face: Face, img_mask: Frame, fill_scale=None):
        if fill_scale is None:
            fill_scale = self.fill_scale

        # Face KPS is 5x2 array of x,y coordinates
        # Left eye, right eye, nose, left mouth, right mouth
        M = estimate_norm(target_face.kps, img_mask.shape[0])
        target2crop_kps = cv2.transform(target_face.kps.reshape(1, -1, 2), M)[0]
        target2crop_landmark_2d_106 = cv2.transform(
            target_face.landmark_2d_106.reshape(1, -1, 2), M
        )[0]

        hull_points = cv2.convexHull(target2crop_landmark_2d_106)

        if fill_scale != 1.0:
            center = np.mean(hull_points, axis=0)
            translated_points = hull_points - center
            scaled_points = translated_points * fill_scale
            hull_points = scaled_points + center
            hull_points[hull_points >= img_mask.shape[0]] = img_mask.shape[0] - 1
            hull_points[hull_points < 0] = 0
            target2crop_kps = target2crop_kps - center
            target2crop_kps = target2crop_kps * fill_scale
            target2crop_kps = target2crop_kps + center
            target2crop_kps[target2crop_kps >= img_mask.shape[0]] = (
                img_mask.shape[0] - 1
            )
            target2crop_kps[target2crop_kps < 0] = 0

        center_eye = np.mean(target2crop_kps[0:2], axis=0)
        center_mouth = np.mean(target2crop_kps[3:5], axis=0)
        radius = int(0.7 * fill_scale * np.linalg.norm(center_eye - center_mouth))

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
        side = 512
        if bgr_fake.shape[0] != side:
            bgr_fake = cv2.resize(bgr_fake, (side, side))
        bgr_target = norm_crop(target_frame, target_face.kps, side)

        img_mask = np.full((side, side), 0, dtype=np.float32)
        self.fill_face(target_face, img_mask)
        blur = 2 * int(0.1 * side) + 1
        img_mask = cv2.GaussianBlur(img_mask, (blur, blur), 0)
        img_mask = np.reshape(img_mask, [side, side, 1])
        bgr_fake = img_mask * bgr_fake + (1 - img_mask) * bgr_target.astype(np.float32)
        self.warp(bgr_fake.astype(np.uint8), target_face, target_frame)

    def get_faces(self, img: Frame):
        faces = self.face_analyser.get(img)
        for face in faces:
            latent = face.normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self.inswapper.emap)
            latent /= np.linalg.norm(latent)
            face.latent = latent
        return faces

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


class RoughFaceSwap(FaceSwap):
    def __init__(self, upscale=True, profile=True):
        super().__init__(upscale=upscale, profile=profile)

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

    def fill_face(self, target_face: Face, img_mask: Frame, fill_scale=None):
        if fill_scale is None:
            fill_scale = self.fill_scale
        # Face KPS is 5x2 array of x,y coordinates
        # Left eye, right eye, nose, left mouth, right mouth

        M = estimate_norm(target_face.kps, img_mask.shape[0])
        target2crop_kps = cv2.transform(target_face.kps.reshape(1, -1, 2), M)[0]

        center = np.mean(target2crop_kps, axis=0)
        if fill_scale != 1.0:
            target2crop_kps = target2crop_kps - center
            target2crop_kps = target2crop_kps * fill_scale
            target2crop_kps = target2crop_kps + center
            target2crop_kps[target2crop_kps >= img_mask.shape[0]] = (
                img_mask.shape[0] - 1
            )
            target2crop_kps[target2crop_kps < 0] = 0

        center_eye = np.mean(target2crop_kps[0:2], axis=0)
        center_mouth = np.mean(target2crop_kps[3:5], axis=0)
        radius = int(0.5 * fill_scale * np.linalg.norm(center_eye - center_mouth))

        cv2.circle(
            img_mask,
            (int(center_eye[0]), int(center_eye[1])),
            (int(1.3 * radius)),
            (1, 1, 1),
            -1,
        )
        cv2.circle(
            img_mask,
            (int(center[0]), int(center[1])),
            int(1.6 * radius),
            (1, 1, 1),
            -1,
        )
        for point in target2crop_kps:
            cv2.circle(
                img_mask,
                (int(point[0]), int(point[1])),
                int(0.5 * radius),
                (1, 1, 1),
                -1,
            )
        cv2.circle(
            img_mask,
            (int(center_mouth[0]), int(center_mouth[1])),
            int(1.4 * radius),
            (1, 1, 1),
            -1,
        )


class CameraSwap(RoughFaceSwap):
    # https://github.com/dorssel/usbipd-win/wiki/WSL-support
    def __init__(self, camera_id: int, face_path: str, profile=False):
        super().__init__(upscale=False, profile=profile)
        self.camera_id = camera_id
        self.camera = cv2.VideoCapture(self.camera_id)
        self.resize = 2
        if sys.platform.startswith("linux"):
            cv2.setLogLevel(0)
            self.resize = 1
            self.camera.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )
        if not self.camera.isOpened():
            self.camera.release()
            raise Exception(f"Camera {camera_id} not found")
        self.source_face = self.get_face(face_path)
        self.input_data = (0, None, None)
        self.output_frame = None
        self.desired_fps = 25

    def read_camera(self):
        frame_count = 0
        while self.running:
            frame_count += 1
            tic = time.time()
            success, frame = self.camera.read()
            if success:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] // self.resize, frame.shape[0] // self.resize),
                )
                faces = self.get_faces(frame)
                self.input_data = (frame_count, frame, faces)
            toc = time.time()
            time.sleep(max(0, 1 / self.desired_fps - (toc - tic)))

    def process_frame(self):
        prev_frame_count = 0
        while self.running:
            tic = time.time()
            frame_count, frame, faces = self.input_data
            if frame is not None and frame_count != prev_frame_count:
                for target_face in faces:
                    bgr_fake = self.inswap(frame, target_face, self.source_face)
                    self.combine(frame, target_face, bgr_fake)
                prev_frame_count = frame_count
                self.output_frame = frame
            toc = time.time()
            time.sleep(max(0, 1 / self.desired_fps - (toc - tic)))

    def start(self):
        self.running = True
        threading.Thread(target=self.read_camera).start()
        threading.Thread(target=self.process_frame).start()

    def stop(self):
        super().stop()
        self.camera.release()
