import os
import cv2
import sys
import logging
from typing import List, Optional, Tuple

from crap_code.image import RoughFaceSwap, FaceSwap
from crap_code.video import MediaDirector
from crap_code.util import is_image, is_video, normalize_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Operator:
    def __init__(self, upscale: bool, profile: bool, rough=False):
        if rough:
            self.swapper = RoughFaceSwap(upscale=upscale, profile=profile)
        else:
            self.swapper = FaceSwap(upscale=upscale, profile=profile)
        self.output_dir = "output"
        self.faces_dir = "faces"

    def get_face_path(self, face_name: str) -> Optional[str]:
        if os.path.exists(face_name):
            return face_name
        face_path = os.path.join(self.faces_dir, face_name)
        if os.path.exists(face_path):
            return face_path
        face_path += ".jpg"
        if os.path.exists(face_path):
            return face_path
        raise ValueError(f"Face {face_name} not found")

    def get_faces(
        self, source_path: Optional[str], target_path: str
    ) -> List[Tuple[str, str]]:
        target_name = os.path.join(target_path).split(os.path.sep)[-1]
        target_type = target_name.split(".")[-1]
        target_name = target_name.replace(f".{target_type}", "")

        if source_path is not None:
            source_path = self.get_face_path(source_path)

            if not is_image(source_path):
                raise ValueError(f"{source_path} is not an image")

            face_name = source_path.split(os.path.sep)[-1].split(".")[0]
            output_path = os.path.join(
                self.output_dir, f"{target_name}_{face_name}.{target_type}"
            )
            return [(source_path, output_path)]

        targets: List[Tuple[str, str]] = []
        for current_source_file in os.listdir(self.faces_dir):
            if is_image(current_source_file):
                face_name = current_source_file.split(os.path.sep)[-1].split(".")[0]
                face_path = os.path.join(self.faces_dir, current_source_file)
                output_path = os.path.join(
                    self.output_dir, f"{target_name}_{face_name}.{target_type}"
                )
                targets.append((face_path, output_path))

        return targets

    def process_dir(self, source_path, input_path):
        if not os.path.isdir(input_path):
            raise ValueError(f"{input_path} is not a directory")

        source_path = self.get_face_path(source_path)
        if source_path is None:
            raise ValueError(f"{source_path} is required when processing video")

        face_name = source_path.split(os.path.sep)[-1].split(".")[0]
        source_face = self.swapper.get_face(source_path)
        input_dir = input_path
        output_dir = os.path.join(self.output_dir, face_name)
        for root, _, files in os.walk(input_dir):
            rel_path = os.path.relpath(root, input_dir)
            output_sub_dir = os.path.join(output_dir, rel_path)
            os.makedirs(output_sub_dir, exist_ok=True)
            for file in files:
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_sub_dir, file)
                if os.path.exists(output_file):
                    continue
                if is_image(file):
                    logger.info(f"Processing {input_file}")
                    try:
                        frame = cv2.imread(input_file)
                        self.swapper.swap_face(source_face, frame)
                        cv2.imwrite(output_file, frame)
                    except Exception:
                        logger.exception(f"Error processing {input_file}")

    def process_image(self, source_path: Optional[str], target_path: str):
        # mogrify -format jpg *.HEIC
        if not is_image(target_path):
            raise ValueError(f"{target_path} is not an image")

        target_frame = cv2.imread(f"{target_path}")
        for source_path, output_path in self.get_faces(source_path, target_path):
            current_target_frame = target_frame.copy()
            source_face = self.swapper.get_face(source_path)
            self.swapper.swap_face(source_face, current_target_frame)
            cv2.imwrite(output_path, current_target_frame)

    def process_video(self, face_path: Optional[str], in_filename: str):
        # ffmpeg -i input.mp4 -ss 2 -t 10 -c copy output.mp4  # Cut between 0:00:02 and 0:00:12
        # ffmpeg -i input.mp4 -ss '0:10:00' -t "0:02:00" -c copy output.mp4  # Cut between 0:10:0 and 0:12:00
        if not is_video(in_filename):
            raise ValueError(f"{in_filename} is not a video")
        source_path = self.get_face_path(face_path)
        if source_path is None:
            raise ValueError(f"{face_path} is required when processing video")
        file = os.path.join(in_filename).split(os.path.sep)[-1]
        face_name = source_path.split(os.path.sep)[-1].split(".")[0]
        out_filename = os.path.join(self.output_dir, f"{face_name}_{file}")
        director = MediaDirector(self.swapper, source_path, in_filename, out_filename)
        director.run()

    def process(self, face_path, target_path):
        target_path = normalize_path(target_path)
        if os.path.isdir(target_path):
            self.process_dir(face_path, target_path)
        elif is_image(target_path):
            self.process_image(face_path, target_path)
        elif is_video(target_path):
            self.process_video(face_path, target_path)
        else:
            logger.error(f"Unsupported target: {target_path}")
        self.swapper.print_stats()
