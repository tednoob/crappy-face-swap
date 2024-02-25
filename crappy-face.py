#!/usr/bin/env python3

import os
import sys
import cv2
import argparse
import debugpy
import logging
from typing import Optional

from crap_code.operations import Operator
from crap_code.image import FaceSwap
from crap_code.util import exit_with_error, is_image, is_video, normalize_path
from crap_code.video import MediaDirector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A crappy face swapper")
    parser.add_argument(
        "target",
        type=str,
        help="The path to the target directory, image, or video",
    )
    parser.add_argument(
        "-f",
        "--face",
        type=str,
        default=None,
        help="The path/name of the source face image",
    )
    parser.add_argument(
        "-p", "--profile", action="store_true", default=False, help="Enable profiling"
    )
    parser.add_argument(
        "-u",
        "--upscale",
        action="store_true",
        default=None,
        help="Always upscale the image [default: Auto]",
    )
    parser.add_argument(
        "-n",
        "--no-upscale",
        action="store_true",
        default=None,
        help="Never upscale the image [default: Auto]",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Enable the debugger [default: False]",
    )

    args = parser.parse_args()
    if args.debug:
        debugpy.listen(5678)
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Waiting for debugger")
        debugpy.wait_for_client()

    if args.upscale and args.no_upscale:
        exit_with_error("Cannot specify both --upscale and --no-upscale")

    upscale = True  # Later auto
    if args.upscale:
        upscale = True
    elif args.no_upscale:
        upscale = False

    try:
        Operator(upscale, args.profile).process(args.face, args.target)
    except Exception as e:
        exit_with_error(str(e))
