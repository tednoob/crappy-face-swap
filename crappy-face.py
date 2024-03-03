#!/usr/bin/env python3
import sys
import argparse
import debugpy
import logging

from crap_code.operations import Operator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def exit_with_error(msg: str):
    logger.error(f"Error: {msg}")
    sys.exit(1)


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
    parser.add_argument(
        "-r",
        "--rough",
        action="store_true",
        default=False,
        help="Dodgy face detection in favor of speed [default: False]",
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

    Operator(upscale, args.profile, rough=args.rough).process(args.face, args.target)
