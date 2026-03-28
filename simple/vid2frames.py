#!/usr/bin/env python3
import argparse
import logging as lg
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from tdvutil.argparse import CheckFile, NegateAction


def vid2frames(source: Path, destdir: Path, args: argparse.Namespace):
    destdir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-hwaccel", "auto", "-i", source,
                "-f", "image2", "-qscale:v", "2",
                "-y", f"{destdir}/%06d.jpg",
            ],
            shell=False, stdin=subprocess.DEVNULL, check=True, timeout=args.timeout
        )
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(
            f"ERROR: frame extraction process timed out after {args.timeout} seconds", file=sys.stderr)
        # tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: frame extraction process failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
        # tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except KeyboardInterrupt:
        # tmpfile.unlink(missing_ok=True)
        raise
    except Exception as e:
        print(f"ERROR: unknown error during frame extraction process, aborting", file=sys.stderr)
        # tmpfile.unlink(missing_ok=True)
        raise

    # seems like it ran ok, rename the temp file
    # if args.debug:
    #     print(f"DEBUG: replacing {dest} with {tmpfile}", file=sys.stderr)
    # tmpfile.replace(dest)

    # if args.delete and source != dest:
    #     if args.debug:
    #         print(f"DEBUG: deleting source file {source}", file=sys.stderr)
    #     source.unlink()

    print(f"\nDONE: completed frame extraction to {destdir}")


valid_extensions = {"mkv", "mp4", "mov", "m4v", "mpg", "avi", "webm"}

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        prog='vid2frames.py',
        description='Extract frames from a video file',
    )

    parser.add_argument(
        "--debug",
        action='store_const',
        const=True,
        default=False,
        help="Enable debugging output",
    )

    parser.add_argument(
        "-o",
        "--out",
        default=None,
        type=Path,
        action=CheckFile(extensions=valid_extensions, must_exist=False),
        help="output directory to write frames to (defaults to <basename>_frames)",
    )

    # parser.add_argument(
    #     "-f",
    #     "--fps",
    #     default=60.0,
    #     type=float,
    #     help="output framerate to use",
    # )

    # parser.add_argument(
    #     "-q",
    #     "--qp",
    #     default=18,
    #     type=int,
    #     help="QP value to use for CQP",
    # )

    # parser.add_argument(
    #     "--delete",
    #     "--no-delete",
    #     dest="delete",
    #     default=False,
    #     action=NegateAction,
    #     nargs=0,
    #     help="delete original file after remuxing"
    # )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 60,
        help="timeout for frame extraction operation",
    )

    parser.add_argument(
        "files",
        type=Path,
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        # action=CheckFile(must_exist=True),
        nargs='+',
        help="file(s) to extract",
        metavar="FILE",
    )

    parsed_args = parser.parse_args(argv[1:])

    # if we're doing a batch, we must use the default filenames
    if len(parsed_args.files) > 1 and parsed_args.out is not None:
        parser.error("can't use '-o' with multiple input files")

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv)
    # loglevel = "DEBUG" if args.debug else "INFO"
    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    # log = lg.getLogger()

    for file in args.files:
        if args.out is None:
            outfile = file.parent / (file.stem + "_frames")
        else:
            outfile = args.out

        try:
            if args.debug:
                print(f"DEBUG: executing for in: {file}, out: {outfile}", file=sys.stderr)
            vid2frames(file, outfile, args)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
