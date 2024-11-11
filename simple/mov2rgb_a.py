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

def log(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.stderr.flush()


# Create a temp file safely, then return the name of the file. The caller is
# responsible for deleting the file.
def mkouttmp(dest: Path) -> Path:
    fh, _tmpfile = tempfile.mkstemp(suffix=dest.suffix, prefix="split_tmp_", dir=dest.parent)
    os.close(fh)
    return Path(_tmpfile)


# FIXME: proper error handling (and returning error code or exception)
#
# FIXME: check for existing destination files and handle appropriately
def alphasplit(args: argparse.Namespace, source: Path, dest_rgb: Path, dest_alpha: Path):
    tmp_rgb = mkouttmp(dest_rgb)
    tmp_alpha = mkouttmp(dest_alpha)

    if args.debug:
        print(f"DEBUG: rgb tempfile is {tmp_rgb}", file=sys.stderr)
        print(f"DEBUG: alpha tempfile is {tmp_alpha}", file=sys.stderr)

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-hwaccel", "auto",
        "-i", str(source),
        "-filter_complex", "[0]alphaextract[alpha]",

        # rgb output
        # FIXME: what should color_range actually be?
        "-map", "0:v", "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "uhq",
        "-rc", "constqp", "-qp", str(args.qp), "-profile:v", "rext", "-color_range", "2",
        "-pix_fmt", "yuv444p16le",

        "-c:a", "copy",
        "-y", str(tmp_rgb),

        # alpha output
        "-map", "[alpha]", "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "uhq",
        "-rc", "constqp", "-qp", str(args.qp), "-profile:v", "rext", "-color_range", "2",
        "-pix_fmt", "yuv444p16le",

        "-an",
        "-y", str(tmp_alpha),
    ]

    try:
        subprocess.run(ffmpeg_cmd, shell=False, stdin=subprocess.DEVNULL,
                       check=True, timeout=args.timeout)
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        tmp_rgb.unlink(missing_ok=True)
        tmp_alpha.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: split process timed out after {args.timeout} seconds", file=sys.stderr)
        tmp_rgb.unlink(missing_ok=True)
        tmp_alpha.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: split process failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
        tmp_rgb.unlink(missing_ok=True)
        tmp_alpha.unlink(missing_ok=True)
        sys.exit(1)
    except Exception:
        print("ERROR: unknown error during split process, aborting", file=sys.stderr)
        tmp_rgb.unlink(missing_ok=True)
        tmp_alpha.unlink(missing_ok=True)
        sys.exit(1)

    # seems like it ran ok, rename the temp file
    if args.debug:
        print(f"DEBUG: renaming {tmp_rgb} to {dest_rgb}", file=sys.stderr)
        print(f"DEBUG: renaming {tmp_alpha} to {dest_alpha}", file=sys.stderr)
    tmp_rgb.rename(dest_rgb)
    tmp_alpha.rename(dest_alpha)

    # if args.delete and source != dest:
    #     if args.debug:
    #         print(f"DEBUG: deleting source file {source}", file=sys.stderr)
    #     source.unlink()

    print(f"\nDONE: completed split to {dest_rgb} and {dest_alpha}")


valid_extensions = {"mov", }

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mov2rgb+a.py",
        description="Split a file w/ alpha into separate RGB and Alpha files",
    )

    parser.add_argument(
        "--debug",
        action='store_const',
        const=True,
        default=False,
        help="Enable debugging output",
    )

    # parser.add_argument(
    #     "-o",
    #     "--out",
    #     default=None,
    #     type=Path,
    #     action=CheckFile(extensions=valid_extensions, must_exist=False),
    #     help="output file to write (defaults to <basename>.mp4",
    # )

    # parser.add_argument(
    #     "--delete",
    #     "--no-delete",
    #     dest="delete",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    #     help="delete original file after remuxing"
    # )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10 * 60,
        help="timeout for split operation",
    )

    parser.add_argument(
        "--qp",
        type=int,
        default=5,
        help="ffmpeg qp value (for constqp rate control)",
    )

    parser.add_argument(
        "files",
        metavar="FILE",
        type=Path,
        nargs='+',
        # FIXME: Make CheckFile work with multiple files
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        help="video file(s) to remux",
    )

    parsed_args = parser.parse_args()

    return parsed_args


def main() -> int:
    args = parse_arguments()
    # loglevel = "DEBUG" if args.debug else "INFO"
    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    # log = lg.getLogger()

    for inputfile in args.files:
        dest_rgb = inputfile.parent / f"{inputfile.stem}_RGB.mp4"
        dest_alpha = inputfile.parent / f"{inputfile.stem}_ALPHA.mp4"
        alphasplit(args, inputfile, dest_rgb, dest_alpha)

    # FIXME: check for errors and return appropriately
    return 0


if __name__ == "__main__":
    sys.exit(main())
