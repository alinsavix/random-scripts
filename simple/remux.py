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


def remux(source: Path, dest: Path, args: argparse.Namespace):
    fh, _tmpfile = tempfile.mkstemp(suffix=dest.suffix, prefix="remux_", dir=dest.parent)
    os.close(fh)

    tmpfile = Path(_tmpfile)
    if args.debug:
        print(f"DEBUG: tmpfile is {tmpfile}", file=sys.stderr)

    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-i", source, "-c", "copy", "-y", tmpfile],
                       shell=False, stdin=subprocess.DEVNULL, check=True, timeout=args.timeout)
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: remux process timed out after {args.timeout} seconds", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: remux process failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except Exception:
        print("ERROR: unknown error during remux process, aborting", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)

    # seems like it ran ok, rename the temp file
    if args.debug:
        print(f"DEBUG: replacing {dest} with {tmpfile}", file=sys.stderr)
    tmpfile.replace(dest)

    if args.delete and source != dest:
        if args.debug:
            print(f"DEBUG: deleting source file {source}", file=sys.stderr)
        source.unlink()

    print(f"\nDONE: completed remux to {dest}")


valid_extensions = {"mkv", "mp4", "mov", "m4v", "mpg", "avi", "webm"}

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        prog='remux.py',
        description='Remux a video file with ffmpeg',
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
        help="output file to write (defaults to <basename>.mp4",
    )

    parser.add_argument(
        "--delete",
        "--no-delete",
        dest="delete",
        default=True,
        action=NegateAction,
        nargs=0,
        help="delete original file after remuxing"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10 * 60,
        help="timeout for remux operation",
    )

    parser.add_argument(
        "file",
        type=Path,
        action=CheckFile(extensions=valid_extensions, must_exist=True),
        help="video file(s) to remux",
    )

    parsed_args = parser.parse_args(argv[1:])

    if parsed_args.out is None:
        parsed_args.out = parsed_args.file.with_suffix(".mp4")

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv)
    # loglevel = "DEBUG" if args.debug else "INFO"
    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    # log = lg.getLogger()

    remux(args.file, args.out, args)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
