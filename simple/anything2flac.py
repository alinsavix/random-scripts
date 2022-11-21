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


def conv2flac(source: Path, dest: Path, args: argparse.Namespace):
    fh, _tmpfile = tempfile.mkstemp(suffix=dest.suffix, prefix="remux_", dir=dest.parent)
    os.close(fh)

    tmpfile = Path(_tmpfile)
    if args.debug:
        print(f"DEBUG: tmpfile is {tmpfile}", file=sys.stderr)

    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-i", source, "-vn", "-f", "flac", "-compression_level", "5", "-y", tmpfile],
                       shell=False, stdin=subprocess.DEVNULL, check=True, timeout=args.timeout)
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: flac conversion timed out after {args.timeout} seconds", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: flac conversion failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except Exception:
        print("ERROR: unknown error during flac conversion, aborting", file=sys.stderr)
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

    print(f"\nDONE: completed flac conversion, output to {dest}")


valid_extensions = {"flac"}

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        prog='anything2flac.py',
        description='Convert an audio file to flac with ffmpeg',
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
        help="output file to write (defaults to <basename>.flac",
    )

    parser.add_argument(
        "--delete",
        "--no-delete",
        dest="delete",
        default=False,
        action=NegateAction,
        nargs=0,
        help="delete original file after conversion"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10 * 60,
        help="timeout for conversion operation",
    )

    parser.add_argument(
        "file",
        type=Path,
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        action=CheckFile(must_exist=True),
        help="file(s) to convert",
    )

    parsed_args = parser.parse_args(argv[1:])

    if parsed_args.out is None:
        parsed_args.out = parsed_args.file.with_suffix(".flac")

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv)
    # loglevel = "DEBUG" if args.debug else "INFO"
    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    # log = lg.getLogger()

    conv2flac(args.file, args.out, args)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
