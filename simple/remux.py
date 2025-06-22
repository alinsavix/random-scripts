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


def remux(source: Path, dest: Path, args: argparse.Namespace) -> None:
    fh, _tmpfile = tempfile.mkstemp(suffix=dest.suffix, prefix="remux_", dir=dest.parent)
    os.close(fh)

    tmpfile = Path(_tmpfile)
    if args.debug:
        print(f"DEBUG: tmpfile is {tmpfile}", file=sys.stderr)

    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-i", source, "-c", "copy", "-map", "0", "-y", tmpfile],
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


valid_extensions = {"mkv", "mp4", "mov", "m4v", "mpg", "avi", "flv", "webm"}

def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='remux.py',
        description='Remux video files with ffmpeg',
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
        help="output directory for remuxed files (defaults to same directory as input files)",
    )

    parser.add_argument(
        "--delete",
        "--no-delete",
        dest="delete",
        default=True,
        action=NegateAction,
        nargs=0,
        help="delete original files after remuxing"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10 * 60,
        help="timeout for remux operation",
    )

    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        help="video file(s) to remux",
    )

    parsed_args = parser.parse_args(argv[1:])

    if parsed_args.out is not None:
        # If -o is specified, ensure it's a directory
        if not parsed_args.out.exists():
            parsed_args.out.mkdir(parents=True)
        elif not parsed_args.out.is_dir():
            parser.error("Output path must be a directory when processing multiple files")

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv)

    for source_file in args.files:
        if args.out is not None:
            # If output directory is specified, place file there with .mp4 extension
            out_file = args.out / source_file.with_suffix(".mp4").name
        else:
            # Otherwise, place in same directory as source with .mp4 extension
            out_file = source_file.with_suffix(".mp4")

        if args.debug:
            print(f"\nDEBUG: Processing {source_file} -> {out_file}")

        try:
            remux(source_file, out_file, args)
        except Exception as e:
            print(f"ERROR: Failed to process {source_file}: {str(e)}", file=sys.stderr)
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
