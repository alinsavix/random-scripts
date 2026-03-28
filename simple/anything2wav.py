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


def conv2wav(source: Path, dest: Path, args: argparse.Namespace):
    fh, _tmpfile = tempfile.mkstemp(suffix=dest.suffix, prefix="wavconv_", dir=dest.parent)
    os.close(fh)

    tmpfile = Path(_tmpfile)
    if args.debug:
        print(f"DEBUG: tmpfile is {tmpfile}", file=sys.stderr)

    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-i", source, "-vn", "-c:a", "pcm_s24le", "-f", "wav", "-compression_level", "5", "-y", tmpfile],
                       shell=False, stdin=subprocess.DEVNULL, check=True, timeout=args.timeout)
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: wav conversion timed out after {args.timeout} seconds", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: wav conversion failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
        tmpfile.unlink(missing_ok=True)
        sys.exit(1)
    except Exception:
        print("ERROR: unknown error during wav conversion, aborting", file=sys.stderr)
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

    print(f"\nDONE: completed wav conversion {source} -> {dest}")



valid_extensions = {"wav"}

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        prog='anything2wav.py',
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
        help="output file to write (defaults to <basename>.wav",
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
        "files",
        type=Path,
        # action=CheckFile(extensions=valid_extensions, must_exist=True),
        # action=CheckFile(must_exist=True),
        nargs='+',
        help="file(s) to convert",
        metavar="FILE",
    )

    parsed_args = parser.parse_args(argv[1:])

    # if parsed_args.out is None:
    #     parsed_args.out = parsed_args.file.with_suffix(".wav")

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv)
    # loglevel = "DEBUG" if args.debug else "INFO"
    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    # log = lg.getLogger()

    if len(args.files) > 1 and args.out is not None:
        print(f"ERROR: can't use '-o' with multiple input files", file=sys.stderr)
        sys.exit(1)

    for file in args.files:
        if args.out is None:
            outfile = file.with_suffix(".wav")
        else:
            outfile = args.out

        conv2wav(file, outfile, args)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
