#!/usr/bin/env python3
import argparse
# import logging as lg
# import re
import subprocess
import sys
# from enum import IntEnum
from pathlib import Path
from typing import (Any, Dict, List, Optional, Sequence, Set, Text, Tuple,
                    Type, Union)

from tdvutil.argparse import CheckFile

#
# things we need in config file:
#   name
#   filename pattern (maybe)
#   number of frames
#   beats per cycle
#
# encode: ffmpeg -i %d.png -vcodec png z.mov
# decode: ffmpeg -i z.mov -f image2 export2\%d.png
# BPM=384 ; ./makebpm.py $BPM > list.txt && ffmpeg -f concat -safe 0 -i list.txt -r 60 -c:v h264_nvenc -cq:v 23 -vf "format=yuv420p" -y $BPM.mp4 && start $BPM.mp4

def extract_anim(zipfile: Path, outdir: Path) -> None:
    pass


def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Generate an animation appropriate for a given bpm",
        allow_abbrev=True,
    )

    # parser.add_argument(
    #     "--debug",
    #     action='store_true',
    #     default=False,
    #     help="Enable debugging output",
    # )

    parser.add_argument(
        "--animdir",
        type=Path,
        default=Path(__file__).parent / "anims",
        action=CheckFile(must_exist=True),
        help="directory in which to find animation data",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default=None,
        action=CheckFile(extensions={'mov'}),
        help="output .mov file to generate (defaults to <animname>_<bpm>.mov)",
    )

    parser.add_argument(
        "-b",
        "--bpm",
        type=int,
        default=120,
        help="bpm to which animation should be matched",
    )

    # parser.add_argument(
    #     "--noninteractive",
    #     default=True,
    #     action='store_false',
    #     dest="interactive",
    #     help="don't show interactive graph (implies -w)",
    # )

    # parser.add_argument(
    #     "--short",
    #     "--no-short",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    #     help="generate plots for short (3s window) loudness (default: yes)",
    # )

    # positional arguments
    parser.add_argument(
        "animation",
        type=str,
        nargs="?",
        help="animation name that should be rendered",
    )

    parsed_args = parser.parse_args(argv)

    # make sure we enable writing the graph if the user specifies a filename
    # if parsed_args.outfile:
    #     parsed_args.write_graph = True

    # if not parsed_args.interactive:
    #     parsed_args.write_graph = True

    # if parsed_args.write_graph and parsed_args.outfile is None:
    #     parsed_args.outfile = parsed_args.file.with_suffix(".loudness.png")
    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])
    print(f"args: {args}")

    # match_bpm = int(sys.argv[1]) or 198
    match_bpm = args.bpm
    match_bpm_every = 2   # restart every n beats

    anim_frames = 128    # numbered starting at 1
    video_fps = 60
    # video_len_s = 10
    video_len_s = 10

    time_per_video_frame_s = 1.0 / video_fps

    one_cycle = (match_bpm / match_bpm_every)   # how many times it needs to run per minute
    one_cycle_time_s = 60 / one_cycle


    # hokay, so, if we have one_cycle_time_s, and we have anim_frames in an
    # animation, we can figure out how long a given frame must exist to get
    # that length of animation.
    time_per_anim_frame_s = one_cycle_time_s / anim_frames

    print(f"time per video frame (s): {time_per_video_frame_s}", file=sys.stderr)
    print(f"time per animation cycle (s): {one_cycle_time_s}", file=sys.stderr)
    print(f"time per anim frame (s): {time_per_anim_frame_s}", file=sys.stderr)

    # sys.exit(0)
    # And now, output our frame numbers for any given combination of things
    t = 0.0
    f = 0
    while t < video_len_s:
        x = t / time_per_anim_frame_s
        useframe = (int(x) % anim_frames) + 1

        # print(f"frame {f:4d} t = {t:0.5f}  anim frame {useframe:08d}")
        print(f"file '{useframe:08d}.png'")
        print(f"duration {time_per_video_frame_s}")

        t += time_per_video_frame_s
        f += 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
