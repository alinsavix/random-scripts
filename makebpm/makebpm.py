#!/usr/bin/env python3
import argparse
import json
import logging as lg
import shutil
# import re
import subprocess
import sys
import tempfile
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

# FIXME: figure out better when to use "cachedir" (the parent cache dir)
# and when to use "animdir" (the cache dir with the anim name appended)
def extract_anim(archive: Path, cachedir: Path) -> None:
    log = lg.getLogger()

    if not archive.exists():
        print(
            f"ERROR: can't extract animation from non-existant zip file '{archive}", file=sys.stderr)
        sys.exit(1)

    if cachedir.exists():
        print(
            f"ERROR: animation extraction directory '{cachedir}' already exists", file=sys.stderr)

    shutil.unpack_archive(archive, cachedir)

    log.debug(f"DEBUG: extracted archive '{archive}' to cache directory '{cachedir}'")


def cache_anim(animname: str, animdir: Path, cachedir: Path) -> None:
    log = lg.getLogger()

    # check to see if this animation has already been extracted to the
    # cache
    cached_path = cachedir / Path(animname)
    if cached_path.exists():
        log.debug(f"cached animation exists at {cached_path}, nothing to do")
        return

    # make sure we have a zip to extract
    archive_path = animdir / Path(animname).with_suffix(".zip")
    if not archive_path.exists():
        print(
            f"ERROR: no archive exists for animation {animname}, can't continue", file=sys.stderr)
        sys.exit(1)

    # make sure the cache directory exists
    if not cachedir.exists():
        log.debug(f"cache directory {cachedir} doesn't exist, creating")
        cachedir.mkdir(mode=0o700, parents=True)

    extract_anim(archive_path, cached_path)


# FIXME: might want to validate the config, too
AnimConfig = Dict[str, Union[str, int, float]]
def load_config(animname: str, animdir: Path) -> AnimConfig:
    configpath = animdir / Path(animname).with_suffix(".json")
    if not configpath.exists():
        print(f"ERROR: no configuration exists for animation {animname}", file=sys.stderr)
        sys.exit(1)

    try:
        with configpath.open("r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: couldn't load configuration for animation {animname}: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: couldn't open configuration for animation {animname}: {e}")
        sys.exit(1)

    return config


def gen_framelist(bpm: float, cycles: int, output_fps: float, config: AnimConfig) -> List[int]:
    # match_bpm = int(sys.argv[1]) or 198
    beats_per_cycle = int(config["beats_per_cycle"])  # restart every n beats

    anim_frames = int(config["framecount"])  # numbered starting at 1
    # video_len_s = 10
    video_len_s = 10  # temp, should use 'cycles' instead

    time_per_video_frame_s = 1.0 / output_fps

    one_cycle = (bpm / beats_per_cycle)   # how many times it needs to run per minute
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
    # f = 0
    frames: List[int] = []
    while t < video_len_s:
        x = t / time_per_anim_frame_s
        useframe = (int(x) % anim_frames) + 1

        # print(f"frame {f:4d} t = {t:0.5f}  anim frame {useframe:08d}")
        # print(f"file '{useframe:08d}.png'")
        # print(f"duration {time_per_video_frame_s}")
        frames.append(useframe)

        t += time_per_video_frame_s
        # f += 1

    return frames


def write_framelist(framelist_file: Path, animcache: Path, fps: float, framelist: List[int]) -> None:
    time_per_frame_s = 1.0 / fps
    video_len_s = time_per_frame_s * len(framelist)

    with framelist_file.open("w") as f:
        t = 0.0
        for frame in framelist:
            fn = animcache / Path(f"{frame:08d}.png")
            f.write(f"file '{fn}'\n")
            f.write(f"duration {time_per_frame_s}\n")
            # print(f"file '{useframe:08d}.png'")
            # print(f"duration {time_per_video_frame_s}")




def render_video(outfile: Path, animcache: Path, fps: float, framelist: List[int]):
    timeout = 30  # FIXME: make configurable?
    framelist_file = outfile.with_suffix(outfile.suffix + ".framelist")
    write_framelist(framelist_file, animcache, fps, framelist)

    # ffmpeg -f concat -safe 0 -i list.txt -r 60 -c:v h264_nvenc -cq:v 23 -vf "format=yuv420p" -y $BPM.mp4 && start $BPM.mp4
    # FIXME: this should encode as prores w/ alpha (at least optionally)
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-f", "concat", "-safe", "0",
                      "-i", str(framelist_file), "-r", str(fps), "-c:v", "libx264", "-qp", "23",
                      "-vf", "format=yuv420p", "-y", str(outfile)]
    try:
        subprocess.run(cmd, shell=False, stdin=subprocess.DEVNULL, check=True, timeout=timeout)
    except FileNotFoundError:
        print("ERROR: couldn't execute ffmpeg, please make sure it exists in your PATH")
        framelist_file.unlink()
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"ERROR: remux process timed out after {timeout} seconds", file=sys.stderr)
        framelist_file.unlink()
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: remux process failed with ffmpeg exit code {e.returncode}", file=sys.stderr)
        framelist_file.unlink()
        sys.exit(1)
    except Exception:
        print("ERROR: unknown error during remux process, aborting", file=sys.stderr)
        framelist_file.unlink()
        sys.exit(1)

    framelist_file.unlink()

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Generate an animation appropriate for a given bpm",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Enable debugging output",
    )

    parser.add_argument(
        "--animdir",
        type=Path,
        default=Path(__file__).parent / "anims",
        action=CheckFile(must_exist=True),
        help="directory in which to find animation data",
    )

    parser.add_argument(
        "--cachedir",
        type=Path,
        default=Path(__file__).parent / "cache",
        action=CheckFile(must_exist=True),
        help="directory in which to cache extracted animation data",
    )

    parser.add_argument(
        "-b",
        "--bpm",
        type=int,
        default=120,
        help="bpm to which animation should be matched",
    )

    parser.add_argument(
        "--cycles",
        "-c",
        type=int,
        default=10,
        help="number of animation cycles to render",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="fps of created output video",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default=None,
        action=CheckFile(extensions={'mov'}),
        help="output .mov file to generate (defaults to <animname>_<bpm>bpm.mov)",
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
        default="deerbutt",
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
    loglevel = "DEBUG" if args.debug else "INFO"
    LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    animname = args.animation
    config = load_config(args.animation, args.animdir)
    cache_anim(args.animation, args.animdir, args.cachedir)
    framelist = gen_framelist(bpm=args.bpm, cycles=args.cycles,
                              output_fps=args.fps, config=config)

    if args.outfile:
        outfile = args.outfile
    else:
        outfile = Path(f"{animname}_{args.bpm}bpm.mov")
    animcache = args.cachedir / animname
    render_video(outfile=outfile, animcache=animcache,
                 fps=args.fps, framelist=framelist)

    # extract_anim(Path("anims/deerbutt.zip"), Path("anims/deerbutt"))
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
