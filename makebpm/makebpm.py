#!/usr/bin/env python3
import argparse
import json
import logging as lg
import shutil
# import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
# from enum import IntEnum
from pathlib import Path
from typing import (Any, Dict, List, Optional, Sequence, Set, Text, Tuple,
                    Type, Union, cast)

from dataclasses_json import DataClassJsonMixin
from tdvutil.argparse import CheckFile


# A couple of useful datatypes
@dataclass
class AnimConfig(DataClassJsonMixin):
    name: str
    description: str
    framecount: int
    beats_per_cycle: int
    has_alpha: bool

@dataclass
class EncoderInfo:
    args_all: List[str]
    args_noalpha: List[str]
    args_alpha: Optional[List[str]]
    valid_containers: Set[str]


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




def load_config(animname: str, animdir: Path) -> AnimConfig:
    configpath = animdir / Path(animname).with_suffix(".json")
    if not configpath.exists():
        print(f"ERROR: no configuration exists for animation {animname}", file=sys.stderr)
        sys.exit(1)

    try:
        config_str = configpath.read_text()
        config = AnimConfig.from_json(config_str)
    except json.JSONDecodeError as e:
        print(f"ERROR: couldn't load configuration for animation {animname}: {e}")
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: couldn't open configuration for animation {animname}: {e}")
        sys.exit(1)

    return config


def gen_framelist(bpm: float, cycles: int, output_fps: float, config: AnimConfig) -> List[int]:
    beats_per_cycle = int(config.beats_per_cycle)  # restart every n beats

    anim_frames = int(config.framecount)  # frames numbered starting at 1
    # video_len_s = 10  # temp, should use 'cycles' instead

    time_per_video_frame_s = 1.0 / output_fps

    one_cycle = (bpm / beats_per_cycle)   # how many times it needs to run per minute
    one_cycle_time_s = 60 / one_cycle

    # hokay, so, if we have one_cycle_time_s, and we have anim_frames in an
    # animation, we can figure out how long a given frame must exist to get
    # that length of animation.
    time_per_anim_frame_s = one_cycle_time_s / anim_frames

    # And figure out long the final video will be for a given number of cycles
    video_len_s = one_cycle_time_s * cycles

    print(f"time per video frame: {time_per_video_frame_s:.6f}s")
    print(f"time per animation cycle: {one_cycle_time_s:.6f}s")
    print(f"time per anim frame: {time_per_anim_frame_s:.6f}s")
    print(f"total video time: {video_len_s:6f}s")

    # And now, output our frame numbers for any given combination of things
    t = 0.0
    frames: List[int] = []
    while t <= video_len_s:
        x = t / time_per_anim_frame_s
        useframe = (int(x) % anim_frames) + 1

        # print(f"frame {f:4d} t = {t:0.5f}  anim frame {useframe:08d}")
        # print(f"file '{useframe:08d}.png'")
        # print(f"duration {time_per_video_frame_s}")
        frames.append(useframe)

        t += time_per_video_frame_s

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


# ffmpeg -hide_banner -hwaccels
# encoders = {
#     "default",
#     "nvenc",
#     "videotoolbox",
# }


codecs: Dict[str, EncoderInfo] = {
    "h264": EncoderInfo(
        args_all=[
            "-c:v", "libx264",
            "-qp", "23",
            "-vf", "format=yuv420p"
        ],
        args_alpha=None,
        args_noalpha=[],
        valid_containers={"mp4", "mov", "mkv"}
    ),

    "h264_nvenc": EncoderInfo(
        args_all=[
            "-c:v", "h264_nvenc",
            "-qp", "23",
            "-vf", "format=yuv420p"
        ],
        args_alpha=None,
        args_noalpha=[],
        valid_containers={"mp4", "mov", "mkv"}
    ),

    "prores": EncoderInfo(
        args_all=[
            "-c:v", "prores_ks",
            "-quant_mat", "auto",
            "-qscale:v", "9", "-vendor", "apl0",
        ],
        args_alpha=[
            "-profile:v", "4",  # 4444
            "-pix_fmt", "yuva444p10le",
            "-alpha_bits", "8",
        ],
        args_noalpha=[
            "-profile:v", "2",  # standard
            "-pix_fmt", "yuv422p10le"
        ],
        valid_containers={"mov", }
    ),

    "webm": EncoderInfo(
        args_all=[
            "-c:v", "libvpx", "-auto-alt-ref", "0", "-crf", "30",
            "-b:v", "10M",
        ],
        args_alpha=[
            "-vf", "format=rgba",
        ],
        args_noalpha=[
            "-vf", "format=rgb",
        ],
        valid_containers={"webm", }
    ),
}


def get_encode_codec(codec: str, extension: str, has_alpha: bool) -> List[str]:
    if codec not in codecs:
        raise KeyError(f"Invalid codec '{codec}' requested")

    codec_info = codecs[codec]
    extension = extension.lstrip(".")
    if extension not in codec_info.valid_containers:
        raise ValueError(f"Incompatible container '{extension}' rquested for codec '{codec}'")

    codec_str = codec_info.args_all
    if not has_alpha:
        codec_str.extend(codec_info.args_noalpha)
    else:
        if codec_info.args_alpha is None:
            raise ValueError(
                f"video requires alpha transparency, which codec '{codec}' does not support")
        else:
            codec_str.extend(codec_info.args_alpha)

    return codec_str


def render_video(outfile: Path, codec: str, has_alpha: bool, animcache: Path, fps: float, framelist: List[int]):
    log = lg.getLogger()
    timeout = 180  # FIXME: make configurable?
    framelist_file = outfile.with_suffix(outfile.suffix + ".framelist")
    write_framelist(framelist_file, animcache, fps, framelist)


    codec_str = get_encode_codec(codec, outfile.suffix, has_alpha)

    if log.isEnabledFor(lg.DEBUG):
        log_level = "info"
    else:
        log_level = "warning"

    cmd: List[str] = [
        "ffmpeg", "-hide_banner", "-loglevel", log_level, "-stats",
        "-f", "concat", "-safe", "0",
        "-i", str(framelist_file),
        "-r", str(fps),
        "-an",
        *codec_str,
        "-y",
        str(outfile)
    ]

    log.debug(f"Attempting to encode with command: {cmd}")

    print("Rendering...")
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
        "-b",
        "--bpm",
        type=float,
        default=120,
        help="bpm to which animation should be matched (default: 120)",
    )

    parser.add_argument(
        "--cycles",
        "-c",
        type=int,
        default=10,
        help="number of animation cycles to render (default: 10)",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="fps of created output video (default: 60)",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default=None,
        action=CheckFile(extensions={'mp4', 'mov', 'webm'}),
        help="output video file to generate (default: <animname>_<bpm>bpm.mov)",
    )

    parser.add_argument(
        "--codec",
        type=str,
        default="prores",
        choices=["h264", "h264_nvenc", "prores", "webm"],
        help="encoder/codec to use for created video (default: prores)",
    )

    parser.add_argument(
        "--animdir",
        type=Path,
        default=Path(__file__).parent / "anims",
        action=CheckFile(must_exist=True),
        help="directory in which to find animation data (default: 'anims/')",
    )

    parser.add_argument(
        "--cachedir",
        type=Path,
        default=Path(__file__).parent / "cache",
        action=CheckFile(must_exist=True),
        help="directory in which to cache extracted animation data (default: 'cache/')",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Enable debugging output",
    )

    # positional arguments
    parser.add_argument(
        "animation",
        type=str,
        # default="deerbutt",
        # nargs="?",
        help="animation name that should be rendered",
    )

    parsed_args = parser.parse_args(argv)

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])
    loglevel = "DEBUG" if args.debug else "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] %(levelname)s: %(message)s"
    lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    animname = args.animation
    animcache = args.cachedir / animname

    if args.outfile:
        outfile = args.outfile
    else:
        if str(args.bpm).endswith(".0"):
            bpm_str = str(int(args.bpm))
        else:
            bpm_str = str(args.bpm)
        outfile = Path(f"{animname}_{bpm_str}bpm.mov")

    cache_anim(args.animation, args.animdir, args.cachedir)
    config = load_config(args.animation, args.animdir)
    framelist = gen_framelist(bpm=args.bpm, cycles=args.cycles,
                              output_fps=args.fps, config=config)

    render_video(outfile=outfile, codec=args.codec, has_alpha=config.has_alpha,
                 animcache=animcache, fps=args.fps, framelist=framelist)

    print(f"SUCCESS: rendered bpm-matched video to '{outfile}'")
    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
