#!/usr/bin/env python3
import argparse
import errno
import os
# import logging as lg
import re
import subprocess
import sys
import time
import warnings
from enum import IntEnum
from pathlib import Path
from typing import (Any, Dict, Generator, Iterable, Iterator, List, Optional,
                    Sequence, Set, Text, Tuple, Type, Union, cast)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import animation
from matplotlib.artist import Artist  # yet more just for typing
from matplotlib.axes import Axes  # Just for typing
from matplotlib.figure import Figure  # also also just for typing
from matplotlib.lines import Line2D  # also just for typing

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    FFPROBE_BIN = os.path.join(sys._MEIPASS, "ffprobe")
    FFMPEG_BIN = os.path.join(sys._MEIPASS, "ffmpeg")
else:
    FFPROBE_BIN = "ffprobe"
    FFMPEG_BIN = "ffmpeg"
# ffmpeg -i thing.mp4 -af ebur128=peak=true -f null -
#
# output (note: actual output is not linewrapped):
# [Parsed_ebur128_0 @ 00000132f15e35c0] t: 763.8
# TARGET:-23 LUFS    M: -14.9 S: -15.4     I: -17.1 LUFS
# LRA: 17.2 LU     FTPK: -7.2 -6.7 dBFS  TPK: -4.2 -3.9 dBFS
#
# field numbers if just split on colon and/or whitespace:
# FIXME: consider doing this instead of regex matches
# [Parsed_ebur128_0 @ 00000132f15e35c0]        0 1 2
# t: 763.8                                     3 4             time
# TARGET:-23 LUFS                              5 6 7
# M: -14.9                                     8 9             momentary
# S: -15.4                                     10 11           short
# I: -17.1 LUFS                                12 13 14        integrated
# LRA: 17.2 LU                                 15 16 17        lra
# FTPK: -7.2 -6.7 dBFS                         18 19 20 21     tpk_l tpk_r    (frame true peak)
# TPK: -4.2 -3.9 dBFS                          22 23 24 25

# also outputs a summary:
# [Parsed_ebur128_0 @ 000001fab00a3500] Summary:
#
#   Integrated loudness:
#     I:         -22.7 LUFS
#     Threshold: -33.7 LUFS
#
#   Loudness range:
#     LRA:        14.1 LU
#     Threshold: -43.7 LUFS
#     LRA low:   -33.6 LUFS
#     LRA high:  -19.5 LUFS
#
#   True peak:
#     Peak:       -4.2 dBFS


# regex for parsing the ffmpeg output
re_ffmpeg = re.compile(r"""
\[Parsed_ebur128_0 \s @ \s (0x)?[0-9a-f]+ \]
\s+
t: \s (?P<time> [\d.]+)
\s+
TARGET: \s* (?P<target> [.0-9-]+) \s LUFS
\s+
M: \s* (?P<momentary> [.0-9-]+)
\s+
S: \s* (?P<short> [.0-9-]+)
\s+
I: \s* (?P<integrated> [.0-9-]+) \s LUFS
\s+
LRA: \s* (?P<lra> [.0-9-]+) \s LU
\s+
FTPK: \s* (?P<ftpk_l> [.0-9-]+) \s+ (?P<ftpk_r> [.0-9-]+) \s dBFS
\s+
TPK: \s* (?P<tpk_l> [.0-9-]+) \s+ (?P<tpk_r> [.0-9-]+) \s dBFS
""", re.VERBOSE | re.IGNORECASE)

# regex for parsing the ffmpeg summary output
re_ffmpeg_summary = re.compile(r"""
\s+
(?P<field> [A-Z ]+) :
\s+
(?P<value> [.0-9-]+)
\s+
""", re.VERBOSE | re.IGNORECASE)


# Theoretically you can name your fields in a multi-dimensional array, but I
# wasn't smart enough to figure out how to make that work, so we'll just number
# them instead (and use an enum to sanely identify them)
class Fields(IntEnum):
    TIME = 0
    TARGET = 1
    MOMENTARY = 2
    SHORT = 3
    INTEGRATED = 4
    LRA = 5
    FTPK = 6
    TPK = 7


# Convert seconds to [HH:]MM:SS
def sec_to_hms(secs: float) -> str:
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    secs = int(secs)

    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:d}:{secs:02d}"


# Convert seconds to e.g. 1h15m6s
def sec_to_str(secs: float) -> str:
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    secs = int(secs)

    if hours:
        return f"{hours:d}h{minutes:d}m{secs:d}s"
    elif minutes:
        return f"{minutes:d}m{secs:d}s"
    else:
        return f"{secs:d}s"


#
# class: Loudness - retrieve a series of loudness values from a media file
#
class LoudnessException(Exception):
    pass

class LoudnessNoFfmpegException(LoudnessException):
    pass

class LoudnessDurationException(LoudnessException):
    pass

class LoudnessNoFfprobeException(LoudnessDurationException):
    pass

class LoudnessFfprobeError(LoudnessDurationException):
    pass

class LoudnessFfprobeNoResult(LoudnessDurationException):
    pass


# FIXME: We should make this whole thing a context manager
class Loudness:
    _args: argparse.Namespace
    _file: Path
    _proc: Optional[subprocess.Popen[str]]
    _duration: Optional[float] = None

    def __init__(self, file: Path, args: argparse.Namespace) -> None:
        if not file.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(file))

        self._file = file
        self._args = args
        self._proc = None


    # make sure children get reaped when we get destroyed.
    def __del__(self) -> None:
        self.close()


    # use ffprobe to get the duration of the media
    def duration(self) -> float:
        if self._duration is not None:
            return self._duration


        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            output = subprocess.check_output(
                [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(self._file)],
                shell=False,
                stdin=subprocess.DEVNULL,
                universal_newlines=True,
                timeout=5,
                startupinfo=startupinfo,
            )

        except FileNotFoundError:
            raise LoudnessNoFfprobeException(
                "ERROR: Unable to execute ffprobe, verify it exists in your PATH"
            )

        except subprocess.CalledProcessError as e:
            raise LoudnessFfprobeError(
                f"ERROR: ffprobe returned exit code {e.returncode}, duration not available"
            )

        except subprocess.TimeoutExpired:
            raise LoudnessFfprobeError(
                "ERROR: ffprobe timed out before returning a media duration"
            )

        # This is kinda lazy, but the output of ffprobe should be exactly one
        # line, as a floating point number, so just try to use it directly
        try:
            self._duration = float(output)
            return self._duration
        except ValueError:
            raise LoudnessFfprobeNoResult(
                "ERROR: ffprobe ran successfully, but did not output a parseable result"
            )


    def _ffmpeg_open_loudness(self) -> subprocess.Popen[str]:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            proc = subprocess.Popen(
                [FFMPEG_BIN, "-i", str(self._file), "-af",
                 "ebur128=peak=true", "-f", "null", "-"],
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                startupinfo=startupinfo,
            )

        except FileNotFoundError:
            raise LoudnessNoFfmpegException(
                "ERROR: Unable to execute ffmpeg, verify it exists in your PATH")

        return proc


    def _ffmpeg_close(self, proc: subprocess.Popen[str]) -> Optional[int]:
        if not proc:
            raise ValueError("I/O operation when no file is open")

        retval = None
        try:
            retval = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()

        if not retval:
            try:
                retval = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        if not retval:
            try:
                retval = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                warnings.warn("Unable to terminate ffmpeg subprocess",
                              category=UserWarning)

        if retval is not None and retval != 0:
            warnings.warn(f"ffmpeg exited with return code {retval}", category=UserWarning)

        return retval


    # FIXME: do we really just want to support a single value iterable per
    # loudness object?
    def values(self) -> Generator[List[float], None, None]:
        if self._proc is not None:
            self.close()

        self._proc = self._ffmpeg_open_loudness()
        assert self._proc.stdout is not None

        while self._proc.stdout.readable():
            line = self._proc.stdout.readline()

            if not line:
                break

            if self._args.debug:
                print(f"DEBUG: ffmpeg: {line}", end="", file=sys.stderr)

            # if we hit the summary block, we're mostly done, so bail
            if " Summary:" in line:
                break

            m = re_ffmpeg.match(line.strip())
            if not m:
                # print(f"DEBUG: SKIPPING LINE: {line}")
                continue

            # Until we get a full 'short' chunk of audio, many of our stats are
            # going to be super wonky, so we're just going to completely skip that
            # data, since we won't have any pathological cases where it actually
            # matters
            if float(m.group("time")) <= 3.1:
                continue

            # Make a single datapoint, and append it to our collection. Pity that
            # you can't specify data types as part of the regex itself so that
            # you don't have to convert each thing here...
            one = [
                float(m.group("time")),
                float(m.group("target")),
                float(m.group("momentary")),
                float(m.group("short")),
                float(m.group("integrated")),
                float(m.group("lra")),
                max(float(m.group("ftpk_l")), float(m.group("ftpk_r"))),
                max(float(m.group("tpk_l")), float(m.group("tpk_r")))
            ]

            yield one


    # Call this when we run out of data, or we hit the summary block. It
    # expects proc.stdout to be positioned before the data portion of the
    # summary block in the ffmpeg output.
    def summary(self) -> Dict[str, float]:
        assert self._proc is not None
        assert self._proc.stdout is not None

        summary: Dict[str, float] = {}
        while self._proc.stdout.readable():
            line = self._proc.stdout.readline()

            if not line:
                break

            if self._args.debug:
                print(f"DEBUG: ffmpeg: {line}", end="", file=sys.stderr)

            m = re_ffmpeg_summary.match(line)
            if not m:
                # print(f"DEBUG: SKIPPING LINE: {line}")
                continue

            summary[m.group("field")] = float(m.group("value"))

        return summary


    def close(self) -> None:
        if self._proc is not None:
            self._ffmpeg_close(self._proc)
            self._proc = None


#
# Main code
#
X_PADDING = 10.0
def prep_graph(fig: Figure, ax1: Axes, ax2: Optional[Axes], duration: float, title: str, args: argparse.Namespace) -> Tuple[Dict[int, Line2D], Dict[int, Line2D]]:
    ax1_lines: Dict[int, Line2D] = {}
    ax2_lines: Dict[int, Line2D] = {}

    plt.suptitle(title, size=16.0)

    # ax1 = axs[0][0]
    # ax1.set_xlabel("seconds")

    # This sometimes breaks the graphs and sometimes doesn't and I have exactly
    # zero clue how to get it to work reliably. :(
    # happy = Path(__file__).parent / "ongHappy.png"
    # if happy.exists():
    #     img = plt.imread(str(happy))
    #     ax2.imshow(img, alpha=0.3,
    #                extent=[duration - 20.0 + X_PADDING, duration + X_PADDING, 0.0, 20.0])

    if duration > (15 * 60):
        xtick_interval = 60
    else:
        xtick_interval = 30

    ax1.set_xlim(xmin=0, xmax=duration + X_PADDING)
    ax1.set_xticks(list(range(0, int(duration), xtick_interval)))
    ax1.set_xticklabels([sec_to_hms(x) for x in ax1.get_xticks()])

    ax1.set_ylim(ymin=-60, ymax=1)
    ax1.set_ylabel("Loudness (LUFS)")
    ax1.set_yticks(list(range(-60, 6, 6)))
    ax1.set_yticks(list(range(-57, 4, 6)), minor=True)
    ax1.grid(True, linestyle="dotted")

    ax1.format_coord = lambda x, y: f"x={sec_to_hms(x)}, y={y:0.2f}"

    # FIXME: should this be here, or in the actual plotting?
    ax1.axhline(y=args.target_lufs, color='orange',
                label="_Integrated Target", linewidth=3, alpha=1.0)
    ax1.annotate(" " + args.target_desc, (0, args.target_lufs + 0.1), fontsize=14)

    if args.momentary:
        ax1_lines[Fields.MOMENTARY] = ax1.plot(
            [], [], label="Momentary", color='m', linewidth=0.5)[0]

    if args.short:
        ax1_lines[Fields.SHORT] = ax1.plot([], [], label="Short", color='b', linewidth=0.75)[0]

    if args.integrated:
        ax1_lines[Fields.INTEGRATED] = ax1.plot(
            [], [], label="Integrated", color='b', linewidth=2)[0]

    if args.clipping:
        ax1_lines[Fields.FTPK] = ax1.plot(
            [], [], label=f"Clipping ({args.clip_at}dB)", color='r', linewidth=3)[0]

    ax1.legend(loc='lower left', shadow=True, fontsize='large')


    if args.lra:
        assert ax2 is not None

        ax2.xaxis.tick_top()
        ax2.grid(True, linestyle="dotted")
        ax2.set_xlim(xmin=0, xmax=duration + X_PADDING)
        ax2.set_ylim(ymin=0.0, ymax=20.5)
        ax2.set_yticks(list(range(0, 21, 5)))

        # ax2.set_ylim(ymin=0, ymax=22)
        # ax2.set_yticks(list(range(0, 21, 4)))
        ax2.set_ylabel("Loudness Range (LU)")
        ax2.format_coord = lambda x, y: f"x={sec_to_hms(x)}, y={y:0.2f}"

        ax2_lines[Fields.LRA] = ax2.plot(
            [], [], label="Loudness Range", color='g', linewidth=1.5)[0]
    else:
        ax2 = None

    return ax1_lines, ax2_lines


class LUFSLoadAnimation:
    args: argparse.Namespace
    values_per_tick: int
    title: str

    _loudness: Loudness  # so we can get the summary at the end
    _duration: float
    _values: Iterator[List[float]]
    _first: bool  # are we on the very first frame?
    _end: bool  # We're at the end and should add extra data

    _fig: Figure
    _axs: List[Axes]
    _lines: List[Dict[int, Line2D]]
    _data: List[List[float]]

    def __init__(self, loudness: Loudness, args: argparse.Namespace) -> None:
        self.args = args
        self.values_per_tick = 50
        self.title = ""

        self._loudness = loudness


    def graph_setup(self) -> None:
        if self.args.lra:
            numrows = 2
            gridspec = {'height_ratios': [5, 1]}
        else:
            numrows = 1
            gridspec = {}

        fig, axs = plt.subplots(numrows, 1, sharex=True, gridspec_kw=gridspec, squeeze=False,
                                figsize=(19.20, 10.80), dpi=72, linewidth=0.1, tight_layout=True)
        self._fig = fig
        self._axs = [axs[0][0], axs[1][0]] if self.args.lra else [axs[0][0]]


    def graph_finalize(self, xloc: float) -> List[Artist]:
        changed: List[Artist] = []
        summary = self._loudness.summary()

        if summary["LRA low"] and summary["LRA high"]:
            lra = self._axs[0].axhspan(summary["LRA low"],
                                       summary["LRA high"], color='gainsboro', alpha=0.3)
            changed.append(lra)

            range = summary["LRA high"] - summary["LRA low"]
            rangestring = f"Final Loudness Range: {range:.1f} ({summary['LRA low']} to {summary['LRA high']})"
            lra_text = self._axs[0].annotate(rangestring, (xloc + (X_PADDING * 0.75), summary["LRA low"]),
                                             horizontalalignment='right', fontsize=14)
            changed.append(lra_text)

        if summary["I"]:
            dot = self._axs[0].plot([xloc], summary["I"], label="_Integrated Final",
                                    color='b', marker='o', markersize=7)[0]
            changed.append(dot)

            anno = self._axs[0].annotate(
                str(summary["I"]), (xloc, summary["I"] + 0.5), fontsize=18)
            changed.append(anno)

        if self.args.lra and summary["LRA"]:
            dot = self._axs[1].plot([xloc], summary["LRA"], label="_LRA Final",
                        color='g', marker='o', markersize=7)[0]
            changed.append(dot)

            anno = self._axs[1].annotate(
                str(summary["LRA"]), (xloc, summary["LRA"] + 0.7), fontsize=16)
            changed.append(anno)

        return changed


    def anim_fig_init(self) -> Iterable[Artist]:
        ax1 = self._axs[0]
        ax2 = self._axs[1] if self.args.lra else None
        ax1_lines, ax2_lines = prep_graph(
            self._fig, ax1, ax2, self._duration, self.title, self.args)

        self._lines = [ax1_lines, ax2_lines]

        # I think I'm returning the right things here
        artists: List[Artist] = []
        artists += cast(List[Artist], self._axs)  # FIXME: may not need the axes?
        artists += cast(List[Artist], self._lines[0].values())
        artists += cast(List[Artist], self._lines[1].values())

        return artists


    def anim_frame(self) -> Iterator[List[List[float]]]:
        while True:
            accum: List[List[float]] = []

            for _ in range(self.values_per_tick):
                one = next(self._values, None)

                if one is None:
                    break

                accum.append(one)

            if len(accum) > 0:
                yield accum
            else:
                # If we're out of data, set the end flag and yield an empty
                # chunk of data so that the update function can add in the
                # bits that can only be added at the end
                if not self._end:
                    self._end = True
                    yield []
                else:
                    return


    def anim_update(self, frame: List[List[float]]) -> Optional[Iterable[Artist]]:
        self._data += frame

        # FIXME: is there a better/more efficient way to manage all this?
        lind: npt.NDArray[Any] = np.array(self._data)  # type: ignore

        T = lind[:, Fields.TIME]
        momentary = lind[:, Fields.MOMENTARY]
        short = lind[:, Fields.SHORT]
        integrated = lind[:, Fields.INTEGRATED]
        lra = lind[:, Fields.LRA]
        ftpk = lind[:, Fields.FTPK]

        if self.args.momentary:
            self._lines[0][Fields.MOMENTARY].set_data(
                T, np.ma.masked_where(momentary <= -120.7, momentary))

        if self.args.short:
            self._lines[0][Fields.SHORT].set_data(T, np.ma.masked_where(short <= -120.7, short))

        if self.args.integrated:
            self._lines[0][Fields.INTEGRATED].set_data(
                T, np.ma.masked_where(integrated <= -70.0, integrated))

        if self.args.clipping:
            self._lines[0][Fields.FTPK].set_data(
                T, np.ma.masked_where(ftpk < self.args.clip_at, ftpk.clip(0.0, 0.0)))

        if self.args.lra:
            # Sometimes the beginning and end of a track have exceedingly large
            # loudness ranges, which throw off the scale for the rest of the graph,
            # so we'll remove the first 15 seconds and last 6 seconds before
            # plotting. This is absolutely not optimal at all, and better ideas
            # are vigorously accepted.
            # self._lines[1][Fields.LRA].set_data(T[150:-60], lra[150:-60].clip(None, 20.0))
            self._lines[1][Fields.LRA].set_data(T[150:], lra[150:].clip(None, 20.0))
            self._axs[1].relim()
            self._axs[1].autoscale_view(scalex=False, scaley=True)

        # Anything we need to do on the first frame only
        if self._first:
            self._first = False

        final_changes: List[Artist] = []
        if self._end:
            final_changes = self.graph_finalize(T[-1])

        # Not 100% sure why the casting is needed here, tbh, since Line2D
        # should still be an Artist? I'm probably missing something stupid.
        changed: List[Artist] = []
        changed += cast(List[Artist], list(self._lines[0].values()))
        changed += cast(List[Artist], list(self._lines[1].values()))
        changed += final_changes

        return changed


    def animate(self) -> None:
        self._duration = self._loudness.duration()
        self._values = self._loudness.values()
        self._end = False
        self._first = True
        self._data = []

        self.graph_setup()
        _anim = animation.FuncAnimation(
            fig=self._fig,
            func=self.anim_update,
            frames=self.anim_frame,
            init_func=self.anim_fig_init,
            save_count=0,
            interval=1,
            repeat=False,
            blit=False,  # why doesn't this completely work?
            cache_frame_data=False,
        )

        plt.show()


    def oneshot(self) -> None:
        self._duration = self._loudness.duration()
        self._values = self._loudness.values()
        self._end = True
        self._first = False
        self._data = []

        self.graph_setup()
        self.anim_fig_init()

        # read everything
        while True:
            one = next(self._values, None)

            if one is None:
                break

            self._data.append(one)

        # the data is already in _data, no reason to feed it again here.
        self.anim_update([])
        plt.draw()


    def save_final(self, outfile: Path):
        self._fig.savefig(str(outfile))
        print(f"wrote {outfile}", file=sys.stderr)


def gen_loudness(file: Path, title: str, args: argparse.Namespace) -> None:
    loudness = Loudness(file, args)

    anim = LUFSLoadAnimation(loudness, args=args)
    anim.values_per_tick = 50
    anim.title = title

    if args.interactive:
        anim.animate()
    else:
        anim.oneshot()

    if args.write_graph:
        anim.save_final(args.outfile)

    return


lufs_targets = {
    "amazon": (-11.0, "Amazon Integrated Target"),
    "apple": (-16.0, "Apple Music Integrated Target"),
    "beatport": (-8.0, "Beatport/DJ Stores Integrated Target"),
    "spotify": (-14.0, "Spotify Integrated Target"),
    "youtube": (-14.0, "YouTube Integrated Target"),
}

# returns the target string unchanged (it'll be used later) or raise a type error
def check_lufs_target(target: str) -> str:
    try:
        val = float(target)
        if val > 0.0 or val < -60.0:
            raise argparse.ArgumentTypeError("Custom LUFS target must be between -60.0 and 0.0")

        return target
    except ValueError:
        if target.lower() not in lufs_targets:
            raise argparse.ArgumentTypeError(
                f"LUFS target must be FLOAT or one of: {', '.join(lufs_targets.keys())}")

    # else, we're good
    return target

# sets args.target_lufs, args.target_desc
def set_lufs_target(args: argparse.Namespace, target) -> None:
    try:
        val = float(target)
        args.target_lufs = val
        args.target_desc = f"Custom Integrated Target ({val:.1f} LUFS)"
    except ValueError:
        val, desc = lufs_targets[target.lower()]
        args.target_lufs = val
        args.target_desc = f"{desc} ({int(val)} LUFS)"

    return


def CheckFile(extensions: Optional[Set[str]] = None, must_exist: bool = False) -> Type[argparse.Action]:
    class Act(argparse.Action):
        def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                     values: Any, option_string: Optional[Text] = "") -> None:
            if not isinstance(values, Path):
                parser.error(
                    f"CheckFile called but argument is {type(values)} and not a pathlib path? (bug)")

            if extensions:
                ext = values.suffix[1:]
                if ext not in extensions:
                    parser.error(f"file '{values}' doesn't end with one of {extensions}")

            if must_exist:
                if not values.exists():
                    parser.error(f"file '{values}' does not exist")

            setattr(namespace, self.dest, values)

    return Act

# FIXME: I think there's some way to expose an 'nargs' value so that we don't
# have to explicity specify "nargs=0" when using this?
class NegateAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 _values: Union[Text, Sequence[Any], None], option_string: Optional[Text] = "") -> None:
        if option_string is None:
            parser.error("NegateAction can only be used with non-positional arguments")

        setattr(namespace, self.dest, option_string[2:4] != 'no')

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Generate a BS.1770-based loudness graph for a file w/ audio",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Enable debugging output",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        type=Path,
        default=None,
        action=CheckFile(extensions={'png'}),
        help="filename to output generated graph to (defaults based on input name)",
    )

    parser.add_argument(
        "--noninteractive",
        default=True,
        action='store_false',
        dest="interactive",
        help="don't show interactive graph (implies -w)",
    )

    parser.add_argument(
        "-w",
        "--write",
        default=False,
        action='store_true',
        dest="write_graph",
        help="write graph to file on disk",
    )

    parser.add_argument(
        "--target",
        default="youtube",
        type=check_lufs_target,
        help="target LUFS, or one of: amazon, apple, beatport, spotify, youtube (default: youtube)"
    )

    parser.add_argument(
        "--target_lufs",
        default=0.0,
        type=float,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--target_desc",
        default="",
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--clip-at",
        default=-1.0,
        type=float,
        help="dB value above which is considered to be clipping (default: -1.0)",
    )

    parser.add_argument(
        "--momentary",
        "--no-momentary",
        default=False,
        action=NegateAction,
        nargs=0,
        help="generate plots for momentary (400ms window) loudness (default: no)",
    )

    parser.add_argument(
        "--short",
        "--no-short",
        default=True,
        action=NegateAction,
        nargs=0,
        help="generate plots for short (3s window) loudness (default: yes)",
    )

    parser.add_argument(
        "--integrated",
        "--no-integrated",
        default=True,
        action=NegateAction,
        nargs=0,
        help="generate plots for integrated (full-length) loudness (default: yes)",
    )

    parser.add_argument(
        "--lra",
        "--no-lra",
        default=True,
        action=NegateAction,
        nargs=0,
        help="generate plot for loudness range (default: yes)",
    )

    parser.add_argument(
        "--clipping",
        "--no-clipping",
        default=True,
        action=NegateAction,
        nargs=1,
        help="show where true peak is higher than -1.0dbFS (default: yes)",
    )

    # positional arguments
    parser.add_argument(
        "file",
        type=Path,
        action=CheckFile(must_exist=True),
        help="file for which to analyze loudness",
    )

    parsed_args = parser.parse_args(argv)
    set_lufs_target(parsed_args, parsed_args.target)

    # make sure we enable writing the graph if the user specifies a filename
    if parsed_args.outfile:
        parsed_args.write_graph = True

    if not parsed_args.interactive:
        parsed_args.write_graph = True

    if parsed_args.write_graph and parsed_args.outfile is None:
        parsed_args.outfile = parsed_args.file.with_suffix(".loudness.png")
    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    # loglevel = "DEBUG" if args.debug else "INFO"
    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)
    # log = lg.getLogger()

    gen_loudness(args.file, f"Loudness Analysis: {args.file.name}", args)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


# common loudness ranges:
#
# Classical music is most dynamic, so you can expect LRA readings of 9 or more.
# Country or jazz will have less dynamics; a reading of 6 to 8 is typical. Rock
# and EDM often hit around 5 to 6, and hip-hop, 5 or less. But again, LRA
# readings vary all over the place within any specific genre, as well as within
# an album. On my most recent album, LRA readings varied from 4 for a slamming,
# full-tilt track up to 10 for a longer, more nuanced song.
#
#
#

# Loudness ranges:
#
# A low LRA low reading indicates a track with narrow dynamic range and a high
# LRA indicates a track with wide dynamic range.
#
# So it is not unusual for a techno track to have an LRA of 3LU and a classical
# music track can have as much as 15 LU or more.
#
# Depending on genre an LRA of around 9 - 10 LU is considered a good indication
# of a healthy dynamic range.


# Peak to Loudness Ratio (PLR)  [not implemented]
#
# A dynamic range meter may give a measurement called a Peak to Loudness
# Ratio (PLR).The PLR is the difference between the highest True Peak and the
# integrated loudness. A high PLR indicates wide dynamic range. ("crest factor")
#
# There is also a measure called the PSR (peak to short term ratio) which does
# the same measurement, but based off a three second window and short term
# loudness. PSR measures the dynamics of the last three seconds and a high
# number indicates well preserved transients, a small PSR indicates strong
# limiting or compression.


# Some suggested limits for various online services:
#
# Youtube:
#   short:  -10 / -18
#   tpk: -1.0
#   target: -14.0
#
# amazon music:
#   short:  -10 / -18
#   tpk: -2.0
#   target: -14.0
#
# spotify:
#   short: -10 / -18
#   tpk: -1
#   target: -14.0
#
# apple music:
#   short: -12 / -20
#   tpk: -1
#   target: -16
#
# soundcloud:
#   short: -10 / -18
#   tpk: -1
#   target: -14
#
# tidal:
#   short: -10 / -18
#   tpk: -1
#   target: -14
#
# ATSC:
#   use LM1 / 10s short window
#   short:  -16 / -32
#   tpk: -2
#   target: -24
#
# •EBU uses foreground audio as the loudness anchor.
# •LM1 measures and averages loudness across the whole program.


# EBU TECH 3343 says:
# Momentary - Loudness (M) -- time window: 400ms
# Short Term - Loudness (S) -- time window: 3s
# Integrated - Loudness (I) -- time window: all
