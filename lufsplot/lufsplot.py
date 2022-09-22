#!/usr/bin/env python3
import argparse
# import logging as lg
import re
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import (Any, Dict, List, Optional, Sequence, Set, Text, Tuple,
                    Type, Union)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

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


# Can't quiiiiite type the multi-dimensional numpy array correctly, alas
def gen_loudness(file: Path, args: argparse.Namespace) -> Tuple[npt.NDArray[Any], Dict[str, float]]:
    # Create python lists now, then convert to a numpy array at the end, for
    # much better performance.
    li: List[List[float]] = []

    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-i", str(file), "-af", "ebur128=peak=true", "-f", "null", "-"],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except FileNotFoundError:
        print("ERROR: Unable to execute ffmpeg, verify it exists in your PATH", file=sys.stderr)
        sys.exit(1)

    assert proc.stdout is not None

    # read and parse the results, a line at a time
    # FIXME: if we use split() instead of regexes here, how does that change performance?
    while proc.stdout.readable():
        line = proc.stdout.readline()

        if not line:
            break

        if args.debug:
            print(f"DEBUG: ffmpeg: {line}", end="", file=sys.stderr)

        # if we hit the summary block, we're mostly done, so bail
        if " Summary:" in line:
            break

        # print(line)

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
        li.append(one)

    # If we got here, we either ran out of data, or we hit the summary
    # block, so do our best to read whatever summary info if it exists:
    summary: Dict[str, float] = {}
    while proc.stdout.readable():
        line = proc.stdout.readline()

        if not line:
            break

        if args.debug:
            print(f"DEBUG: ffmpeg: {line}", end="", file=sys.stderr)

        m = re_ffmpeg_summary.match(line)
        if not m:
            # print(f"DEBUG: SKIPPING LINE: {line}")
            continue

        summary[m.group("field")] = float(m.group("value"))

    proc.wait(timeout=5)
    if proc.returncode != 0:
        print(
            f"\nERROR: ffmpeg returned error code {proc.returncode}, can't continue", file=sys.stderr)
        sys.exit(1)

    # We've gathered all the data, put it in a useful form
    lind: npt.NDArray[Any] = np.array(li)  # type: ignore

    return lind, summary


# Most of the matplotlib stuff have shitty typing support :(
def gen_graph(lind: npt.NDArray[Any], summary: Dict[str, float], title: str, args: argparse.Namespace) -> None:
    T = lind[:, Fields.TIME]
    momentary = lind[:, Fields.MOMENTARY]
    short = lind[:, Fields.SHORT]
    integrated = lind[:, Fields.INTEGRATED]
    lra = lind[:, Fields.LRA]
    ftpk = lind[:, Fields.FTPK]

    if args.lra:
        numrows = 2
        gridspec = {'height_ratios': [5, 1]}
    else:
        numrows = 1
        gridspec = {}

    _, axs = plt.subplots(numrows, 1, sharex=True, gridspec_kw=gridspec, squeeze=False,
                          figsize=(19.20, 10.80), dpi=72, linewidth=0.1, tight_layout=True)
    plt.suptitle(title, size=16.0)

    ax1 = axs[0][0]
    # ax1.set_xlabel("seconds")
    ax1.set_xticks(list(range(0, int(np.max(T)), 30)))
    ax1.set_xticklabels([sec_to_hms(x) for x in ax1.get_xticks()])

    ax1.set_ylim(ymin=-60, ymax=6)
    ax1.set_ylabel("Loudness (LUFS)")
    ax1.set_yticks(list(range(-60, 7, 6)))
    ax1.set_yticks(list(range(-57, 4, 6)), minor=True)
    ax1.grid(True, linestyle="dotted")

    ax1.axhline(y=-14.0, color='orange',
                label="_Youtube Integrated Target", linewidth=3, alpha=1.0)
    ax1.annotate("Youtube Integrated Target (-14.0 LUFS)", (0, -13.9), fontsize=14)


    if args.momentary:
        ax1.plot(T, np.ma.masked_where(momentary <= -120.7, momentary),
                 label="Momentary", color='m', linewidth=0.5)

    if args.short:
        ax1.plot(T, np.ma.masked_where(short <= -120.7, short),
                 label="Short", color='b', linewidth=0.75)

    if args.integrated:
        ax1.plot(T, np.ma.masked_where(integrated <= -70.0, integrated),
                 label="Integrated", color='b', linewidth=2)

    if args.clipping:
        ax1.plot(T, np.ma.masked_where(ftpk < -1.0, ftpk),
                 label="Clipping", color='r', linewidth=3)

    ax1.legend(loc='upper left', shadow=True, fontsize='large')

    if summary["LRA low"] and summary["LRA high"]:
        ax1.axhspan(summary["LRA low"], summary["LRA high"], color='g', alpha=.1)

    if summary["I"]:
        ax1.plot(T[-1:], summary["I"], label="_Integrated Final",
                 color='b', marker='o', markersize=7)
        ax1.annotate(str(summary["I"]), (T[-1], summary["I"] + 0.5), fontsize=18)

    if args.lra:
        ax2 = axs[1][0]

        # ax2 = ax1.twinx()
        ax2.xaxis.tick_top()
        ax2.grid(True, linestyle="dotted")
        # ax2.set_ylim(ymin=0, ymax=22)
        # ax2.set_yticks(list(range(0, 21, 4)))
        ax2.set_ylabel("Loudness Range (LU)")

        # Sometimes the beginning and end of a track have exceedingly large
        # loudness ranges, which throw off the scale for the rest of the graph,
        # so we'll remove the first 15 seconds and last 6 seconds before
        # plotting. This is absolutely not optimal at all, and better ideas
        # are vigorously accepted.
        # FIXME: should this have a fixed scale instead?
        ax2.plot(T[150:-60], lra[150:-60], label="Loudness Range", color='g', linewidth=1.5)
        # ax2.legend(loc='upper right', shadow=True, fontsize='large')

    if args.write_graph:
        plt.savefig(args.outfile)

    if args.interactive:
        plt.show()

    # print(summary)

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
                    # option_string = '({})'.format(option_string) if option_string else ''
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

        # assert option_string is not None
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
        nargs=0,
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

    lind, summary = gen_loudness(args.file, args)
    gen_graph(lind, summary, f"Loudness Analysis: {args.file.name}", args)

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
