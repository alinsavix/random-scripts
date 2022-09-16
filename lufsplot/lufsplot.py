#!/usr/bin/env python3
import argparse
import dataclasses
import logging as lg
import re
import subprocess
import sys
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy
import numpy as np
import numpy.typing as npt
from tdvutil.argparse import CheckFile, NegateAction

# ffmpeg -i thing.mp4 -af ebur128=peak=true -ar 4800 -f null -
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


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description='Generate a BS.1770-based loudness graph for a file w/ audio'
    )

    # parser.add_argument(
    #     "--debug",
    #     action='store_const',
    #     const=True,
    #     default=False,
    #     # help="Enable debugging",
    # )

    # parser.add_argument(
    #     "-o",
    #     "--out",
    #     type=Path,
    #     default=None,
    #     action=CheckFile(extensions={'png'}),
    #     help="filename to output generated graph to",
    # )

    # parser.add_argument(
    #     "--noninteractive",
    #     default=True,
    #     action='store_false',
    #     dest="interactive",
    #     help="don't show interactive graph",
    # )

    # positional arguments
    parser.add_argument(
        "file",
        type=Path,
        action=CheckFile(must_exist=True),
        help="file for which to analyze loudness",
    )

    parsed_args = parser.parse_args(argv)
    print(parsed_args)

    return parsed_args


def main(argv: List[str]) -> int:
    # args = parse_arguments(argv)
    #
    # loglevel = "DEBUG" if args.debug else "INFO"

    args = parse_arguments(argv[1:])

    # loglevel = "INFO"
    # LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s] (%(name)s) %(message)s"
    # lg.basicConfig(level=loglevel, format=LOG_FORMAT)

    # log = lg.getLogger()

    proc = subprocess.Popen(
        ["ffmpeg", "-i", str(args.file), "-af", "ebur128=peak=true",
            "-ar", "4800", "-f", "null", "-"],
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    assert proc.stdout is not None


    class Fields(IntEnum):
        TIME = 0
        TARGET = 1
        MOMENTARY = 2
        SHORT = 3
        INTEGRATED = 4
        LRA = 5
        FTPK = 6
        TPK = 7


    # loudness: List[LoudnessInfo] = []
    # plt.figure(figsize=(14, 5))
    # plt.ion()
    # hl, = plt.plot([], [])
    # plt.show()
    # li = LoudnessInfo()
    li: List[List[float]] = []

    while proc.stdout.readable():
        line = proc.stdout.readline()
        if not line:
            break

        if " Summary:" in line:
            # print("starting summary")
            break

        # print(line)

        m = re_ffmpeg.match(line.strip())
        if not m:
            # print(f"DEBUG: SKIPPING LINE: {line}")
            continue



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
    summary = {}
    while proc.stdout.readable():
        line = proc.stdout.readline()

        if not line:
            break

        m = re_ffmpeg_summary.match(line)
        if not m:
            # print(f"DEBUG: SKIPPING LINE: {line}")
            continue

        summary[m.group("field")] = float(m.group("value"))


    # We've gathered all the data, put it in a useful form
    lind = np.array(li)
    T = lind[:, Fields.TIME]
    momentary = lind[:, Fields.MOMENTARY]
    short = lind[:, Fields.SHORT]
    integrated = lind[:, Fields.INTEGRATED]
    lra = lind[:, Fields.LRA]
    ftpk = lind[:, Fields.FTPK]

    # plt.figure(figsize=(14, 8))
    fig, ax1 = plt.subplots(figsize=(19.20, 10.80), dpi=72,
                            linewidth=0.1, tight_layout=True)
    ax2 = ax1.twinx()

    ax1.set_ylim(ymin=-60, ymax=6)
    ax2.set_ylim(ymin=0, ymax=31)

    ax1.set_xlabel("seconds")
    ax1.set_xticks(list(range(0, int(np.max(T)), 30)))

    ax1.set_ylabel("Loudness (LUFS)")
    ax1.set_yticks(list(range(-60, 7, 6)))
    ax1.set_yticks(list(range(-57, 4, 6)), minor=True)
    ax1.grid(True, linestyle="dotted")

    ax1.axhline(y=-14.0, color='orange',
                label="_Youtube Integrated Limit", linewidth=3, alpha=1.0)
    ax1.annotate("Youtube Integrated Limit (-14.0 LUFS)", (0, -13.9), fontsize=14)

    ax2.set_ylabel("Loudness Range (LU)")
    ax2.set_yticks(list(range(0, 31, 2)))


    ax1.plot(T, np.ma.masked_where(momentary <= -120.7, momentary),
             label="Momentary", color='r', linewidth=0.5)
    ax1.plot(T, np.ma.masked_where(short <= -120.7, short),
             label="Short", color='b', linewidth=1)
    ax1.plot(T, np.ma.masked_where(integrated <= -70.0, integrated),
             label="Integrated", color='b', linewidth=2)
    ax1.plot(T, np.ma.masked_where(ftpk < -1.0, ftpk), label="Clipping", color='r', linewidth=3)



    ax2.plot(T, lra, label="Loudness Range", color='g', linewidth=1.5)

    ax1.legend(loc='upper left', shadow=True, fontsize='large')
    ax2.legend(loc='upper right', shadow=True, fontsize='large')

    if summary["LRA low"] and summary["LRA high"]:
        ax1.axhspan(summary["LRA low"], summary["LRA high"], color='g', alpha=.1)

    # fig.align_labels()

    plt.show()

    return 0

    # y_vals_masked = np.ma.masked_where(np.asfarray(li.momentary) < -25, li.momentary)
    # y_vals_masked = np.ma.masked_where(M < -25, M)

    # plt.plot(li.time, y_vals_masked)
    # plt.plot(T, y_vals_masked, label="Momentary", color='r', linewidth=1)
    # plt.plot(T, lind[:, IDX_SHORT], label="Short", color='b', linewidth=1)
    # plt.plot(T, lind[:, IDX_INTEGRATED], label="Integrated", color='b', linewidth=2)
    # plt.show()
    # print(lind.shape)
    # print(np.obj2sctype(lind))


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

# Values to look out for
#
# A low LRA low reading indicates a track with narrow dynamic range and a high
# LRA indicates a track with wide dynamic range.
#
# So it is not unusual for a techno track to have an LRA of 3 LU (Loudness
# range is measured in LU = Loudness Units) and a classical music track can
# have as much as 15 LU or more.
#
# Depending on genre an LRA of around 9 - 10 LU is considered a good indication
# of a healthy dynamic range.
#
# The LRA is not typically useful for music shorter than 30 seconds.
#
# Dynamic Range Meter
#
# A dynamic range meter will give a measurement called a Peak to Loudness Ratio (PLR)
#
# The PLR is the difference between the highest True Peak and the integrated
# loudness. A high PLR indicates wide dynamic range. ("crest factor")
#
# There is also a measure called the PSR (peak to short term ratio) which does
# the same measurement, but based off a three second window and short term
# loudness. PSR measures the dynamics of the last three seconds and a high
# number indicates well preserved transients, a small PSR indicates strong
# limiting or compression.
