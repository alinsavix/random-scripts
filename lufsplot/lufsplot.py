#!/usr/bin/env python3

import re
import subprocess
import dataclasses
from typing import Tuple, List, Optional
import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
import numpy.typing as npt

# ffmpeg -i thing.mp4 -af ebur128=peak=true -ar 4800 -f null -
#
# [Parsed_ebur128_0 @ 00000132f15e35c0] t: 763.8      TARGET:-23 LUFS    M: -14.9 S: -15.4     I: -17.1 LUFS       LRA: 17.2 LU  FTPK: -7.2 - 6.7 dBFS  TPK: -4.2 -3.9 dBFS

# [Parsed_ebur128_0 @ 00000132f15e35c0]        0 1 2
# t: 763.8                                     3 4             time
# TARGET:-23 LUFS                              5 6 7
# M: -14.9                                     8 9             momentary
# S: -15.4                                     10 11           short
# I: -17.1 LUFS                                12 13 14        integrated
# LRA: 17.2 LU                                 15 16 17        lra
# FTPK: -7.2 -6.7 dBFS                         18 19 20 21     tpk_l tpk_r    (frame true peak)
# TPK: -4.2 -3.9 dBFS                          22 23 24 25

#
# also summary:
# [Parsed_ebur128_0 @ 000001fab00a3500] Summary:

#   Integrated loudness:
#     I:         -22.7 LUFS
#     Threshold: -33.7 LUFS

#   Loudness range:
#     LRA:        14.1 LU
#     Threshold: -43.7 LUFS
#     LRA low:   -33.6 LUFS
#     LRA high:  -19.5 LUFS

#   True peak:
#     Peak:       -4.2 dBFS


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

re_ffmpeg_summary = re.compile(r"""
\s+
(?P<field> [A-Z ]+) :
\s+
(?P<value> [.0-9-]+)
\s+
""", re.VERBOSE | re.IGNORECASE)

proc = subprocess.Popen(
    ["ffmpeg", "-i", "test.mp4", "-af", "ebur128=peak=true",
        "-ar", "4800", "-f", "null", "-"],
    shell=False,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
)
assert proc.stdout is not None

# @dataclass
# class LoudnessInfo():
#     time: float
#     target: float
#     momentary: Optional[float]
#     short: Optional[float]
#     integrated: Optional[float]
#     lra: Optional[float]
#     ftpk: float
#     tpk: float


@dataclasses.dataclass
class LoudnessInfo():
    time: List[float] = dataclasses.field(default_factory=list)
    target: List[float] = dataclasses.field(default_factory=list)
    momentary: List[float] = dataclasses.field(default_factory=list)
    short: List[float] = dataclasses.field(default_factory=list)
    integrated: List[float] = dataclasses.field(default_factory=list)
    lra: List[float] = dataclasses.field(default_factory=list)
    ftpk: List[float] = dataclasses.field(default_factory=list)
    tpk: List[float] = dataclasses.field(default_factory=list)


IDX_TIME = 0
IDX_TARGET = 1
IDX_MOMENTARY = 2
IDX_SHORT = 3
IDX_INTEGRATED = 4
IDX_LRA = 5
IDX_FTPK = 6
IDX_TPK = 7

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
T = lind[:, IDX_TIME]
momentary = lind[:, IDX_MOMENTARY]
short = lind[:, IDX_SHORT]
integrated = lind[:, IDX_INTEGRATED]
lra = lind[:, IDX_LRA]
ftpk = lind[:, IDX_FTPK]

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

ax1.axhline(y=-14.0, color='orange', label="_Youtube Integrated Limit", linewidth=3, alpha=1.0)
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

# y_vals_masked = np.ma.masked_where(np.asfarray(li.momentary) < -25, li.momentary)
# y_vals_masked = np.ma.masked_where(M < -25, M)

# plt.plot(li.time, y_vals_masked)
# plt.plot(T, y_vals_masked, label="Momentary", color='r', linewidth=1)
# plt.plot(T, lind[:, IDX_SHORT], label="Short", color='b', linewidth=1)
# plt.plot(T, lind[:, IDX_INTEGRATED], label="Integrated", color='b', linewidth=2)
# plt.show()
# print(lind.shape)
# print(np.obj2sctype(lind))

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
