#!/usr/bin/env python3
#
# A (very) stupid/bad little script to make it possible to switch between
# 'scenes' in a DeepFaceLab workspace, to allow training or merging to be
# done in smaller chunks than 'the entire destination'. Name your directories
# "aligned.<name>" and "frames.<name>" (etc) and then send them to this
# script (via shell:sendto, on windows) and it will shuffle symlinks to make
# "aligned" and "frames" point to the appropriate 'scene'.
#
# Put the complete set of source frames in a directory named "frames.all" and
# it will automatically copy the needed source frame subset the "frames.<name>"
# directory during the switch (and delete any that aren't in that particular
# scene anymore, as judged by the contents of "aligned.<name>" directory.
import argparse
import re
import shutil
# import os
import sys
from pathlib import Path
from typing import List

import win32api
from tdvutil.argparse import CheckFile, NegateAction

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Manipulate DeepFaceLab workspace directories",
        allow_abbrev=True,
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="Enable debugging output",
    )

    # positional arguments
    parser.add_argument(
        "directory",
        type=Path,
        action=CheckFile(must_exist=True),
        help="workspace directory to activate",
    )

    parsed_args = parser.parse_args(argv)

    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    scenedir: Path = args.directory

    workdir = scenedir.parent
    if not workdir.exists() or not workdir.is_dir():
        win32api.MessageBox(
            0, f"Parent directory ({workdir}) doesn't exist, or isn't a directory", "DFL Scene Change: Sanity Check Failed")
        return 1

    scenename = None
    path_c = scenedir.name.split(".", 1)
    if len(path_c) == 2:
        dirname = path_c[0]
        scenename = path_c[1]

    if dirname != "aligned" and dirname != "merged":
        win32api.MessageBox(
            0, f"Requested scene directory should be an 'aligned' directory (not '{dirname}')", "DFL Scene Change: Bad Scene Dir")
        return 1

    if not scenename:
        win32api.MessageBox(
            0, f"Couldn't find a scene name in provied directory name ({scenedir})", "DFL Scene Change: Unknown Scene")
        return 1

    # direcctory types to do something with
    bases = [Path(x) for x in ["aligned", "frames", "merged", "merged_mask"]]

    # make sure all those directories are either nonexistant, or are symlinks, so we don't
    # overwrite anything.
    for base in bases:
        testpath = workdir / base
        if testpath.exists() and not testpath.is_symlink():
            win32api.MessageBox(
                0, f"Current scene link for {testpath} exists, but isn't a symlink", "DFL Scene Change: Symlink Dir Error")
            return 1

        testpath2 = testpath.with_suffix(f".{scenename}")
        if testpath2.exists() and not testpath2.is_dir():
            win32api.MessageBox(
                0, f"Target directory for base type {base} exists, but isn't a symlink", "DFL Scene Change: Symlink Dir Error")
            return 1


    # Special case: Make sure the 'aligned' target directory actually exists, because
    # otherwise there's no point.
    if not (workdir / "aligned").with_suffix(f".{scenename}").is_dir():
        win32api.MessageBox(
            0, f"Aligned directory for scene (aligned.{scenename}) isn't a directory or doesn't exist", "DFL Scene Change: Symlink Dir Error")
        return 1

    # Everything seems ok? Lets do some work
    for base in bases:
        linkpath = workdir / base
        targetdir = workdir / base.with_suffix(f".{scenename}")

        linkpath.unlink(missing_ok=True)
        if not targetdir.exists():
            targetdir.mkdir(exist_ok=True)

        linkpath.symlink_to(targetdir, target_is_directory=True)


    # Main linking done, now time for extras
    #
    # If there's a frames directory and a frames.all directory, try to fix
    # up the frames directory to try to make it match the aligned directory.
    frames_dir = (workdir / "frames").with_suffix(f".{scenename}")
    frames_all = workdir / "frames.all"

    if not frames_dir.is_dir() or not frames_all.is_dir():
        return 0

    aligned_dir = (workdir / "aligned"). with_suffix(f".{scenename}")
    aligned_frames = {re.split("[_.]", x.name)[0] for x in aligned_dir.iterdir()}
    full_frames = {x.name.split(".")[0] for x in frames_dir.iterdir()}

    in_aligned_only = aligned_frames.difference(full_frames)  # Frames to add
    in_full_only = full_frames.difference(aligned_frames)  # Frames to remove

    # win32api.MessageBox(
    #     0, f"Frames to add: {in_aligned_only}", "DFL Scene Change: Frame directory")
    # win32api.MessageBox(
    #     0, f"Frames to remove: {in_full_only}", "DFL Scene Change: Frame directory")

    # actually delete
    for f in in_full_only:
        fr = (frames_dir / f).with_suffix(".jpg")
        fr.unlink(missing_ok=True)

    # and copy missing
    for f in in_aligned_only:
        src_file = (frames_all / f).with_suffix(".jpg")
        dst_file = (frames_dir / f).with_suffix(".jpg")

        # We copy instead of linking, just to maybe provide some amount of safety
        shutil.copy(src_file, dst_file)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
