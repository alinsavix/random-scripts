#!/usr/bin/env python3
#
# A stupid little script to change between DeepFaceLab workspaces quickly, when
# added to shell:sendto (on windows)

import argparse
# import logging as lg
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

    # parser.add_argument(
    #     "--debug",
    #     action='store_true',
    #     default=False,
    #     help="Enable debugging output",
    # )

    parser.add_argument(
        "--workspace-dir", "--wd",
        type=Path,
        action=CheckFile(must_exist=True),
        help="main DeepFaceLab workspace directory",
        default="W:\\AI\\DeepFaceLab_MVE\\workspace",
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

    workspace: Path = args.workspace_dir
    if workspace.exists() and not workspace.is_symlink():
        win32api.MessageBox(0, f"Workspace directory ({workspace}) exists, but is not a symlink", "DFL Activate: Workspace Directory Error")
        return 1

    workspace_target: Path = args.directory
    if not workspace_target.exists() or not workspace_target.is_dir():
        win32api.MessageBox(0, f"Target workspace directory ({workspace_target}) either doesn't exist, or isn't a directory", "DFL Activate: Target Directory Error")
        return 1

    workspace.unlink(missing_ok=True)
    workspace.symlink_to(workspace_target, target_is_directory=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
