#!/usr/bin/env python3
"""
patch_streamdeck.py - Disable StreamDeck's background disk/process scanning.

StreamDeck.exe has a built-in app discovery system that:
  1. Subscribes to WMI events for EVERY process start/stop (polling every 1s)
  2. For each process event, resolves the exe path, reads version info from
     the file, reads registry keys for browser detection, UWP identity, etc.
  3. Scans the Start Menu Programs folder for .exe and .lnk files
  4. Caches icons for all discovered applications

This script patches the binary to neutralize these subsystems:
  - The WMI subscription is made to SUCCEED but deliver no events (the thread
    blocks waiting forever instead of retrying in a tight loop)
  - The Start Menu scan path is invalidated so directory iteration finds nothing
  - The file glob is changed to match nothing

The "Open Application" action's auto-populated app list will be empty, and the
ApplicationsToMonitor plugin feature won't work. Everything else is unaffected.

Usage:
  python patch_streamdeck.py                  # Patch default install
  python patch_streamdeck.py --dry-run        # Preview changes
  python patch_streamdeck.py --restore        # Restore from backup
  python patch_streamdeck.py "D:\\StreamDeck\\StreamDeck.exe"  # Custom path

Note: Close StreamDeck before running. Needs admin (file is in Program Files).
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_PATH = Path(r"C:\Program Files\Elgato\StreamDeck\StreamDeck.exe")


def to_utf16le(s: str) -> bytes:
    return s.encode("utf-16-le")


def find_all(data: bytes, pattern: bytes) -> list[int]:
    positions = []
    start = 0
    while True:
        pos = data.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


# --- v1 patches that need to be reverted (they cause a retry loop) -----------
# These made the WMI connection/query FAIL, which caused procmon-os-runloop to
# spin retrying. If found, they are reverted before applying the v3 patches.
LEGACY_REVERSALS = [
    (
        "v1 WMI query corruption (XELECT)",
        to_utf16le("XELECT TargetInstance.ProcessId FROM __InstanceOperationEvent"),
        to_utf16le("SELECT TargetInstance.ProcessId FROM __InstanceOperationEvent"),
    ),
    (
        "v1 WMI namespace corruption (XIMV2)",
        to_utf16le(r"\\.\ROOT\XIMV2"),
        to_utf16le(r"\\.\ROOT\CIMV2"),
    ),
]

# --- v2/v3 patches (current) --------------------------------------------------
# Strategy:
# - Process monitor: make WMI subscription SUCCEED but deliver NO events
# - Start Menu scanner: corrupt the inline .lnk extension constant so shortcut
#   files are never recognized and never resolved (kills SHCore threads)
# - FOLDERID: corrupt the CommonPrograms GUID so SHGetKnownFolderPath fails
PATCHES = [
    # === Process Monitor (WMI) ===
    (
        "WMI event filter (WHERE clause)",
        "Makes the WMI event subscription match nothing.\n"
        "         Changes '<>' to '= ' and 'Modification' to 'Xodification'.\n"
        "         The subscription succeeds (no retry loop) but __Class never\n"
        "         equals '__InstanceXodificationEvent', so no events are delivered.",
        to_utf16le("__Class <> '__InstanceModificationEvent'"),
        to_utf16le("__Class =  '__InstanceXodificationEvent'"),
    ),
    (
        "WMI polling interval",
        "Reduces WMI internal polling from every 1s to every 9s.\n"
        "         Even though the WHERE clause filters everything, this reduces\n"
        "         the overhead of WMI's internal Win32_Process snapshot diffing.",
        to_utf16le("WITHIN 1 WHERE"),
        to_utf16le("WITHIN 9 WHERE"),
    ),

    # === Start Menu Scanner (.lnk resolution) ===
    # The scanner walks Start Menu\Programs directories (paths from Qt's
    # QStandardPaths::standardLocations at runtime) and resolves .lnk shortcuts
    # to discover exe paths. This is the expensive part - each .lnk resolution
    # spawns SHCore.dll threads via COM/IShellLink.
    # We patch the inline ".lnk" byte constant (0x6B6E6C2E) used in the code's
    # extension comparison so no file ever matches as a shortcut.
    (
        "Inline .lnk constant (CMP site)",
        "Patches the inline '.lnk' constant in a CMP instruction.\n"
        "         Changes '.lnk' to '.lxk' so no file matches as a shortcut.\n"
        "         Prevents IShellLink resolution (eliminates SHCore thread spam).",
        # In code: CMP DWORD PTR [reg], 0x6B6E6C2E  (".lnk" little-endian)
        # Context: ...u..81 38 2E 6C 6E 6B 75...  (cmp [rax], ".lnk"; jne)
        b"\x81\x38\x2E\x6C\x6E\x6B\x75",
        b"\x81\x38\x2E\x6C\x78\x6B\x75",  # ".lxk"
    ),
    (
        "Inline .lnk constant (MOV site)",
        "Patches the inline '.lnk' constant stored on the stack for comparison.\n"
        "         Changes '.lnk' to '.lxk' so endsWith('.lnk') never matches.",
        # In code: MOV DWORD PTR [rsp+XX], 0x6B6E6C2E; MOV BYTE PTR [rsp+XX], 0
        # Context: C7 44 24 58 2E 6C 6E 6B C6 44 24 5C 00
        b"\xC7\x44\x24\x58\x2E\x6C\x6E\x6B\xC6\x44\x24\x5C\x00",
        b"\xC7\x44\x24\x58\x2E\x6C\x78\x6B\xC6\x44\x24\x5C\x00",  # ".lxk"
    ),

    # === FOLDERID_CommonPrograms GUID ===
    # SHGetKnownFolderPath(FOLDERID_CommonPrograms) returns the all-users
    # Start Menu\Programs path. Corrupting one GUID byte makes the API return
    # E_INVALIDARG, so no path is returned for scanning.
    (
        "FOLDERID_CommonPrograms GUID",
        "Corrupts the FOLDERID_CommonPrograms GUID (one byte: 0x01 -> 0x00).\n"
        "         SHGetKnownFolderPath returns E_INVALIDARG for the invalid GUID.\n"
        "         Eliminates the all-users Start Menu scan path.",
        # {0139D44E-6AFE-49F2-8690-3DAFCAE6FFB8} in bytes_le
        b"\x4E\xD4\x39\x01\xFE\x6A\xF2\x49\x86\x90\x3D\xAF\xCA\xE6\xFF\xB8",
        b"\x4E\xD4\x39\x00\xFE\x6A\xF2\x49\x86\x90\x3D\xAF\xCA\xE6\xFF\xB8",
    ),

    # === Fallback/UI strings (kept from v2 for completeness) ===
    (
        "Start Menu Programs path (fallback string)",
        "Invalidates the hardcoded Start Menu path string.\n"
        "         Changes 'ProgramData' -> 'ProgramXata'.",
        b"C:/ProgramData/Microsoft/Windows/Start Menu/Programs",
        b"C:/ProgramXata/Microsoft/Windows/Start Menu/Programs",
    ),
    (
        "App collector file glob (UI)",
        "Prevents matching in the Open App file picker UI.\n"
        "         Changes '*.exe *.lnk' -> '*.xxx *.xxx'.",
        b"*.exe *.lnk",
        b"*.xxx *.xxx",
    ),
]


def apply_replacements(data, replacements, dry_run, label):
    """Apply a list of (name, search, replace) or (name, desc, search, replace)
    tuples to data. Returns (applied, already, skipped) counts."""
    applied = already = skipped = 0

    for entry in replacements:
        if len(entry) == 3:
            name, search, replace = entry
            desc = None
        else:
            name, desc, search, replace = entry

        assert len(search) == len(replace), f"Bug: length mismatch in '{name}'"

        found = find_all(bytes(data), search)
        found_patched = find_all(bytes(data), replace)

        # If search == replace (identity), skip
        if search == replace:
            continue

        if found_patched and not found:
            print(f"  [ALREADY DONE]    {name}")
            already += 1
            continue

        if not found:
            # Pattern not present — not an error for legacy reversals
            continue

        if len(found) > 1:
            print(f"  [SKIPPED]         {name}")
            print(f"         Found {len(found)} occurrences (expected 1) — skipping for safety")
            skipped += 1
            continue

        pos = found[0]
        verb = "WOULD APPLY" if dry_run else "APPLIED"
        print(f"  [{verb:11s}]    {name}")
        if desc:
            print(f"         {desc}")
        print(f"         Offset: 0x{pos:08X}")

        if not dry_run:
            data[pos : pos + len(search)] = replace
        applied += 1

    return applied, already, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Patch StreamDeck.exe to disable background disk/process scanning"
    )
    parser.add_argument(
        "exe_path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help=f"Path to StreamDeck.exe (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore from backup (.exe.bak)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be patched without modifying the file",
    )
    args = parser.parse_args()

    exe = Path(args.exe_path)
    if not exe.exists():
        print(f"Error: {exe} not found")
        sys.exit(1)

    if args.restore:
        backup = exe.with_suffix(".exe.bak")
        if not backup.exists():
            print(f"Error: backup not found at {backup}")
            sys.exit(1)
        print(f"Restoring from {backup} ...")
        shutil.copy2(backup, exe)
        print("Restored successfully. Restart StreamDeck for changes to take effect.")
        return

    data = bytearray(exe.read_bytes())
    print(f"Loaded {exe} ({len(data):,} bytes)")

    # Step 1: Revert any v1 patches that cause the retry loop
    rev_applied, _, _ = apply_replacements(
        data, LEGACY_REVERSALS, args.dry_run, "legacy"
    )
    if rev_applied:
        print()
        print(f"  Reverted {rev_applied} old patch(es) that caused thread respawning.")
    print()

    # Step 2: Apply v2 patches
    applied, already, skipped = apply_replacements(
        data, PATCHES, args.dry_run, "patch"
    )

    print()
    total_patches = len(PATCHES)

    if applied == 0 and already == total_patches:
        print("All patches already applied. Nothing to do.")
        return

    if applied == 0 and already == 0:
        print("No patches could be applied. Binary may be a different version.")
        sys.exit(1)

    if args.dry_run:
        summary = f"{applied} would be applied"
        if already:
            summary += f", {already} already done"
        if skipped:
            summary += f", {skipped} skipped"
        print(summary + ". Run without --dry-run to apply.")
        return

    # Create backup of the ORIGINAL (unpatched) binary — only on first run
    backup = exe.with_suffix(".exe.bak")
    if not backup.exists():
        # Read the original file again for backup (before our changes)
        shutil.copy2(exe, backup)
        print(f"Backup saved: {backup}")
    else:
        print(f"Backup exists:  {backup} (not overwriting)")

    try:
        exe.write_bytes(bytes(data))
    except PermissionError:
        print(
            f"\nError: cannot write to {exe}\n"
            "Make sure StreamDeck is closed and you're running as Administrator."
        )
        sys.exit(1)

    summary = f"Done: {applied} applied"
    if rev_applied:
        summary += f", {rev_applied} old patch(es) reverted"
    if already:
        summary += f", {already} already done"
    if skipped:
        summary += f", {skipped} skipped"
    print(summary)
    print("\nRestart StreamDeck for changes to take effect.")


if __name__ == "__main__":
    main()
