# StreamDeck Disk (and registry) Scanning Patcher

The Elgato StreamDeck software contains a background app-discovery system that **nearly continuously scans your disk** even when you're not using the "Open Application" action (which is the main source of scanning). On a system without much on it, it's no big deal, but on a system with a lot of data, it can end up chewing through a significant amount of CPU time -- on Alinsa's system, the scanning uses basically an entire core of CPU, and is running 30 - 40% of the time when her system is in active use. Elgato hasn't been responsive to requests to allow this to be disabled, so here we are.

What the StreamDeck software actually does:
1. Subscribes to WMI events for **every process start/stop** (polling every 1 second)
2. Walks the Start Menu Programs folders, resolving every `.lnk` shortcut via COM/IShellLink (spawning dozens of SHCore.dll threads)
3. Reads exe version info, registry keys, and caches icons for all discovered applications (and refreshes these regularly)

This causes constant disk I/O, CPU usage, and thousands of file-access events visible in Process Monitor — even on a fresh install with no "Open Application" actions configured.

This script makes patches to the StreamDeck.exe binary to disable this scanning while leaving all other functionality intact. We hope. This is definitely a ***huge*** hack and you use it at your own risk; if it breaks something, you get to keep both pieces.

---

## Usage

```powershell
# Preview what would be changed (no modifications):
python streamdeck_patch.py --dry-run

# Apply patches (default install path):
python streamdeck_patch.py

# Apply patches to a custom install path:
python streamdeck_patch.py "D:\StreamDeck\StreamDeck.exe"

# Restore the original binary from backup:
python streamdeck_patch.py --restore
```

### Requirements

- **Python 3.10+** (uses `match` syntax and type hints)
- **Close StreamDeck** before patching (the exe is locked while running)
- **Run as Administrator** (because StreamDeck.exe lives in `Program Files`)
- No third-party dependencies

### Backup

On first run, the script creates `StreamDeck.exe.bak` alongside the original. Use `--restore` to revert.

---

## Patches

| # | Patch Name | What It Changes | How It Stops Scanning | StreamDeck Functionality Lost |
|---|-----------|----------------|----------------------|------------------------------|
| 1 | **WMI event filter (WHERE clause)** | `__Class <> '__InstanceModificationEvent'` → `__Class = '__InstanceXodificationEvent'` (UTF-16LE) | The WMI subscription succeeds but the WHERE clause never matches any real event class, so no process start/stop notifications are delivered. The thread blocks waiting forever instead of retrying. | **ApplicationsToMonitor** plugin feature won't work — plugins don't get notified when their companion apps start/stop. |
| 2 | **WMI polling interval** | `WITHIN 1` → `WITHIN 9` (UTF-16LE) | Reduces WMI's internal polling from 1s to 9s, lowering the overhead of Windows diffing the process list internally. Belt-and-suspenders with patch #1. | None beyond #1. |
| 3 | **Inline .lnk constant (CMP site)** | `CMP [reg], ".lnk"` → `CMP [reg], ".lxk"` (raw x86-64 bytes) | The code that checks if a discovered file is a shortcut never matches, so no `.lnk` file is passed to `IShellLink::GetPath()`. Eliminates SHCore.dll thread spam. | **"Open Application" action** auto-populated app list will be empty. |
| 4 | **Inline .lnk constant (MOV site)** | `MOV [rsp+58h], ".lnk"` → `MOV [rsp+58h], ".lxk"` (raw x86-64 bytes) | Second code path that constructs `.lnk` for comparison — same effect as #3, different call site (called from 8 locations). | Same as #3. |
| 5 | **FOLDERID_CommonPrograms GUID** | One byte of `{0139D44E-6AFE-49F2-...}`: `0x01` → `0x00` | `SHGetKnownFolderPath()` returns `E_INVALIDARG` for the corrupted GUID. The all-users Start Menu Programs path is never returned for scanning. | Same as #3. |
| 6 | **Start Menu Programs path (fallback)** | `C:/ProgramData/Microsoft/Windows/Start Menu/Programs` → `C:/ProgramXata/...` | Corrupts the hardcoded fallback path string so directory iteration finds nothing if the API path somehow still works. | Same as #3. |
| 7 | **App collector file glob (UI)** | `*.exe *.lnk` → `*.xxx *.xxx` | The file filter in the "Open Application" dropdown/search matches nothing. | "Open Application" search/filter shows no results. |

---

## Impacts

### What stops working

- **"Open Application" action auto-complete:** The dropdown that shows installed apps will be empty. You can still use the action by clicking "Choose App" and manually browsing to an exe file.
- **ApplicationsToMonitor (plugin feature):** Plugins declaring `"ApplicationsToMonitor"` in their manifest won't receive `applicationDidLaunch` / `applicationDidTerminate` events. Currently affects:
  - `bot.streamer.streamdeck` (Streamer.bot connection detection)
  - `com.elgato.volume-controller` (ElgatoAudioControlServer detection)

  These plugins still *work* — they just won't auto-detect when their companion apps start.

### What keeps working (everything else)

- All button actions, hotkeys, multi-actions, folders, pages, profiles
- Device management, display, brightness, screensaver
- OBS, Twitch, Spotify, and other stream integrations
- Icon library, custom icons
- Plugin system (all plugins continue to function)
- Stream Deck mobile app connection
- Drag-and-drop, import/export profiles

---

## Version Compatibility

This script uses **byte-pattern matching** rather than fixed offsets, so it works across StreamDeck versions as long as the compiled patterns remain the same. The script will:

- **Refuse to patch** if a pattern is found more than once (safety check)
- **Report "already done"** for patterns that were previously patched
- **Skip cleanly** if a pattern isn't found (e.g., future version changed the code)

Tested on StreamDeck.exe v7.4.0.22712

If a future update changes the compiled code, the script will report which patterns couldn't be found.
