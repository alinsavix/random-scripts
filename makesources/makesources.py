#!/usr/bin/env python3
import csv
import msvcrt
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from obswebsocket import obsws, requests
from tdvutil import alintrospect, ppretty

CHEERS_SCENE = "CHEERS"

# global counters (yeah, this is laaaaame)
counters = {
    "ok": 0,
    "exists": 0,
    "missing": 0,
    "failure": 0,
}

# output and flush (unbuffered) without a newline
def out(s: str) -> None:
    print(s, end="")
    sys.stdout.flush()


# output and flush (unbuffered) with a newline
def outl(s: str) -> None:
    print(s)
    sys.stdout.flush()


def parse_bool(s: str):
    if not s or s.lower() == "false":
        return False

    if s.lower() == "true":
        return True

    # else we don't know what it is
    outl(f"WARNING: Unknown bool value {s}")
    return False


def scene_by_name(ws: Any, name: str) -> Optional[int]:
    scenes = ws.call(requests.GetSceneList()).getScenes()
    for s in scenes:
        if name == s['sceneName']:
            return s['sceneIndex']

    return None


def media_source_by_name(ws: Any, name: str) -> Optional[Dict[str, Any]]:
    ret = ws.call(requests.GetInputSettings(inputName=name))
    if not ret.status:
        return None

    # else
    return ret.getInputSettings()


def create_media_source_from_cfg(ws: Any, scene: str, name: str, cfg: Dict[str, Any]) -> Optional[bool]:
    res = ws.call(
        requests.CreateInput(sceneName=scene, inputName=name,
                             inputKind="ffmpeg_source", sceneItemEnabled=False, inputSettings=cfg))
    if not res.status:
        # outl(f"WARING: failed to create media source '{name}'")
        return None

    id = res.getSceneItemId()
    # outl(f"OK: Created media source '{name}' as id {id}")
    return id


# Create an alert source, and print the result (yeah, the print isn't great)
def create_alert(ws: Any, scene: str, name: str, file=Path, loop=False) -> Optional[int]:
    input_cfg = {
        'is_local_file': True,
        'local_file': str(file.resolve()),
        'looping': loop,
        'restart_on_activate': True,
        'hw_decode': True,
        'clear_on_media_end': True,
        'linear_alpha': False,
        'speed_percent': 100,
        'close_when_inactive': False,
    }

    global counters
    src = media_source_by_name(ws, name)
    if src is not None:
        outl("EXISTS (not touching)")
        counters["exists"] += 1
        return None
    else:
        if not file.exists():
            outl(f"FAILURE (missing video file '{file}')")
            counters["missing"] += 1
            return None

        newsrc = create_media_source_from_cfg(ws, scene, name, input_cfg)
        if newsrc is None:
            outl("FAILURE (unknown)")
            counters["failure"] += 1
            return None
        else:
            outl(f"created with id {newsrc}")
            counters["ok"] += 1
            return newsrc


def set_source_transform(ws: Any, scene: str, id: str, transform: Dict[str, Any]) -> bool:
    res = ws.call(
        requests.SetSceneItemTransform(sceneName=scene, sceneItemId=id, sceneItemTransform=transform))
    return res.status


def set_monitoring_all(ws: Any, source: str) -> bool:
    res = ws.call(requests.SetInputAudioMonitorType(inputName=source,
                  monitorType='OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT'))
    return res.status


def set_monitor_only(ws: Any, source: str) -> bool:
    res = ws.call(requests.SetInputAudioMonitorType(inputName=source,
                  monitorType='OBS_MONITORING_TYPE_MONITOR_ONLY'))
    return res.status


def set_volume(ws: Any, source: str, vol: float) -> None:
    res = ws.call(requests.SetInputVolume(inputName=source, inputVolumeMul=vol))
    return res.status


def main():
    if getattr(sys, 'frozen', False):
        # os.chdir(sys._MEIPASS)
        os.chdir(os.path.dirname(sys.executable))
    elif __file__:
        os.chdir(os.path.dirname(__file__))

    ws = obsws("localhost", 4455, "secret")
    ws.connect()
    ver = ws.call(requests.GetVersion()).getObsVersion()
    outl(f"INFO: Connected to OBS, version {ver}")

    sceneidx = scene_by_name(ws, CHEERS_SCENE)

    if sceneidx is not None:
        outl(f"INFO: Found existing scene '{CHEERS_SCENE}' at index {sceneidx}, using that")
    else:
        outl(f"INFO: Couldn't find a scene '{CHEERS_SCENE}', creating")
        x = ws.call(requests.CreateScene(sceneName=CHEERS_SCENE))
        if not x.status:
            outl(f"ERROR: Failed to create scene '{CHEERS_SCENE}'")
            sys.exit(1)


    outl("CREATING SOURCES:")

    with open("botconfig.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Scene"] != CHEERS_SCENE:
                continue

            # otherwise, create it
            out(f"  {row['Source']}: ")
            id = create_alert(ws, CHEERS_SCENE, row["Source"], Path(row["File"]), loop=parse_bool(row["Repeat"]))

            x_loc = int(row["X"]) if row["X"] else 0
            y_loc = int(row["Y"]) if row["Y"] else 0
            scale = float(row["Scale"]) if row["Scale"] else 1.0

            if parse_bool(row["Needs Offset"]):
                y_loc = y_loc - 50

            # set the location, if there's one requested, and we just created it
            if id is not None:
                xform = {
                    "positionX": x_loc,
                    "positionY": y_loc,
                    "scaleX": scale,
                    "scaleY": scale,
                }
                set_source_transform(ws, CHEERS_SCENE, id, xform)

            set_monitor_only(ws, row["Source"])

            vol = float(row["Vol%"]) / 100 if row["Vol%"] else 1.0
            set_volume(ws, row["Source"], vol)



    outl("\nDONE!")
    outl(f"{counters['ok']} created, {counters['exists']} skipped (existing), {counters['missing']} missing video files, {counters['failure']} failures")

    outl("Press any key to exit")
    msvcrt.getch()

    ws.disconnect()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        outl(f"\n\n!!  ERROR: SCRIPT FAILED\n\nexception: {e}")
        outl("\nPress any key to exit")
        msvcrt.getch()



# useful tidbits when I was developing, keeping around in case they're useful:

# zot5 = ws.call(requests.CreateInput(sceneName="CHEERS", inputName="test input",
# inputKind = "ffmpeg_source", sceneItemEnabled = False, inputSettings = input_cfg))
#
# zot = ws.call(requests.GetSceneItemList(sceneName="CHEERS"))
# zot2 = zot.getSceneItems()
# print(ppretty(zot2))

# zot3 = ws.call(requests.GetInputSettings(inputName="test input")).getInputSettings()
# print(zot3)

# zot4 = ws.call(requests.GetInputDefaultSettings(inputKind="ffmpeg_source"))
# print(zot4)

# zot5 = ws.call(requests.CreateInput(sceneName="CHEERS", inputName="test input",
#                inputKind="ffmpeg_source", sceneItemEnabled=False, inputSettings=input_cfg))
# print(zot5)
# print(zot5.status)
# print(zot5.getSceneItemId())
# # scenes = ws.call(requests.GetSceneList()).getScenes()
# # print(scenes)

# # for s in scenes:
# #     name = s['sceneName']
# #     print(f"scene: {name}")
# #     print(s)
