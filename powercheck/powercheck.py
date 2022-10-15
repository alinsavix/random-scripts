#!/usr/bin/env python3
import threading
from pathlib import Path

import psutil
import pystray
from PIL import Image


# handle triggering everything to shut down
def exit_action(tray):
    tray.visible = False
    exit_event.set()
    tray.stop()


# do the actual monitoring
def powermon(tray):
    tray.visible = True

    on_power = True  # current state, so we know when it changes

    global exit_event
    while not exit_event.is_set():
        bat = psutil.sensors_battery()
        if bat is None:
            tray.remove_notification()
            tray.notify("No power sensor available", title="Power Status Unknown")
            exit_event.wait(30)
            return(exit_action(tray))

        if on_power != bat.power_plugged:
            if bat.power_plugged:
                # we're on power
                tray.remove_notification()
                tray.notify("Power has been reconnected", title="Power On")
            else:
                # we're on battery!
                tray.remove_notification()
                tray.notify("System is now on battery power! Look out!", title="Power Off")

            # sync our state
            on_power = bat.power_plugged

        exit_event.wait(15)  # sleep for 15 seconds, or we get our exit event


def load_icon():
    iconfile = Path(__file__).resolve().with_name("powercheck_icon.png")

    return Image.open(iconfile)


def main():
    # set up a semaphore we can use to tell the application thread that it
    # should be exiting. Yeah, a little sloppy, but good enough for this
    # stupid little script.
    global exit_event
    exit_event = threading.Event()

    tray = pystray.Icon('Power Check')
    tray.title = "Power Check"
    tray.icon = load_icon()
    tray.menu = pystray.Menu(
        pystray.MenuItem(text="Quit", action=lambda: exit_action(tray)),
    )

    tray.run(powermon)


if __name__ == '__main__':
    main()
