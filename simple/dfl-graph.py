#!/usr/bin/env python3
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Optional, Dict, List
from functools import partial
from pathlib import Path
from tdvutil.argparse import CheckFile
import argparse

# def moving_average(data, window):
#     return np.convolve(data, np.ones(window), 'valid') / window

def moving_avg_dev_slope(data, window):
    padding = (window - 1) // 2

    avg = np.convolve(data, np.ones(window), 'valid') / window
    padded_avg = np.pad(avg, (padding, padding), mode='edge')

    stddev = np.sqrt(np.convolve(np.square(data - padded_avg), np.ones(window), 'valid') / window)
    padded_stddev = np.pad(stddev, (padding, padding), mode='edge')

    seq = np.arange(0, len(padded_avg))
    slope = np.diff(padded_avg) / np.diff(seq)
    padded_slope = np.pad(slope, (1, 0), mode='edge')

    # seq = np.arange(0, len(data))
    # slope = np.diff(data) / np.diff(seq)
    # padded_slope = np.pad(slope, (1, 0), mode='edge')

    # print (len(padded_avg))
    # print (len(padded_stddev))
    # print (len(padded_slope))

    return padded_avg, padded_stddev, padded_slope


def moving_regression(data, window):
    npdata = np.array(data)
    data_roll = np.lib.stride_tricks.as_strided(npdata, shape=(len(npdata) - window + 1, window), strides=(npdata.strides[0], npdata.strides[0]))
    data_mean = data_roll.mean(axis=1, keepdims=True)
    y = np.arange(window).reshape(-1, 1)
    beta = np.linalg.lstsq(y, data_roll.T - data_mean.T, rcond=None)[0].T
    z = beta[:, 0]

    padding = (window - 1) // 2
    extra_padding = (window - 1) % 2
    padded_beta = np.pad(z, (padding, padding + extra_padding), mode='edge')

    return padded_beta



def check_val(val: str) -> bool:
    # print(val)
    # check if it's a float
    try:
        f = float(val)
        if f != 0:
            return True
        else:
            return False
    except ValueError:
        pass

    # Wasn't a float, so just check for typical yes/no things now
    if val == "True" or val == "y":
        return True

    return False


def is_enabled(val: str, if_true: Optional[str] = None, if_false: Optional[str] = None) -> Optional[str]:
    # if the thing we're checking is true
    if check_val(val):
        return if_true.format(val) if if_true is not None else None

    # else, it's false
    return if_false.format(val) if if_false is not None else None


configthings = {
    "uniform_yaw": partial(is_enabled, if_true="UY"),
    "lr_dropout": partial(is_enabled, if_true="LRD"),
    "random_warp": partial(is_enabled, if_true="RW"),
    "true_face_power": partial(is_enabled, if_true="TFP={}"),
    "face_style_power": partial(is_enabled, if_true="FSP={}"),
    "bg_style_power": partial(is_enabled, if_true="BSP={}"),
    "random_src_flip": partial(is_enabled, if_true="SRCFLIP"),
    "random_dst_flip": partial(is_enabled, if_true="DSTFLIP"),
    "batch_size": partial(is_enabled, if_true="BATCH={}"),
    "gan_power": partial(is_enabled, if_true="GAN={}"),
    "retraining_samples": partial(is_enabled, if_true="RETRAIN"),
    "eyes_prio": partial(is_enabled, if_true="EYES"),
    "mouth_prio": partial(is_enabled, if_true="MOUTH"),
    "loss_function": partial(is_enabled, if_true="{}"),
    "random_color": partial(is_enabled, if_true="RNDCOLOR"),
    "random_downsample": partial(is_enabled, if_true="RNDDOWN"),
    "random_noise": partial(is_enabled, if_true="RNDNOISE"),
    "random_blur": partial(is_enabled, if_true="RNDBLUR"),
    "random_jpg": partial(is_enabled, if_true="RNDJPG"),
    # "lr": partial(is_enabled, if_true="LR={}")
}

def norm_config(conf: str, val: str) -> Optional[str]:
    if conf not in configthings:
        return None

    f = configthings[conf]
    return f(val)


def build_config_str(config: Dict[str, str]) -> str:
    # print(config)
    accum: List[str] = []
    for k, v in config.items():
        # print(k, v)
        n = norm_config(k, v)
        if n is not None:
            accum.append(n)

    return "_".join(sorted(accum))


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

    # parser.add_argument(
    #     "--inactive-workspace-dir", "--iwd",
    #     type=Path,
    #     action=CheckFile(must_exist=True),
    #     help="directory in which to store inactive workspaces",
    #     default="W:\\AI\\DeepFaceLab_MVE\\inactive-workspaces",
    # )

    # parser.add_argument(
    #     "--workspace-dir", "--wd",
    #     type=Path,
    #     action=CheckFile(must_exist=True),
    #     help="main DeepFaceLab workspace directory",
    #     default="W:\\AI\\DeepFaceLab_MVE\\workspace",
    # )

    # parser.add_argument(
    #     "-o",
    #     "--outfile",
    #     type=Path,
    #     default=None,
    #     action=CheckFile(extensions={'png'}),
    #     help="filename to output generated graph to (defaults based on input name)",
    # )

    # parser.add_argument(
    #     "--noninteractive",
    #     default=True,
    #     action='store_false',
    #     dest="interactive",
    #     help="don't show interactive graph (implies -w)",
    # )

    # parser.add_argument(
    #     "-w",
    #     "--write",
    #     default=False,
    #     action='store_true',
    #     dest="write_graph",
    #     help="write graph to file on disk",
    # )

    # parser.add_argument(
    #     "--target",
    #     default="general",
    #     type=check_lufs_target,
    #     help="target LUFS, or one of: general, amazon, apple, beatport, spotify, youtube (default: youtube)"
    # )

    # parser.add_argument(
    #     "--target_lufs",
    #     default=0.0,
    #     type=float,
    #     help=argparse.SUPPRESS,
    # )

    # parser.add_argument(
    #     "--target_desc",
    #     default="",
    #     type=str,
    #     help=argparse.SUPPRESS,
    # )

    # parser.add_argument(
    #     "--momentary",
    #     "--no-momentary",
    #     default=False,
    #     action=NegateAction,
    #     nargs=0,
    #     help="generate plots for momentary (400ms window) loudness (default: no)",
    # )

    # parser.add_argument(
    #     "--short",
    #     "--no-short",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    #     help="generate plots for short (3s window) loudness (default: yes)",
    # )

    # parser.add_argument(
    #     "--integrated",
    #     "--no-integrated",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    #     help="generate plots for integrated (full-length) loudness (default: yes)",
    # )

    # parser.add_argument(
    #     "--lra",
    #     "--no-lra",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    #     help="generate plot for loudness range (default: yes)",
    # )

    # parser.add_argument(
    #     "--peaks",
    #     "--no-peaks",
    #     default=False,
    #     action=NegateAction,
    #     nargs=0,
    #     help="generate plot for peak values",
    # )

    # parser.add_argument(
    #     "--clipping",
    #     "--no-clipping",
    #     default=True,
    #     action=NegateAction,
    #     nargs=0,
    #     help="show where true peak is higher than -1.0dbFS (default: yes)",
    # )

    # parser.add_argument(
    #     "--clip-at",
    #     default=-1.0,
    #     type=float,
    #     help="dB value above which is considered to be clipping (default: -1.0)",
    # )

    # positional arguments
    parser.add_argument(
        "logfile",
        type=Path,
        action=CheckFile(must_exist=True),
        default=Path("training.log"),
        nargs="?",
        help="training log to graph",
    )

    parsed_args = parser.parse_args(argv)

    # # make sure we enable writing the graph if the user specifies a filename
    # if parsed_args.outfile:
    #     parsed_args.write_graph = True

    # if not parsed_args.interactive:
    #     parsed_args.write_graph = True

    # if parsed_args.write_graph and parsed_args.outfile is None:
    #     parsed_args.outfile = parsed_args.file.with_suffix(".loudness.png")
    return parsed_args


def main(argv: List[str]) -> int:
    args = parse_arguments(argv[1:])

    # Read the text file
    with open(args.logfile, 'r') as f:
        text = f.read()

    # Initialize dictionaries to store the data for each combination of configuration variables
    source_losses_dict = {}
    dest_losses_dict = {}
    iteration_times_dict = {}
    iteration_nums_dict = {}

    # Split the text into sections based on the header
    sections = re.split(r'=+ Model Summary =+\n', text)[1:]
    print(f"number of sections: {len(sections)}")
    # Process each section

    current_section = ""
    for i, section in enumerate(sections):
        # Extract the configuration variables
        config = {}

        config_lines = re.findall(r'==([^\n]+)==\n', section)
        # print(config_lines)
        for line in config_lines:
            m = re.match(r'^\s+(.+?): (.+?)\s*$', line)
            # if not m:
            #     print(f"no match: {line}")
            if m:
                config[m.group(1)] = m.group(2)
                # print(line)
                # print(m.group(1,2))

        cfg_str = build_config_str(config)
        if cfg_str != current_section:
            print(f"section {i}: {cfg_str}")
            current_section = cfg_str
        else:
            print(f"section {i}: (continued from prev)")
        # print(cfg_str)


        dataline_re = re.compile(r"""
        ^
        \[
            (?P<timestamp> \d\d:\d\d:\d\d)
        \]
        \[
            \# (?P<iteration> \d+)
        \]
        \[
            (?P<iter_time> \d+)ms
        \]
        \[
            (?P<src_loss> \d+\.\d+)   # Can only ever be positive
        \]
        \[
            (?P<dst_loss> \d+\.\d+)
        \]
        """, re.VERBOSE)
        # Extract the loss values for each timestamp

        losses = {}
        lines = section.split('\n')[1:]
        for line in lines:
            if line.startswith('['):
                m = dataline_re.match(line)
                if not m:
                    continue

                # print(m.groupdict())
                # continue
                #  fields = line.strip('[]').split('][')
                # print(fields)
                timestamp = m.group("timestamp")
                iteration = int(m.group("iteration"))
                iteration_time = int(m.group("iter_time"))
                source_loss = float(m.group("src_loss"))
                dest_loss = float(m.group("dst_loss"))

                losses[iteration] = (source_loss, dest_loss, iteration_time)
                # if timestamp not in losses:
                #     losses[timestamp] = []
                # losses[timestamp].append((source_loss, dest_loss))
                # print(f"kept: {fields}")

        # print(f"storing {len(losses)} items for {cfg_str}")
        # print(f"keys: {source_losses_dict}")

        if cfg_str not in source_losses_dict:
            # print("created new list")
            source_losses_dict[cfg_str] = []
            dest_losses_dict[cfg_str] = []
            iteration_times_dict[cfg_str] = []
            iteration_nums_dict[cfg_str] = []

        source_losses_dict[cfg_str].extend([v[0] for v in losses.values()])
        dest_losses_dict[cfg_str].extend([v[1] for v in losses.values()])
        iteration_times_dict[cfg_str].extend([v[2] for v in losses.values()])
        iteration_nums_dict[cfg_str].extend(losses.keys())

        # print(f"after: {source_losses_dict.keys()}")
        # Add the loss values for this combination of configuration variables to the dictionaries
        # config_str = str(config)
        # for ts, loss_values in losses.items():
        #     source_losses = [x[0] for x in loss_values]
        #     dest_losses = [x[1] for x in loss_values]
        #     if cfg_str not in source_losses_dict:
        #         source_losses_dict[cfg_str] = []
        #         dest_losses_dict[cfg_str] = []
        #     source_losses_dict[cfg_str].extend(source_losses)
        #     dest_losses_dict[cfg_str].extend(dest_losses)


    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.6)
    ax1.set_title("Source Loss")
    ax2.set_title("Dest Loss")
    ax3.set_title("Slopes")
    ax4.set_title("Iteration Time (ms)")


    # ax4.set_title("legend")
    ax5.axis("off")
    ax5.legend(loc='center left')

    # Create the plot
    # print(source_losses_dict)
    WINDOW = 15
    for config_str in source_losses_dict:
        if len(source_losses_dict[config_str]) == 0:
            continue
        print(config_str)
        # padding = [np.nan for _ in range(WINDOW - 1)]

        src_mavg, src_mdev, src_slope = moving_avg_dev_slope(source_losses_dict[config_str], WINDOW)
        # mavg = moving_average(source_losses_dict[config_str], WINDOW)
        # mdev = moving_stddev(source_losses_dict[config_str], WINDOW)

        # ax1.plot(iteration_nums_dict[config_str], src_slope, label=config_str + ' source slope')
        ax1.plot(iteration_nums_dict[config_str], src_mavg, label=config_str + ' source loss')
        # ax3.plot(iteration_nums_dict[config_str], src_slope, label=config_str + ' src slope')

        # src_slope = moving_regression(source_losses_dict[config_str], 500)
        # ax3.plot(iteration_nums_dict[config_str], src_slope, label=config_str + ' src slope')


        # ax1.plot(iteration_nums_dict[config_str], mdev, label=config_str + ' source loss deviation')
        # ax1.fill_between(iteration_nums_dict[config_str], src_mavg - src_mdev, src_mavg + src_mdev, alpha=0.5)
        dst_mavg, dst_mdev, dst_slope = moving_avg_dev_slope(dest_losses_dict[config_str], WINDOW)
        ax2.plot(iteration_nums_dict[config_str], dst_mavg, label=config_str + ' destination loss')
        # ax3.plot(iteration_nums_dict[config_str], dst_slope, label=config_str + ' dst slope')

        # dst_slope = moving_regression(dest_losses_dict[config_str], 500)
        # ax3.plot(iteration_nums_dict[config_str], dst_slope, label=config_str + ' src slope')

        time_mavg, time_mdev, time_slope = moving_avg_dev_slope(iteration_times_dict[config_str], WINDOW)
        ax4.plot(iteration_nums_dict[config_str], time_mavg, label=config_str + ' iteration time')

        ax5.plot([], [], label=config_str + " loss")
    # plt.title('Losses over time for all configurations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
