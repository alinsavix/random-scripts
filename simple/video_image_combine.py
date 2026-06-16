#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class CombineError(Exception):
    pass


@dataclass(frozen=True)
class AudioSpec:
    codec: str
    sample_rate: str
    channels: int
    layout: str
    time_base: str
    bit_rate: str


@dataclass(frozen=True)
class VideoSpec:
    codec: str
    width: int
    height: int
    pix_fmt: str
    field_order: str
    frame_rate: str
    time_base: str
    timescale: int
    audio: tuple[AudioSpec, ...]


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    streams: tuple[dict, ...]
    duration: Decimal

    @property
    def video_streams(self) -> list[dict]:
        return [stream for stream in self.streams if stream.get("codec_type") == "video"]

    @property
    def audio_streams(self) -> list[dict]:
        return [stream for stream in self.streams if stream.get("codec_type") == "audio"]


@dataclass(frozen=True)
class VideoReportRow:
    name: str
    offset: Decimal
    length: Decimal


def log(message: str = "") -> None:
    print(message, file=sys.stderr)


def fail(message: str) -> None:
    raise CombineError(message)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


def capture(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    return completed.stdout


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        fail(f"required command not found: {name}")


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    fail(
        f"unsupported file extension for '{path}' "
        "(expected mp4/m4v/mov video or png/jpg/jpeg/webp/bmp image)"
    )


def parse_positive_decimal(value: str, label: str) -> Decimal:
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        fail(f"{label} must be a positive number of seconds")
    if decimal <= 0:
        fail(f"{label} must be greater than zero")
    return decimal


def parse_duration(value: object, path: Path) -> Decimal:
    if value is None:
        fail(f"failed to read media duration from '{path}'")
    try:
        duration = Decimal(str(value))
    except InvalidOperation:
        fail(f"failed to read media duration from '{path}'")
    if duration < 0:
        fail(f"invalid negative media duration for '{path}'")
    return duration


def probe_media(path: Path) -> MediaInfo:
    try:
        raw = capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    except subprocess.CalledProcessError as exc:
        fail(f"ffprobe failed for '{path}' with exit code {exc.returncode}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"ffprobe returned invalid JSON for '{path}': {exc}")

    streams = tuple(parsed.get("streams") or ())
    duration = parse_duration((parsed.get("format") or {}).get("duration"), path)
    return MediaInfo(path=path, streams=streams, duration=duration)


def require_one_video_stream(info: MediaInfo) -> dict:
    videos = info.video_streams
    if len(videos) != 1:
        fail(f"'{info.path}' must have exactly one video stream; found {len(videos)}")
    return videos[0]


def stream_value(stream: dict, key: str, path: Path, stream_name: str) -> str:
    value = stream.get(key)
    if value is None or value == "":
        fail(f"failed to read {key} from '{path}' {stream_name}")
    return str(value)


def time_base_to_timescale(time_base: str) -> int:
    try:
        fraction = Fraction(time_base)
    except ValueError:
        fail(f"unexpected video time_base '{time_base}'")
    if fraction.numerator <= 0 or fraction.denominator <= 0:
        fail(f"invalid video time_base '{time_base}'")
    if fraction.numerator != 1:
        fail(f"video time_base '{time_base}' cannot be represented as an integer MP4 timescale")
    return fraction.denominator


def layout_for_anullsrc(stream: dict, path: Path, index: int) -> str:
    layout = stream.get("channel_layout") or stream.get("ch_layout")
    if layout and layout != "unknown":
        return str(layout)

    channels_raw = stream.get("channels")
    try:
        channels = int(channels_raw)
    except (TypeError, ValueError):
        fail(f"failed to read channels from '{path}' audio stream {index}")

    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    fail(
        f"audio channel layout is unknown for '{path}' audio stream {index}; "
        f"cannot generate matching silent audio for {channels} channels"
    )


def build_reference_spec(info: MediaInfo) -> VideoSpec:
    video = require_one_video_stream(info)
    codec = stream_value(video, "codec_name", info.path, "video stream")
    if codec != "h264":
        fail(
            "image clip generation currently supports h264 video inputs only; "
            f"'{info.path}' is '{codec}'"
        )

    frame_rate = stream_value(video, "r_frame_rate", info.path, "video stream")
    if frame_rate == "0/0":
        fail(f"failed to determine frame rate for '{info.path}'")

    time_base = stream_value(video, "time_base", info.path, "video stream")
    audio_specs: list[AudioSpec] = []
    for index, audio in enumerate(info.audio_streams):
        audio_codec = stream_value(audio, "codec_name", info.path, f"audio stream {index}")
        if audio_codec != "aac":
            fail(
                "image clip generation currently supports aac audio inputs only; "
                f"'{info.path}' audio stream {index} is '{audio_codec}'"
            )
        sample_rate = stream_value(audio, "sample_rate", info.path, f"audio stream {index}")
        channels = int(stream_value(audio, "channels", info.path, f"audio stream {index}"))
        audio_specs.append(
            AudioSpec(
                codec=audio_codec,
                sample_rate=sample_rate,
                channels=channels,
                layout=layout_for_anullsrc(audio, info.path, index),
                time_base=stream_value(audio, "time_base", info.path, f"audio stream {index}"),
                bit_rate=str(audio.get("bit_rate") or "320000"),
            )
        )

    return VideoSpec(
        codec=codec,
        width=int(stream_value(video, "width", info.path, "video stream")),
        height=int(stream_value(video, "height", info.path, "video stream")),
        pix_fmt=stream_value(video, "pix_fmt", info.path, "video stream"),
        field_order=str(video.get("field_order") or "unknown"),
        frame_rate=frame_rate,
        time_base=time_base,
        timescale=time_base_to_timescale(time_base),
        audio=tuple(audio_specs),
    )


def compare(name: str, actual: object, expected: object, path: Path) -> None:
    if actual != expected:
        fail(f"'{path}' has a different {name}: {actual} (expected {expected})")


def validate_against_reference(info: MediaInfo, reference: VideoSpec) -> None:
    video = require_one_video_stream(info)
    compare("video codec", video.get("codec_name"), reference.codec, info.path)
    compare("width", int(video.get("width", -1)), reference.width, info.path)
    compare("height", int(video.get("height", -1)), reference.height, info.path)
    compare("pixel format", video.get("pix_fmt"), reference.pix_fmt, info.path)
    compare("field order", str(video.get("field_order") or "unknown"), reference.field_order, info.path)
    compare("nominal frame rate", video.get("r_frame_rate"), reference.frame_rate, info.path)
    compare("video timebase", video.get("time_base"), reference.time_base, info.path)

    audio_streams = info.audio_streams
    if len(audio_streams) != len(reference.audio):
        fail(f"'{info.path}' has {len(audio_streams)} audio stream(s), expected {len(reference.audio)}")

    for index, (audio, expected) in enumerate(zip(audio_streams, reference.audio)):
        compare(f"audio stream {index} codec", audio.get("codec_name"), expected.codec, info.path)
        compare(f"audio stream {index} sample rate", audio.get("sample_rate"), expected.sample_rate, info.path)
        compare(f"audio stream {index} channel count", int(audio.get("channels", -1)), expected.channels, info.path)
        compare(
            f"audio stream {index} channel layout",
            layout_for_anullsrc(audio, info.path, index),
            expected.layout,
            info.path,
        )
        compare(f"audio stream {index} audio timebase", audio.get("time_base"), expected.time_base, info.path)


def concat_file_line(path: Path) -> str:
    escaped = path.name.replace("'", "'\\''")
    return f"file '{escaped}'\n"


def format_seconds(seconds: Decimal) -> str:
    millis = int((seconds * Decimal("1000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def remux_video_segment(source: Path, target: Path, reference: VideoSpec) -> None:
    command = ["ffmpeg", "-hide_banner", "-y", "-i", str(source), "-map", "0:v:0"]
    for index in range(len(reference.audio)):
        command += ["-map", f"0:a:{index}"]
    command += ["-dn", "-sn", "-map_chapters", "-1", "-map_metadata", "-1", "-c", "copy", str(target)]
    run(command)


def encode_image_segment(source: Path, target: Path, duration: Decimal, reference: VideoSpec) -> None:
    command = ["ffmpeg", "-hide_banner", "-y", "-loop", "1", "-i", str(source)]
    for audio in reference.audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout={audio.layout}:sample_rate={audio.sample_rate}",
        ]

    command += ["-t", str(duration), "-map", "0:v:0"]
    for index in range(len(reference.audio)):
        command += ["-map", f"{index + 1}:a:0"]

    vf = (
        f"scale={reference.width}:{reference.height}:force_original_aspect_ratio=decrease,"
        f"pad={reference.width}:{reference.height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={reference.frame_rate},format={reference.pix_fmt}"
    )
    command += [
        "-vf",
        vf,
        "-r",
        reference.frame_rate,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "13",
        "-pix_fmt",
        reference.pix_fmt,
        "-video_track_timescale",
        str(reference.timescale),
    ]
    if reference.audio:
        command += ["-c:a", "aac"]
        for index, audio in enumerate(reference.audio):
            command += [f"-b:a:{index}", audio.bit_rate]
    command += ["-shortest", "-map_chapters", "-1", "-map_metadata", "-1", str(target)]
    run(command)

    encoded_info = probe_media(target)
    encoded_video = require_one_video_stream(encoded_info)
    encoded_time_base = encoded_video.get("time_base")
    if encoded_time_base != reference.time_base:
        fail(
            f"encoded image segment '{source}' has time_base '{encoded_time_base}', "
            f"expected '{reference.time_base}'"
        )


def concatenate(concat_list: Path, output: Path, reference: VideoSpec) -> None:
    command = ["ffmpeg", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-map", "0:v:0"]
    for index in range(len(reference.audio)):
        command += ["-map", f"0:a:{index}"]
    command += ["-dn", "-sn", "-map_chapters", "-1", "-map_metadata", "-1", "-c", "copy", str(output)]
    run(command)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="video_image_combine.sh",
        description="Combine matching videos with encoded crash-image clips in between.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  video_image_combine.sh -o combined.mp4 video1.mp4 stream_crashed.png "
            "video2.mp4 stream_crashed_again.png video3.mp4"
        ),
    )
    parser.add_argument("media", nargs="+", type=Path, metavar="video_or_image")
    parser.add_argument("-o", "--output", type=Path, default=Path("combined.mp4"), help="Output MP4 path.")
    parser.add_argument(
        "-d",
        "--duration",
        default="15",
        help="Duration in seconds for each crash-image clip. Default: 15.",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep generated intermediate files for debugging.")
    args = parser.parse_args(argv)
    args.duration = parse_positive_decimal(str(args.duration), "duration")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    require_command("ffmpeg")
    require_command("ffprobe")

    output_parent = args.output.parent
    if output_parent != Path(".") and not output_parent.is_dir():
        fail(f"output directory does not exist: {output_parent}")

    types: list[str] = []
    video_paths: list[Path] = []
    for item in args.media:
        if not item.is_file():
            fail(f"input file does not exist: {item}")
        item_type = media_type(item)
        types.append(item_type)
        if item_type == "video":
            video_paths.append(item)

    if not video_paths:
        fail("at least one video input is required")

    output_abs = args.output.resolve()
    for item in args.media:
        if item.resolve() == output_abs:
            fail(f"output path must not be one of the inputs: {args.output}")

    video_infos = {path: probe_media(path) for path in video_paths}
    reference_info = video_infos[video_paths[0]]
    reference = build_reference_spec(reference_info)
    for path in video_paths:
        validate_against_reference(video_infos[path], reference)

    work_dir = Path(tempfile.mkdtemp(prefix=".video_image_combine.", dir="."))
    try:
        concat_list = work_dir / "inputs.txt"
        report_rows: list[VideoReportRow] = []
        cumulative_offset = Decimal("0")

        log(f"Reference video: {video_paths[0]}")
        log(
            f"Video: {reference.codec}, {reference.width}x{reference.height}, "
            f"{reference.pix_fmt}, fps {reference.frame_rate}, "
            f"time_base {reference.time_base}, MP4 timescale {reference.timescale}"
        )
        log(f"Audio streams: {len(reference.audio)}" if reference.audio else "Audio streams: none")

        with concat_list.open("w", encoding="utf-8", newline="\n") as file:
            for index, (item, item_type) in enumerate(zip(args.media, types)):
                segment = work_dir / f"segment_{index:03d}.mp4"
                if item_type == "video":
                    log(f"Preparing video segment: {item}")
                    remux_video_segment(item, segment, reference)
                else:
                    log(f"Encoding crash-image segment: {item} ({args.duration}s)")
                    encode_image_segment(item, segment, args.duration, reference)

                segment_info = probe_media(segment)
                if item_type == "video":
                    report_rows.append(
                        VideoReportRow(name=str(item), offset=cumulative_offset, length=segment_info.duration)
                    )
                cumulative_offset += segment_info.duration
                file.write(concat_file_line(segment))

        log(f"Concatenating {len(args.media)} segment(s) into {args.output}")
        concatenate(concat_list, args.output, reference)
        log(f"Wrote {args.output}")

        log()
        log("Video offsets:")
        for index, row in enumerate(report_rows, start=1):
            log(
                f"  {index:3d}  offset {format_seconds(row.offset)}  "
                f"length {format_seconds(row.length)}  {row.name}"
            )
    finally:
        if args.keep_temp:
            log(f"kept temporary files in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CombineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
