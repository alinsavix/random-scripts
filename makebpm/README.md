# makebpm: a simple bpm matcher for animations

## What & Why

Every now and then, I (Alinsa) have a desire to match an animation to a specific musical beat, for various reasons. Most recently, I needed to do this for my [highlight video](https://www.youtube.com/watch?v=Gdi5vJvTZmQ) of twitch streamer [Jonathan Ong](https://twitch.tv/JonathanOng)'s on-the-fly/live-learned cover of the venerable Sir Mix-a-Lot song "Baby Got Back". His stream makes regular use of an animation of a dancing deer that I wanted to use as part of the highlight video, but I needed it to dance in time with the music! So I put together a squick script to take a series of animation frames and write them into a video file at the right pacing to make the animation happen at the right speed.

The data for that animation is provided with this repository, under the name "deerbutt". You can also create your own animation data to time your own animations.


## Using

`makebpm` is a command-line script written in python. You'll need python (3.7 or newer) with the `tdvutil` and `dataclasses_json` modules installed (i.e. `pip3 install tdvutil dataclasses_json`), and you will need to have `ffmpeg` installed on your system and available in your PATH.

Using the script is pretty straightforward; in most cases you just need to provide an animation name (as available in the `anims` directory) and desired bpm to match. You can also optionally specify the number of cycles of the animation you wish to have in the generated video file, the framerate of the generated video file, the codec with which to encode the generated video pile, and various directories (the directory containing animations, and the cache directory to use for working with them).

The full set of options:

```text
usage: makebpm.py [-h] [-b BPM] [--cycles CYCLES] [--fps FPS] [-o OUTFILE]
                  [--codec {h264,h264_nvenc,prores,webm}] [--animdir ANIMDIR]
                  [--cachedir CACHEDIR] [--debug]
                  animation

Generate an animation appropriate for a given bpm

positional arguments:
  animation             animation name that should be rendered

options:
  -h, --help            show this help message and exit
  -b BPM, --bpm BPM     bpm to which animation should be matched (default:
                        120)
  --cycles CYCLES, -c CYCLES
                        number of animation cycles to render (default: 10)
  --fps FPS             fps of created output video (default: 60)
  -o OUTFILE, --outfile OUTFILE
                        output video file to generate (default:
                        <animname>_<bpm>bpm.mov)
  --codec {h264,h264_nvenc,prores,webm}
                        encoder/codec to use for created video (default: prores)
  --animdir ANIMDIR     directory in which to find animation data (default:
                        'anims/')
  --cachedir CACHEDIR   directory in which to cache extracted animation data
                        (default: 'cache/')
  --debug               Enable debugging output
```

One gotcha to be careful about is that if you are processing an animation that uses transparency, you must select a codec that supports transparency -- currently, that means using `prores` (the default) or `webm`. If you are creating animations that don't involve transparency, you can use a more traditional codec like `h264` if desired.

### Example: A 120bpm dancing deer at 30fps in a webm container

```bash
$ ./makebpm.py --bpm 120 --cycles 10 --fps 30 --codec webm -o deerbutt_120bpm.webm deerbutt
time per video frame: 0.033333s
time per animation cycle: 1.000000s
time per anim frame: 0.007812s
total video time: 10.000000s
Rendering...
frame=    1 fps=0.0 q=0.0 size=       0kB time=00:00:00.00 bitrate=4088.0kbits/s
frame=   15 fps=0.0 q=0.0 size=       0kB time=00:00:00.46 bitrate=   8.7kbits/s
[...]
frame=  301 fps= 30 q=0.0 size=    2690kB time=00:00:10.00 bitrate=2203.1kbits/s
SUCCESS: rendered bpm-matched video to 'deerbutt_120bpm.webm'
```

## Making Your Own

You can create new animations in the `anims` directory. Each animation requires two files: a zip file with the individual frames of the animation, and a json file containing information about the source animation data, which is needed for creating the final animation.

The zip file (`anims/animname.zip`) should contain png-formatted frames, with a filename consisting of the 8 digit frame number (with leading zeroes if needed) and a the `.png` extension (e.g. `00000127.png`). Frame numbers should be sequential, starting with `00000001`.

The json file (`anims/animname.zip`) is relatively self explanatory, and looks like this:

```json
{
    "name": "deerbutt",
    "description": "that deerbutt, but smooooooth",
    "framecount": 128,
    "beats_per_cycle": 2,
    "has_alpha": true
}
```

The only entry that probably needs explanation is `beats_per_cycle`, which is a count of how many beats the animation data contains. In the case of the included `deerbutt` animation, that number is `2` -- the deer's tail swings to the right (one beat), then left (the second beat).

The `has_alpha` flag is just a safety check that makes certain you use/request a transparency-supporting codec when you are rendering an animation that makes use of transparency.

***HINT***: When creating your own animations, use a tool like [flowframes](https://github.com/n00mkrad/flowframes) to create additional in-between frames. The smoothness of the generated videos is dependent on the number of frames available to the tool, so more frames make for smoother animations, especially for lower bpm figures (e.g. slower animations). This was how the 128 frames of deer animation were created, out of a source animation containing under 30 frames.


## Finally

Have some fun with it. Feel free to report bugs/make feature requests on Alinsa's [github page](https://github.com/alinsavix/random-scripts/issues).
