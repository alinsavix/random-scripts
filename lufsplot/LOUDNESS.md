# lufsplot - Static Loudness Plotter

## The need for loudness measurement (and loudness standards)

Anyone who lived through the 90s and 2000s has probably experienced a
world in which loudness normalization was not yet a standard thing -- if
you've ever been watching TV and had a commercial suddenly come on with
*blaring* volume, or owned an iPod and tried to listen to a bunch of CDs
you ripped, or even just made a mix CD, you know how the volumes of various
content can vary, and sometimes drastically. This is unpleasant at best,
and can damage your hearing(!) at worst. This ain't good.

Normalization helps fix this. But first, some history...

## The simplistic fix: Peak normalization

In the early days of digital audio, at least at the consumer level, people
who cared to try to fix this problem on their own generally turned to
simplistic *peak normalization*: Pick a desired max volume (usually very
close to 100% volume), scan the entire track for the highest peak in the
digital audio data, and multiply all of the audio data by a conversion
factor that raised the volume of all the audio so that the highest peak in
the audio ended up at the desired maximum volume.

This has a major problem: The peak volume in an audio composition is
largely unrelated to the listening experience. Imagine an audio track
that is mostly at 5% of the maximum possible volume, with a single-sample
glitch that is at 80% volume. Performing peak normalization on this track
wouldn't change it much at all -- normalizing to 100% would mean all the
audio data in the track would get multiplied by `1.25`, making the
single-sample glitch (which might not even be audible) at 100% volume,
but the rest of the audio data will only be at 6.25% volume[^1], still
largely inaudible!

Music producers also figured out that they could game this system by
making the average volume of their content much higher, via dynamic range
compression, so that for the same peak volume, the music was louder overall,
resulting in the [loudness war](https://en.wikipedia.org/wiki/Loudness_war).

There are plenty of less drastic examples of this -- any content that has
brief very loud content (imagine, say, a gunshot) will suffer from the
same type of problem, at least without range compression that can take the
*punch* out of sounds and music that's supposed to be punchy. What to do?


## Slightly better: Volume normalization

An alternate solution would be to normalized based on the *average* volume
of the content, and set a target average volume (that must be significantly lower than the maximum possible volume, because approximately half the audio
data will be louder than the average). Do the same math, and things work out
somewhat better.

Still, this poses a few problems:

1. If you use an overall average, the final volume levels still might not
represent what would be "equal" volume for everything -- again with the
example of using cannons in music, a 10 second segment in a 2 minute piece
can still change the average significantly, even though the ideal is that
the rest of the music should be set to your target volume. Or the other
way, a movie that contains long sections of near-silence for artistic
reasons will have a lower average volume, so when normalized, all the
sound that *is* present will be louder than it should be.

2. If the difference between the average volume and the peak volume is large,
it's possible that once you do the math to bring the average volume up (or
down) to your targets, the loudest sounds might end up being at a volume
that is above the loudest possible volume for the sound format, clipping
(a.k.a. clipping) in the process. Nobody wants that.

3. When doing naïve volume normalization, the only thing considered is the
values of the individual audio samples. This provides an average volume at
a strictly mathematical level, but suffers from one major drawback: *Human
hearing doesn't work that way*. Human perception of loudness varies by the
*frequency* of the sound, in addition to the actual volume of the sound.
A tone at 1kHz will sound *significantly* louder than a tone at 16kHz,
despite having the same volume.[^2]

## Before we continue: Some definitions

### decibel (dB)

The decibel is the volume unit that most people are used toseeing. It is a
logarithmic unit, where every 10dB represents 10 times more energy (thus 20dB
is 100 times more energy, etc). It has a bunch of technical meanings (and
 a bunch of different forms, like dBm and dBi), but for our purposes, and
 the purposes of manipulating sound, here's most of the knowledge you need:

* 1dB: The minimum volume change at which humans can detect that there was
a change. In general, changes in volume of less than 1dB will not be noticed
by humans.
* 3dB: The volume change that represents a doubling of sound *energy*. If you
"add" two sounds together, the volume will go up by 3dB. However, see next...
* 10dB: The volume change that represents a doubling of *volume*, because human
hearing works on a logrithmic scale.

The difference between doubling of energy (3dB) and doubling volume (10dB) is
why amplifiers go up in power so quickly: If you have a 10W amplifier, to get
double the energy you need a 20W amplifier, but for it to sound twice as loud
you need a *100W* amplifier.

Technically, decibels are always a *ratio* between two values, rather than
an absolute value. So, saying "this thing is 3dB" doesn't make sense without
context, while "this thing is 3dB louder than *that* thing". Frequently,
though, when discussing sound, though, you see absolute values, like "60dB
is the volume of a normal conversation". This is possible because the world
has standardized on audio volume (in air) being relative to a specific sound
pressure level (0.02 millipascals).

Note that in the world of audio production, you will almost always only work
in relative units; the only place where masuring in the "absolute" units makes
sense is in measuring the actual sound output of speakers (or some other sound
generating thing). *All* values seen elsewhere in the audio chain (from mic to
DAW to amplifier) are *always* relative to something.

### dBFS (dB relative to Full Scale)

In most cases, the "something" that dB values are relative to is the "full
scale". The full scale is the maximum signal level that can be represented
in a digital system. The exact definition is a bit deeper than that, but
basically, the maximum volume a digital system can work with before
distoring (clipping) is 0dB. Volume levels for a digital signal are then
referenced to that signal, which is why the VU meters in your favorite DAW
are labelled "0dB" at the top (maximum volume) and go negative from there
(-3dB, -10dB, etc). Technically, the top of that scale should be labelled
"0dBFS" (0dB relative to Full Scale), but many things just label their meters
as "dB".

As an example of what this means, in some systems it is common for digital
audio to be encoded with 8-bit values.[^3] This means each sample can have a
value between -128 and 127. In this case, a sample having a value of 127 would
be at "full scale", and if you tried to make it even a tiny bit louder, so that
the sample would need a value of 128, that value no longer fits within that
range. The software would need to store a value of 127 instead, which means
instead of your audio waveforms having a nice rounded top, they'll have a
flattened top, like someone "clipped" the top off with a pair of scissors.


## The best we have: Loudness normalization and loudness standards

As mentioned before, human hearing has the issue that it hears sound
differently depending on the frequency, which means normalizing purely
based on volume does not give accurate (to the human ear) results. As well,
there are other concerns -- stretches of silence throwing off averages, the
need to still have sometimes loud sounds, and others.

The industry attempted a number of different ways to measure loudness,
eventually standardizing on [Recommendation ITU-R BS.1770](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf),
along with a sprinkling of [EBU Tech 3341](https://tech.ebu.ch/publications/tech3341)
(and previously [EBU R 128](https://tech.ebu.ch/docs/r/r128.pdf)). The contents
of these standards are very dry and math-heavy, but the basics are:

* An audio signal's level is based on not only the raw volume of the signal,
but also on the frequency of the signal, based on a mathematical model of
human hearing. Two signals with the same loudness measurement will *sound*
the same (more or less) to a human, regardless of the frequency of each.
* Periods of silence below a certain level (generally -70dB) are ignored
* Three measurements are ultimately produced for sections of audio at a given
timescale. The measurements/timescales are:

  * **Momentary Loudness (M)**: An extremely short term measurement, using
  a sliding window of 0.4 seconds
  * **Short-term Loudness (S)**: A longer measurement, using a sliding window of 3 seconds
  * **Integrated Loudness (I)**: An overall measurement of the entire media
  (song, commercial, whatever is appropriate to the specific medium)


## Refs

* [Recommendation ITU-R BS.1770](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf) - Algorithms to Measure Audio Program Loudness
* [EBU Tech 3341](https://tech.ebu.ch/docs/tech/tech3341.pdf) - Loudness Metering
* [EBU Tech 3342](https://tech.ebu.ch/docs/tech/tech3342.pdf) - Loudness Range
* [EBU Tech 3343](https://tech.ebu.ch/docs/tech/tech3343.pdf) - Loudness Production Guidelines
* [EBU R 128](https://tech.ebu.ch/docs/r/r128.pdf) - Loudness Normalization
* [Worldwide Loudness Delivery Standards](https://www.rtw.com/en/blog/worldwide-loudness-delivery-standards.html)

## Footnotes

[^1]: We're playing a bit fast and loose with the math here, since sound
data is logarithmic, but this explanation suffices for our purposes

[^2]: https://pressbooks.pub/sound/chapter/frequency-and-loudness-perception/

[^3]: Modern digital audio is generally using 16-bit or 24-bit audio, but we're
using 8-bit audio here because the numbers are smaller, so are easier to read.
8-bit audio is definitely used in the real world though -- the landline phone
system, for example, uses 8-bit samples for the audio.
