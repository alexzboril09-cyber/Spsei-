# Stick Man "Rules" Video Maker

Makes vertical (9:16) creepy "rules" story videos for TikTok, Shorts and Reels.
They star an animated stick man and have big captions, a clock, rule cards, text
messages, knocking, sound effects, and an ending where viewers **choose what the
stick man does next in the comments** (A or B).

Each video is one small story file in `parts/`. To make the next part, write a new file.

## Quick start

```bash
pip install numpy pillow kokoro-onnx soundfile   # once
# also install ffmpeg: https://ffmpeg.org/download.html

python make_video.py parts/part01.json                      # full quality 1080x1920, with narrator
python make_video.py parts/part01.json --preview            # quick low-res check
python make_video.py parts/part01.json --voice am_michael   # different narrator
python make_video.py parts/part01.json --no-voice           # captions + music only
python make_video.py parts/*.json                           # render all parts
```

Videos go to `output/`. Each video also gets a `.post.txt` file with a ready
caption, the A/B options for a pinned comment, hashtags, and the narration
script (if you'd rather record the voice yourself).

## Narrator voice

An AI narrator reads every `say` line, and the captions follow the voice. It runs offline using the
Kokoro voice model (~120 MB), which downloads automatically into `voice_models/` the first time.
Voices: `am_onyx` (deep, default), `am_michael`, `am_adam`, `bm_george` (British), `af_bella`, `af_nicole`.
Speed: `--rate +10%`. To make the narrator say something different from the caption, add `"voice": "..."` to a beat.

## The series loop

1. Post **Part 1**. It ends with "COMMENT A OR B".
2. Pin a comment such as "A = pull the cord, B = stay seated".
3. Read the comments, and whichever answer has more likes wins.
4. Write the next part starting with `"kicker": "YOU CHOSE A"`, and end it with a new choice.

`part02A.json` and `part02B.json` are both written already, so whichever answer wins, Part 2 is ready.

## Rules for interesting episodes

- **Hook in the first 2 seconds:** start with a time and a problem ("11:58 PM. You missed every bus home.").
- **3 rules per episode, max.** Each one must come back later in the story.
- **Break a rule, or make a rule backfire** (the fake name "Mark" starts texting you).
- **Change the scene every 2-4 seconds.** One sentence per beat.
- **Highlight 1-2 words** per line (`hl`): times, names, the scary word.
- **Make the clock tick forward**, and use `clock_alert` when a rule's time arrives.
- **End on a cliffhanger choice** where both options sound risky. Never use an obvious safe answer.
- **Keep it 40-70 seconds.**

## Story file format

```json
{
  "series": "Route 0",          // name shown under the clock
  "part": 3,
  "accent": "#c6ff3d",          // highlight color
  "handle": "@yourchannel",
  "post_caption": "...",
  "beats": [ { ...one beat per scene... } ]
}
```

Every beat can use:

| key | meaning |
|---|---|
| `scene` | which picture (see below) |
| `say` | the line (caption + narrator) |
| `hl` | words to highlight, e.g. `["back", "row."]` |
| `clock` | change the clock, e.g. `"3:07 AM"` |
| `clock_alert` | `true` = red clock |
| `sfx` | `knock`, `tap`, `buzz`, `sting`, `whoosh`, `brakes`, `heartbeat`, `click`, `static`, `riser` (string or list) |
| `shake` / `flicker` / `zoom` / `fade` | camera effects (`shake: 1`, the rest `true`) |
| `dur` | force a length in seconds (otherwise it's timed from the text) |

Scenes and their extra keys:

| scene | extra keys |
|---|---|
| `title` | `kicker`, `title`, `subtitle` |
| `street` | `bus`: `arrive` / `stay` / `leave`; `man` |
| `bus` | `man`, `driver`, `back_figure`: `appear` / `there`, `red_eyes`, `sign`, `stopped` |
| `bus_window` | `hands` (number of handprints), `face_outside`, `stopped` |
| `bedroom` | `man`, `lamp` |
| `door` | `knock`, `feet` (shadow under the door), `light_under` |
| `phone` | `contact`, `time`, `messages` [`{"from":"me"/"them","text":"...","creepy":true}`], `typing` |
| `rule` | `num`, `text`, `icon`: `seat` `name` `window` `eye` `light` `door` `clock` `phone` `cord` `stop` |
| `face` | `expr`: `calm` `worried` `scared` `shocked` `smile` |
| `text` | `lines` (big red words), `size`, `color` |
| `choice` | `question`, `a`, `b`, `next` (e.g. `"PART 3"`), `note` |

`man` controls the stick man: `{"pose": "stand"}`, `{"from": 0.1, "to": 0.6}` (it walks across the screen,
values are 0-1 of the width), `pose` = `stand` `walk` `run` `sit` `phone` `scared` `shiver` `lie`, or
`"man": false` to hide him.

Copy `parts/template.json` to start a new part.
