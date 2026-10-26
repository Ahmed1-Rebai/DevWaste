# DevWaste AI: vertical promo clip

A 29 s, 1080x1920 narrated clip for LinkedIn, built with [HyperFrames](https://hyperframes.heygen.com)
(HTML + GSAP rendered to video). Rendered output:
[`out/devwaste-linkedin-vertical.mp4`](out/devwaste-linkedin-vertical.mp4) (H.264, yuv420p,
30 fps, AAC). One still per scene is in
[`../../docs/media/promo-snapshots/`](../../docs/media/promo-snapshots/).

## Source media

Every terminal frame comes from the real recording in [`docs/media/`](../../docs/media/)
(`devwaste` run against the FlakeRadar Docker stack). Nothing is mocked up. The crops in
`assets/` were cut with FFmpeg:

| Asset | Source | Crop (w:h:x:y) |
| --- | --- | --- |
| `scan-table.png` | `screenshot-scan-train.png` | `1470:375:0:0` |
| `train.png` | `screenshot-scan-train.png` | `1370:472:0:385` |
| `monitor-normal.png` | `screenshot-monitor-alert.png` | `1060:245:0:296` |
| `alert-block.png` | `screenshot-monitor-alert.png` | `1700:335:0:570` |
| `alerts-live.mp4` | `devwaste-demo.mp4`, 16.2 s to 21.4 s | `920:330:0:567` |

```bash
ffmpeg -i ../../docs/media/screenshot-scan-train.png -vf "crop=1470:375:0:0" assets/scan-table.png
ffmpeg -ss 16.2 -t 5.2 -i ../../docs/media/devwaste-demo.mp4 -vf "crop=920:330:0:567" \
  -an -c:v libx264 -pix_fmt yuv420p -crf 16 -r 30 assets/alerts-live.mp4
```

Sound effects (`assets/sfx/`, not committed) are the bundled Pixabay-licensed files from the
HyperFrames `media-use` skill. There is no music.

## Voiceover

Six narration lines, one per scene, generated locally with Kokoro-82M through
`npx hyperframes tts` (voice `am_michael`, speed 1.1). The script is in
[`assets/voice/script.txt`](assets/voice/script.txt) and the WAVs are committed, so a render
does not need the TTS model. Each line's `data-volume` was set with
`npx hyperframes normalize-audio --lufs=-16`. The sound effects sit under the voice.

To regenerate a line (needs `pip install kokoro-onnx soundfile`):

```bash
npx hyperframes tts "Live monitoring: all normal." -v am_michael -s 1.1 -o assets/voice/vo-4.wav
```

The narration says only what the screen shows: the five containers, LSTM autoencoder,
about eight seconds of training, all normal, the Redis spike, the CPU driver, and local /
no LLMs / diagnostic-only. It is 53 words, more than the 25-word limit in the original brief,
because a descriptive voice was requested afterwards.

## Scenes

| Time | Composition | Content |
| --- | --- | --- |
| 0–3 s | `hook.html` | "Which container is wasting your resources?" |
| 3–8 s | `stack.html` | `devwaste scan` table, slow pan across 5 containers |
| 8–13 s | `learn.html` | `devwaste train` output, zoom to threshold 3.2925 and "Training complete" |
| 13–17 s | `watch.html` | `devwaste monitor` "All normal" rounds (max score about 1.4) |
| 17–24 s | `catch.html` | Live ALERT rows for `flakeradar-redis-1`, then a pan along score 83.2280 and driver `cpu_percent` |
| 24–29 s | `endcard.html` | Name, facts, repository link, MIT License |

Captions are burned in (`captions.html`) because LinkedIn autoplays muted.

## Render

From the repository root:

```bash
npx skills add heygen-com/hyperframes --all --full-depth   # once: skills incl. sound effects
npx hyperframes browser ensure                            # once: headless Chrome
make video
```

`make video` installs GSAP locally (`npm install`), copies the four sound effects, runs
`hyperframes check`, renders, then loudness-normalizes the mix to -16 LUFS / -1.5 dBTP with
FFmpeg and writes `out/devwaste-linkedin-vertical.mp4`.
