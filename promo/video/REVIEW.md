# Review notes

## Creative calls

- Followed the storyboard as given: hook, real stack, training, normal monitoring, the alert,
  end card. No alternative storylines were built.
- Scene 5 starts on the real recording (ALERT rows arriving one by one), then swaps mid
  punch-in to the screenshot of the same session so the camera can pan along one row
  (row 9: `Score: 83.2280`, `Driver: cpu_percent (99.2%)`). The other rows are dimmed by a
  flat overlay; the footage pixels are not filtered.
- Scene 4 is cropped to the three rounds with max score 1.4205 / 1.3469 / 1.3961. The fourth
  "All normal" round in the recording (max score 3.4444, suppressed by `--consecutive 2`) is
  outside the crop so the caption "all normal" does not sit next to a score above threshold.
- Transitions: blur swap (hook to scan, alert to end card), vertical whip (scan to train),
  dip cut (train to monitor), hard cut (monitor to alert).

## Measured

- `hyperframes lint` and `hyperframes check`: 0 errors, 0 warnings (layout, contrast, craft,
  motion, audio).
- Render: 28.0 s, 1080x1920, H.264 yuv420p, 30 fps, AAC, 5.7 MB.
- Audio (`hyperframes audio`): -15.4 LUFS integrated, true peak -7.9 dBTP, no clipping; hits at
  2.9 s, 7.3 s, 12.9 s, 17.0 s and 22.1 s. The sound was measured, not heard.

## Weakest shots

- Terminal text is upscaled from a 2138 px wide screenshot (1.45x to 2x), so it is a little
  soft. Re-rendering `docs/media/demo.cast` at a larger font would make it sharper.
- At phone width (360 px) the table in scene 2 and the first second of scene 3 are small;
  the captions carry the message there.
