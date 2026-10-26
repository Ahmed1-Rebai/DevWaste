# Review: DevWaste AI vertical promo

1080x1920, 30 fps, 28 s. Render with `make video` from the repository root, or
`npx hyperframes@0.8.206 render --quality delivery --fps 30 --output out/devwaste-linkedin-vertical.mp4`
from this folder.

## Sources

Every terminal image is a crop of `docs/media/screenshot-scan-train.png` or
`docs/media/screenshot-monitor-alert.png` (copied to `assets/img/`), recorded
from a real DevWaste run against the FlakeRadar Docker stack. Nothing in the
terminal crops is redrawn. Overlays only add highlight boxes, dimming of
neighbouring rows, and opaque covers that crop away rows outside a shot.
On-screen numbers (5 containers, ~8 s on CPU, threshold 3.2925, scores
13.36 to 83.23, driver `cpu_percent`) come from those screenshots and the
recorded run.

## Scenes

| Time | Scene | Source crop (screenshot px) |
|---|---|---|
| 0-3 s | Hook text | none |
| 3-8 s | `devwaste scan` table, pan left to right at 1.62x | scan-train, y 0-380 |
| 8-13 s | `devwaste train`, push from 1.25x to 1.62x on threshold and "Training complete" | scan-train, y 376-900 |
| 13-17 s | `devwaste monitor`, three "All normal" rounds at 1.9x | monitor-alert, y 205-540 |
| 17-24 s | ALERT rows revealed in the printed order, then 1.95x on name, score 83.2280, driver `cpu_percent` | monitor-alert, y 571-905 |
| 24-28 s | End card | none |

The fourth "All normal" round (max score 3.4444, above the threshold but not
yet two consecutive windows) is cropped out of scene 4 so the shot matches the
"max score about 1.4" fact; it is not altered.

## Checks

- `hyperframes lint` and `hyperframes check`: 0 errors, 0 warnings (layout,
  contrast, craft, audio).
- `snapshot --review`: beat, seam, and 360 px phone sheets read. Terminal text
  is about 36-43 px tall in the frame after zoom, except the 0.9 s wide
  establishing frame at the start of scene 3 (about 28 px).
- Sound was measured, not heard: one bundled "error" tone on the first ALERT
  row and two short whooshes on the upward whips. No music, no voiceover.

## Known weak spots

- The lower third of the frame (below y 1330) is intentionally empty to stay
  clear of LinkedIn's caption and button overlay.
- Scene 3 opens wide; the first second is the least readable on a phone.
