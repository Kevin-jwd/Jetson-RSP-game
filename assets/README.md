# AI hand images

Drop three images here, named after the class they show:

| File | Shown when the AI plays |
| --- | --- |
| `rock.png` | rock |
| `paper.png` | paper |
| `scissors.png` | scissors |

`.png` with a transparent background is best — the image is overlaid on the
webcam video, and an opaque background shows as a rectangle. `.jpg`, `.jpeg` and
`.webp` also load. Size does not matter; each image is scaled to about half the
video height, keeping its aspect ratio.

Until the files exist the game draws a labelled card instead, so 1인용 mode is
playable without them.
