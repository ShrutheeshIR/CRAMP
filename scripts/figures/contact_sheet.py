#!/usr/bin/env python3
"""Montage rendered stills side by side so variants can be compared at a glance.

    python3 scripts/figures/contact_sheet.py --out sheet.png \
        --labels "keep" "cull" "glass" "cutaway" --cols 2 a.png b.png c.png d.png

Usually invoked by render_still.py after a sweep rather than by hand.

--bg checker is the default and matters more than it looks: the stills render
RGBA with a transparent background, and a render that silently lost its alpha
channel looks perfectly fine on white. Compositing over a checkerboard makes that
failure obvious, while --bg white previews how the figure will actually sit on
the paper's page.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Matches the sibling repos' backdrop, so a dark sheet reads like the video.
DARK = (0.102, 0.102, 0.18)


def checkerboard(shape, size=16):
    """A light/dark checkerboard the size of an image, for compositing alpha over."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    tiles = ((ys // size) + (xs // size)) % 2
    return np.where(tiles[..., None] == 0, 0.86, 0.72) * np.ones(3)


def composite(path, mode):
    """Load an RGBA PNG and flatten it onto the chosen background."""
    img = plt.imread(path)
    if img.ndim == 2:
        img = np.dstack([img] * 3)
    if img.shape[2] == 3:
        return img[..., :3], False

    rgb, alpha = img[..., :3], img[..., 3:4]
    if mode == "checker":
        bg = checkerboard(img.shape[:2])
    elif mode == "dark":
        bg = np.ones_like(rgb) * np.array(DARK)
    else:
        bg = np.ones_like(rgb)
    return rgb * alpha + bg * (1.0 - alpha), True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one caption per image, in order")
    ap.add_argument("--cols", type=int, default=None,
                    help="columns (default: chosen to stay roughly square)")
    ap.add_argument("--bg", choices=("checker", "white", "dark"), default="checker",
                    help="what transparent pixels are composited over "
                         "(default: checker, which exposes a lost alpha channel)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--width", type=float, default=16.0, help="figure width in inches")
    args = ap.parse_args()

    missing = [p for p in args.images if not os.path.isfile(p)]
    if missing:
        raise SystemExit("[sheet] missing image(s): " + ", ".join(missing))

    n = len(args.images)
    cols = args.cols or min(n, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))

    tiles, had_alpha = [], []
    for path in args.images:
        tile, alpha = composite(path, args.bg)
        tiles.append(tile)
        had_alpha.append(alpha)

    aspect = tiles[0].shape[0] / tiles[0].shape[1]
    height = args.width * aspect * rows / cols
    # Room for the captions, which sit under each tile rather than over it.
    height += 0.35 * rows

    fig, axes = plt.subplots(rows, cols, figsize=(args.width, height))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(tiles[i])
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else os.path.basename(args.images[i]))
        ax.set_title(label, fontsize=11, pad=6)

    if args.title:
        fig.suptitle(args.title, fontsize=14)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi,
                facecolor=DARK if args.bg == "dark" else "white")
    plt.close(fig)

    print(f"[sheet] wrote {args.out} ({rows}x{cols}, {n} images, bg={args.bg})")
    if not all(had_alpha):
        opaque = [os.path.basename(p) for p, a in zip(args.images, had_alpha) if not a]
        print("[sheet] WARNING: these images have no alpha channel, so a transparent "
              "background was never produced: " + ", ".join(opaque))


if __name__ == "__main__":
    main()
