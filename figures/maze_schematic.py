"""
Top-down schematic of the FR3 marker maze, with the traced marker path
overlaid when one is available.

The 3D renders under scripts/figures/ show the board as it looks; this
shows it as it is -- every wall, every channel, and the path through
them, with nothing hidden behind a lid or foreshortened by perspective.

Usage:
    python maze_schematic.py
    python maze_schematic.py --tip-path ../out/figures/scene/tip_path.json
    python maze_schematic.py --show-lids

Produces maze_schematic.png and maze_schematic.svg in this directory.

Reads the maze geometry through scripts/figures/maze_manifest.py, the same
module the 3D pipeline classifies planks with, so the two figures cannot
disagree about what is a wall and what is a lid. The marker path arrives as
JSON from export_maze_scene.py, so this script needs neither pydrake nor
Blender and stays as standalone as the rest of figures/.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "figures"))

import maze_manifest  # noqa: E402

from robot_arm_fig import PALETTE  # noqa: E402

# ----------------------------------------------------------------------
# style constants
# ----------------------------------------------------------------------
WALL_COLOR = "#5c6068"       # interior partitions: the maze itself
BORDER_COLOR = "#3f434a"     # the two full-height outer walls
BASE_COLOR = "#eceef1"       # the board's footprint, as a ground tone
LID_COLOR = "#b9bec7"        # only drawn under --show-lids

PATH_COLOR = PALETTE[0]      # "#1b9e77", the first Dark2 hue
START_COLOR = PALETTE[1]     # "#d95f02"
GOAL_COLOR = PALETTE[2]      # "#7570b3"

PATH_WIDTH = 2.4
MARKER_SIZE = 9.0

# The board is 0.575 m across by 1.415 m along, so plotting world x on the
# horizontal axis would give a figure three times taller than it is wide.
# World y goes across the page instead, which puts the long axis of the board
# along the long axis of the figure. --transpose swaps them back.
FIGURE_WIDTH = 11.0


def board_patches(planks, show_lids, show_base):
    """(patch, zorder) for every plank that should be drawn."""
    patches = []
    for plank in planks:
        role = plank["role"]
        if role == "base":
            if not show_base:
                continue
            color, z = BASE_COLOR, 1
        elif role == "floor":
            # The in-channel floor plates tile the same footprint as the
            # channels themselves; drawing them adds a second flat tone that
            # reads as structure where there is none.
            continue
        elif role == "lid":
            if not show_lids:
                continue
            color, z = LID_COLOR, 3
        elif role == "border":
            color, z = BORDER_COLOR, 4
        else:
            color, z = WALL_COLOR, 4
        patches.append((plank, color, z))
    return patches


def draw_board(ax, planks, show_lids, show_base, transpose):
    """Draw every plank as a flat rectangle, in the figures/ house style."""
    for plank, color, z in board_patches(planks, show_lids, show_base):
        x0, y0 = plank["aabb_min"][0], plank["aabb_min"][1]
        dx = plank["aabb_max"][0] - x0
        dy = plank["aabb_max"][1] - y0
        # Plot coordinates: world y horizontal, world x vertical, unless
        # --transpose puts them back the natural way round.
        if transpose:
            rect = Rectangle((x0, y0), dx, dy)
        else:
            rect = Rectangle((y0, x0), dy, dx)
        rect.set_facecolor(color)
        rect.set_edgecolor("none")
        rect.set_alpha(0.35 if plank["role"] == "lid" else 1.0)
        rect.set_zorder(z)
        ax.add_patch(rect)


def draw_path(ax, xyz, transpose, show_endpoints):
    """Overlay the marker tip's xy track."""
    xs = [p[0] for p in xyz]
    ys = [p[1] for p in xyz]
    hx, vx = (xs, ys) if transpose else (ys, xs)

    ax.plot(hx, vx, color=PATH_COLOR, linewidth=PATH_WIDTH,
            solid_capstyle="round", solid_joinstyle="round", zorder=6)
    if show_endpoints:
        ax.plot(hx[0], vx[0], "o", color=START_COLOR, markersize=MARKER_SIZE,
                markeredgecolor="none", zorder=7)
        ax.plot(hx[-1], vx[-1], "o", color=GOAL_COLOR, markersize=MARKER_SIZE,
                markeredgecolor="none", zorder=7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tip-path", default=None,
                    help="tip_path.json from export_maze_scene.py; without it the "
                         "board is drawn on its own")
    ap.add_argument("--show-lids", action="store_true",
                    help="draw the 25 lid plates translucently over the channels they "
                         "cap -- which is also the argument for culling them in 3D")
    ap.add_argument("--show-base", action="store_true",
                    help="draw the board's base plate as a background tone")
    ap.add_argument("--no-endpoints", action="store_true",
                    help="omit the start and goal markers")
    ap.add_argument("--transpose", action="store_true",
                    help="world x horizontal instead of world y")
    ap.add_argument("--out", default=None,
                    help="output basename (default: maze_schematic, beside this script)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    planks = maze_manifest.load_planks()
    counts = maze_manifest.role_counts(planks)
    lo, hi = maze_manifest.board_aabb(planks)
    print("[schematic] " + ", ".join(f"{r} {counts[r]}" for r in maze_manifest.ROLES))

    xyz = None
    if args.tip_path:
        if not os.path.isfile(args.tip_path):
            raise SystemExit(f"[schematic] no such tip path: {args.tip_path}")
        with open(args.tip_path, "r") as f:
            xyz = json.load(f)["xyz"]
        print(f"[schematic] {len(xyz)} tip samples from {args.tip_path}")
    else:
        print("[schematic] no --tip-path given; drawing the board without a path")

    span_x, span_y = hi[0] - lo[0], hi[1] - lo[1]
    aspect = (span_x / span_y) if not args.transpose else (span_y / span_x)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * aspect))

    draw_board(ax, planks, args.show_lids, args.show_base, args.transpose)
    if xyz:
        draw_path(ax, xyz, args.transpose, not args.no_endpoints)

    pad = 0.01
    if args.transpose:
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
    else:
        ax.set_xlim(lo[1] - pad, hi[1] + pad)
        ax.set_ylim(lo[0] - pad, hi[0] + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    base = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "maze_schematic")
    fig.savefig(base + ".png", dpi=args.dpi, transparent=True)
    fig.savefig(base + ".svg", transparent=True)
    print(f"[schematic] wrote {base}.png and {base}.svg")


if __name__ == "__main__":
    main()
