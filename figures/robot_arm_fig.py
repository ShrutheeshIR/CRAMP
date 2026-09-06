"""
Illustrative 2-link planar robot arm with a rectangular parallel-jaw
gripper, flat mechanical style (sharp rectangular links + pivot discs).

Usage:
    python robot_arm_fig.py

Produces robot_arm_fig.png and robot_arm_fig.svg in this directory.
Tweak ARM_POSES / PALETTE / sizes below to reuse for other figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon

# ----------------------------------------------------------------------
# style constants
# ----------------------------------------------------------------------
LINK_HALF_WIDTH = 0.11       # half-width of arm link rectangles
BASE_HALF_WIDTH = 0.14       # half-width (thickness) of the base rail
JOINT_RADIUS = 0.145         # pivot disc radius at each rotational joint

STUB_LEN = 0.28              # stub length before the gripper crossbar
STUB_HALF_WIDTH = 0.075
CROSSBAR_HALF_WIDTH = 0.06   # half-thickness of the gripper crossbar
FINGER_LEN = 0.45            # length of each parallel finger
FINGER_HALF_WIDTH = 0.06
JAW_HALF_GAP = 0.28          # half-distance between the two fingers

BASE_RAIL_HALF = 0.55        # half-length of the base rail segment


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def forward_kinematics(base, angles, lengths):
    """
    base: (x, y) position of the first joint.
    angles: sequence of relative bend angles (radians), first one measured
        from "straight up", each subsequent one relative to the previous
        link's direction.
    lengths: sequence of link lengths, same length as angles.

    Returns (joint_points, final_direction_angle).
    joint_points includes the base as the first point.
    """
    x, y = base
    heading = np.pi / 2  # start pointing straight up
    pts = [(x, y)]
    for theta, L in zip(angles, lengths):
        heading += theta
        x = x + L * np.cos(heading)
        y = y + L * np.sin(heading)
        pts.append((x, y))
    return pts, heading


def rect_corners(p0, p1, half_width):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-9:
        return None
    dirv = d / length
    normal = np.array([-dirv[1], dirv[0]])
    return np.array([
        p0 + normal * half_width,
        p1 + normal * half_width,
        p1 - normal * half_width,
        p0 - normal * half_width,
    ])


# ----------------------------------------------------------------------
# drawing primitives
# ----------------------------------------------------------------------
def draw_rect_segment(ax, p0, p1, color, half_width, zorder=2):
    corners = rect_corners(p0, p1, half_width)
    if corners is None:
        return
    ax.add_patch(Polygon(corners, closed=True, facecolor=color,
                          edgecolor="none", zorder=zorder))


def draw_joint(ax, center, color, radius=JOINT_RADIUS, zorder=3):
    ax.add_patch(Circle(center, radius, facecolor=color, edgecolor="none",
                         zorder=zorder))


def draw_base_rail(ax, base, color, half_len=BASE_RAIL_HALF,
                    half_width=BASE_HALF_WIDTH, zorder=2):
    x, y = base
    draw_rect_segment(ax, (x - half_len, y), (x + half_len, y), color,
                       half_width, zorder=zorder)


def draw_gripper(ax, wrist, heading, color, zorder=2):
    """Rectangular parallel-jaw gripper: stub -> crossbar -> two fingers,
    all straight rectangular segments, oriented along `heading`."""
    wrist = np.asarray(wrist, dtype=float)
    d = np.array([np.cos(heading), np.sin(heading)])   # along-arm direction
    n = np.array([-np.sin(heading), np.cos(heading)])  # left-hand normal

    stub_tip = wrist + d * STUB_LEN
    draw_rect_segment(ax, wrist, stub_tip, color, STUB_HALF_WIDTH, zorder)

    left_root = stub_tip + n * JAW_HALF_GAP
    right_root = stub_tip - n * JAW_HALF_GAP
    draw_rect_segment(ax, left_root, right_root, color,
                       CROSSBAR_HALF_WIDTH, zorder)

    left_tip = left_root + d * FINGER_LEN
    right_tip = right_root + d * FINGER_LEN
    draw_rect_segment(ax, left_root, left_tip, color, FINGER_HALF_WIDTH,
                       zorder)
    draw_rect_segment(ax, right_root, right_tip, color, FINGER_HALF_WIDTH,
                       zorder)

    # small pivot discs to keep the stub/crossbar/finger seams clean
    draw_joint(ax, stub_tip, color, radius=STUB_HALF_WIDTH, zorder=zorder + 1)
    draw_joint(ax, left_root, color, radius=FINGER_HALF_WIDTH,
               zorder=zorder + 1)
    draw_joint(ax, right_root, color, radius=FINGER_HALF_WIDTH,
               zorder=zorder + 1)


def draw_target(ax, pos, color="#d7263d", w=0.22, h=0.55, zorder=4):
    """Small rectangular 'object' marker."""
    x, y = pos
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.0,rounding_size=0.02",
                          linewidth=0, facecolor=color, zorder=zorder)
    ax.add_patch(box)


def draw_highlight_circle(ax, center, radius=0.55, color="#d7263d",
                           lw=2.5, zorder=5):
    circ = Circle(center, radius, fill=False, edgecolor=color, lw=lw,
                  zorder=zorder)
    ax.add_patch(circ)


def draw_arm(ax, base, angles, lengths, color, draw_rail=True):
    """Draw a full arm: base rail, rectangular links + pivot discs,
    rectangular gripper. Returns joint points and final heading."""
    pts, heading = forward_kinematics(base, angles, lengths)
    if draw_rail:
        draw_base_rail(ax, base, color)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw_rect_segment(ax, p0, p1, color, LINK_HALF_WIDTH)
    for joint_pt in pts[1:-1]:
        draw_joint(ax, joint_pt, color)
    draw_joint(ax, pts[0], color, radius=BASE_HALF_WIDTH)
    draw_gripper(ax, pts[-1], heading, color)
    return pts, heading


# ----------------------------------------------------------------------
# example figure: four arm poses side by side.
# Palette: ColorBrewer "Dark2" -- a qualitative, colorblind-safe set
# chosen specifically for strong contrast on a white background.
# ----------------------------------------------------------------------
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

# (theta1, theta2) relative bend angles in radians for each lane
ARM_POSES = [
    (-0.35, -0.55),
    (-0.15, -0.85),
    (0.05, -0.15),
    (0.30, -0.95),
]
LINK_LENGTHS = (1.0, 1.1)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2))
    for ax, color, angles in zip(axes, PALETTE, ARM_POSES):
        base = (0.0, 0.0)
        pts, heading = draw_arm(ax, base, angles, LINK_LENGTHS, color)

        # example target block near the arm, and a highlight circle
        # around the elbow -- delete/adjust as needed per-panel
        draw_target(ax, (0.75, 1.35))
        draw_highlight_circle(ax, pts[1], radius=0.5)

        ax.set_xlim(-2.0, 2.2)
        ax.set_ylim(-0.5, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("robot_arm_fig.png", dpi=300, transparent=True)
    fig.savefig("robot_arm_fig.svg", transparent=True)
    print("Saved robot_arm_fig.png / robot_arm_fig.svg")


if __name__ == "__main__":
    main()
