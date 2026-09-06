"""
Illustrative-only: sample end-effector targets along a straight line and
drive the 2-link toy arm's pose via closed-form (analytic) 2-link planar
IK, instead of hand-picking joint angles per pose.

This is purely for the figure -- it has nothing to do with a real robot's
IK; it just lets "pose along a line" be the input to the drawing instead
of "joint angles", matching the methodology-figure idea (sample poses ->
IK -> arm configuration per sample).

Usage:
    python ik_line_demo.py
Produces ik_line_demo.png / .svg in this directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from robot_arm_fig import (
    draw_arm, draw_target, LINK_LENGTHS as _DEFAULT_LENGTHS, PALETTE,
)

LINK_LENGTHS = _DEFAULT_LENGTHS  # (L1, L2), reuse the same arm proportions


# ----------------------------------------------------------------------
# analytic 2-link planar IK
# ----------------------------------------------------------------------
def analytic_2link_ik(base, target, lengths, elbow_up=True):
    """
    Closed-form IK for a 2-link planar arm.

    base:    (x, y) position of the shoulder joint.
    target:  (x, y) desired end-effector (wrist) position.
    lengths: (L1, L2) link lengths.
    elbow_up: selects one of the two IK solution branches.

    Returns (theta1_rel, theta2_rel) in the same convention used by
    forward_kinematics in robot_arm_fig.py (theta1_rel is the bend away
    from "straight up", theta2_rel is the elbow bend relative to link 1),
    or None if the target is unreachable (outside [|L1-L2|, L1+L2]).
    """
    L1, L2 = lengths
    dx = target[0] - base[0]
    dy = target[1] - base[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)

    if r > L1 + L2 or r < abs(L1 - L2):
        return None  # unreachable

    cos_theta2 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    if elbow_up:
        theta2 = -theta2

    theta_target = np.arctan2(dy, dx)
    theta1 = theta_target - np.arctan2(
        L2 * np.sin(theta2), L1 + L2 * np.cos(theta2)
    )

    theta1_rel = theta1 - np.pi / 2  # bend away from "straight up"
    theta2_rel = theta2              # already relative to link 1
    return theta1_rel, theta2_rel


# ----------------------------------------------------------------------
# SE(2)/line sampling (our toy arm has no wrist DOF, so a "line in SE(3)"
# reduces to a line of end-effector *positions* here)
# ----------------------------------------------------------------------
def sample_line(p_start, p_end, n):
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p_start[None, :] * (1 - t) + p_end[None, :] * t


# ----------------------------------------------------------------------
# demo
# ----------------------------------------------------------------------
def main():
    base = (0.0, 0.0)
    n_samples = 4
    targets = sample_line((0.9, 1.5), (-0.3, 2.0), n_samples)

    fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 4.2))
    for i, (ax, target) in enumerate(zip(axes, targets)):
        color = PALETTE[i % len(PALETTE)]
        solution = analytic_2link_ik(base, target, LINK_LENGTHS,
                                      elbow_up=True)
        if solution is None:
            ax.set_title("unreachable")
            ax.axis("off")
            continue

        draw_arm(ax, base, solution, LINK_LENGTHS, color)
        draw_target(ax, target)

        ax.set_xlim(-2.0, 2.2)
        ax.set_ylim(-0.5, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig("ik_line_demo.png", dpi=300, transparent=True)
    fig.savefig("ik_line_demo.svg", transparent=True)
    print("Saved ik_line_demo.png / ik_line_demo.svg")


if __name__ == "__main__":
    main()
