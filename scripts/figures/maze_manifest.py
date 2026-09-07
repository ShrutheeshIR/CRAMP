#!/usr/bin/env python3
"""Classify the 124 maze planks by structural role, and map each to its meshcat path.

    python3 scripts/figures/maze_manifest.py --selftest
    python3 scripts/figures/maze_manifest.py --json > planks.json

This is the single definition of "what is a lid", consumed three ways: by
export_maze_scene.py (which writes the manifest alongside the meshcat HTML), by
render_still.py (which computes the cutaway lid set from it), and by
figures/maze_schematic.py (which draws the maze in 2D from it). Keeping one
definition is the point -- a lid rule that disagreed between the 3D render and
the 2D schematic would produce two figures of the same maze that don't match.

The maze (fr3_trajopt/resources/environments/maze_cuboids.json) is 124
axis-aligned boxes, every rotation exactly zero, forming a flat tray:

    role     count  dz      z (raw)   what it is
    base       1    0.015   -0.055    the full-footprint floor plate
    border     2    0.115   -0.005    the two full-height outer walls
    wall      68    0.095   +-0.005   the interior maze partitions, 1.5cm thick
    floor     28    0.015   -0.045    thin plates just above the base plate
    lid       25    0.015   +0.045    thin plates capping channels from above

The lids are why this module exists: they cap the channels the marker traces, so
a render that keeps them shows an opaque grey top and nothing of the maze. See
blender_common.apply_lid_variant.

Classification runs on the RAW JSON values, before scene.MAZE_OFFSET_XYZ is
applied, so the rule is independent of where the board is placed in the world.

Plain `python3` -- stdlib only, no pydrake and no bpy, so it is importable from
the Drake exporter, the driver, and matplotlib alike.
"""

import argparse
import json
import os
import sys

ROLE_BASE = "base"
ROLE_BORDER = "border"
ROLE_WALL = "wall"
ROLE_FLOOR = "floor"
ROLE_LID = "lid"

ROLES = (ROLE_BASE, ROLE_BORDER, ROLE_WALL, ROLE_FLOOR, ROLE_LID)

# The split this maze must produce. Asserted rather than merely reported: if the
# maze JSON is ever regenerated upstream and the structure changes, every
# downstream lid decision silently becomes wrong, and a count mismatch is the
# cheapest possible place to catch that.
EXPECTED_COUNTS = {
    ROLE_BASE: 1,
    ROLE_BORDER: 2,
    ROLE_WALL: 68,
    ROLE_FLOOR: 28,
    ROLE_LID: 25,
}
EXPECTED_TOTAL = 124

# Default locations, resolved from this file so they work from any CWD (the same
# reason fr3_trajopt/common.py derives its paths from __file__).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MAZE_JSON = os.path.join(
    _REPO_ROOT, "fr3_trajopt", "resources", "environments", "maze_cuboids.json")

# Must match fr3_trajopt/scene.py's MAZE_OFFSET_XYZ. Duplicated rather than
# imported because this module must stay importable without pydrake (scene.py
# imports pydrake.all at module level). Kept honest by check_offset_matches().
MAZE_OFFSET_XYZ = (0.285 * 2, 0.0, 0.05)

# Thickness of the thin plates (base/floor/lid) versus the standing partitions.
_THIN_DZ = 0.015
_WALL_DZ = 0.095
_BORDER_DZ = 0.115
_TOL = 1e-6


def classify(box):
    """Return the structural role of one raw maze-JSON box.

    Thickness separates the standing geometry from the thin plates; among the
    thin plates, raw z separates the base plate (-0.055) from the in-channel
    floor plates (-0.045) from the lids (+0.045).
    """
    dz = float(box["dz"])
    z = float(box["z"])

    if abs(dz - _BORDER_DZ) < _TOL:
        return ROLE_BORDER
    if abs(dz - _WALL_DZ) < _TOL:
        return ROLE_WALL
    if abs(dz - _THIN_DZ) < _TOL:
        if z < -0.050:
            return ROLE_BASE
        if z < 0.0:
            return ROLE_FLOOR
        return ROLE_LID

    raise ValueError(
        f"{box.get('name', '<unnamed>')}: dz={dz} matches no known plank role "
        f"(expected {_THIN_DZ}, {_WALL_DZ}, or {_BORDER_DZ}). The maze JSON's "
        "structure has changed; update EXPECTED_COUNTS and this rule together.")


def meshcat_visual_path(name):
    """Meshcat path of a plank's illustration geometry after Drake publishes it.

    Maze boxes are registered on `plant.world_body()` (scene.py::_add_obstacle),
    and MeshcatVisualizer appends no frame segment for the world frame, so the
    path is the visualizer prefix plus the geometry name directly -- NOT the
    `<prefix>/<model>/<link>/...` shape the robot's own geometry gets.
    """
    return f"/drake/visual/{name}_visual"


def meshcat_collision_path(name):
    """Meshcat path of a plank's proximity geometry (prefix "collision" per
    scene.py's col_viz_params). Only needed to recognise and delete it."""
    return f"/drake/collision/{name}_collision"


def load_planks(json_path=DEFAULT_MAZE_JSON, offset=MAZE_OFFSET_XYZ):
    """Parse the maze JSON into per-plank records with world-frame geometry.

    Returns a list of dicts: name, role, size (full extents), center (world),
    aabb_min / aabb_max (world), meshcat_visual_path, meshcat_collision_path.
    Sizes are FULL extents, matching Drake's Box() and scene.py's SceneBox.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    ox, oy, oz = offset
    planks = []
    for box in raw:
        role = classify(box)
        size = (float(box["dx"]), float(box["dy"]), float(box["dz"]))
        center = (float(box["x"]) + ox, float(box["y"]) + oy, float(box["z"]) + oz)
        half = tuple(s / 2.0 for s in size)
        planks.append({
            "name": box["name"],
            "role": role,
            "size": size,
            "center": center,
            "aabb_min": tuple(c - h for c, h in zip(center, half)),
            "aabb_max": tuple(c + h for c, h in zip(center, half)),
            "meshcat_visual_path": meshcat_visual_path(box["name"]),
            "meshcat_collision_path": meshcat_collision_path(box["name"]),
        })
    return planks


def role_counts(planks):
    """Count planks per role, always reporting every role (0 included)."""
    counts = {role: 0 for role in ROLES}
    for plank in planks:
        counts[plank["role"]] += 1
    return counts


def board_aabb(planks):
    """World-frame (min_xyz, max_xyz) of the whole board."""
    mins = tuple(min(p["aabb_min"][i] for p in planks) for i in range(3))
    maxs = tuple(max(p["aabb_max"][i] for p in planks) for i in range(3))
    return mins, maxs


def planks_by_role(planks, role):
    return [p for p in planks if p["role"] == role]


def _segments_intersect_aabb(points, lo, hi):
    """True if any segment of an xy polyline enters the xy box [lo, hi].

    Sampled rather than solved analytically: the tip path is already a dense
    time sample (thousands of points) and the boxes are far larger than the
    spacing, so testing endpoints plus a few interpolants per segment cannot
    miss a lid the path actually crosses. Exact segment-box clipping would be
    more code for no practical difference here.
    """
    (lo_x, lo_y), (hi_x, hi_y) = lo, hi
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        if lo_x <= x <= hi_x and lo_y <= y <= hi_y:
            return True
        if i + 1 < len(points):
            nx, ny = points[i + 1][0], points[i + 1][1]
            for k in range(1, 4):
                t = k / 4.0
                sx, sy = x + (nx - x) * t, y + (ny - y) * t
                if lo_x <= sx <= hi_x and lo_y <= sy <= hi_y:
                    return True
    return False


def select_cutaway_lids(planks, tip_xy, margin=0.02):
    """Names of the lids that occlude the traced path, for the cutaway variant.

    A lid is selected if its world xy footprint, inflated by `margin` metres,
    contains any point of the tip polyline. Computed here (driver-side, in plain
    geometry) rather than inside Blender so the decision is printable, testable,
    and passed to the renderer as an explicit list of names.
    """
    if not tip_xy:
        return []
    selected = []
    for plank in planks_by_role(planks, ROLE_LID):
        lo = (plank["aabb_min"][0] - margin, plank["aabb_min"][1] - margin)
        hi = (plank["aabb_max"][0] + margin, plank["aabb_max"][1] + margin)
        if _segments_intersect_aabb(tip_xy, lo, hi):
            selected.append(plank["name"])
    return selected


def manifest(json_path=DEFAULT_MAZE_JSON, offset=MAZE_OFFSET_XYZ):
    """The full manifest dict written to planks.json and read by every consumer."""
    planks = load_planks(json_path, offset)
    lo, hi = board_aabb(planks)
    return {
        "source": os.path.relpath(json_path, _REPO_ROOT),
        "offset_xyz": list(offset),
        "counts": role_counts(planks),
        "total": len(planks),
        "board_aabb_min": list(lo),
        "board_aabb_max": list(hi),
        "planks": planks,
    }


def check_offset_matches():
    """Warn if scene.py's MAZE_OFFSET_XYZ has drifted from the copy here.

    Text-scraped instead of imported: importing scene.py pulls in pydrake, which
    would break this module's no-heavy-dependencies contract. Best effort -- a
    parse failure is silent, since this is a guard, not a gate.
    """
    scene_py = os.path.join(_REPO_ROOT, "fr3_trajopt", "scene.py")
    try:
        for line in open(scene_py):
            if line.startswith("MAZE_OFFSET_XYZ"):
                literal = line.split("=", 1)[1].strip()
                if tuple(eval(literal)) != tuple(MAZE_OFFSET_XYZ):  # noqa: S307
                    print(f"[manifest] WARNING: scene.py MAZE_OFFSET_XYZ = {literal}, "
                          f"but maze_manifest.py has {MAZE_OFFSET_XYZ}. They must match.")
                    return False
                return True
    except Exception:
        pass
    return True


def selftest(json_path=DEFAULT_MAZE_JSON):
    """Assert the expected plank split and print the board's world extent."""
    planks = load_planks(json_path)
    counts = role_counts(planks)
    lo, hi = board_aabb(planks)

    print(f"[manifest] {len(planks)} planks from {os.path.relpath(json_path, _REPO_ROOT)}")
    print("[manifest] " + ", ".join(f"{role} {counts[role]}" for role in ROLES))
    print(f"[manifest] board world aabb "
          f"x [{lo[0]:.4f}, {hi[0]:.4f}]  "
          f"y [{lo[1]:.4f}, {hi[1]:.4f}]  "
          f"z [{lo[2]:.4f}, {hi[2]:.4f}]")
    print(f"[manifest] board size "
          f"{hi[0] - lo[0]:.3f} x {hi[1] - lo[1]:.3f} x {hi[2] - lo[2]:.3f} m")

    # The top faces of the lids, the tall interior walls and the border walls are
    # exactly coplanar. Reported because exactly-coincident faces render as black
    # patches under Cycles' CPU/Embree backend; blender_common nudges the lids to
    # break the tie, and this is where the number comes from.
    top_z = max(p["aabb_max"][2] for p in planks)
    coplanar = sum(1 for p in planks if abs(p["aabb_max"][2] - top_z) < _TOL)
    print(f"[manifest] {coplanar} planks share the top plane z={top_z:.4f} "
          "(z-fighting risk; see --z-fight-nudge)")

    ok = check_offset_matches()
    if counts != EXPECTED_COUNTS:
        print(f"[manifest] FAIL: counts {counts} != expected {EXPECTED_COUNTS}")
        ok = False
    if len(planks) != EXPECTED_TOTAL:
        print(f"[manifest] FAIL: {len(planks)} planks, expected {EXPECTED_TOTAL}")
        ok = False
    names = [p["name"] for p in planks]
    if len(set(names)) != len(names):
        print("[manifest] FAIL: duplicate plank names")
        ok = False

    print("[manifest] selftest " + ("ok" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maze-json", default=DEFAULT_MAZE_JSON,
                    help="maze cuboid JSON (default: fr3_trajopt's)")
    ap.add_argument("--selftest", action="store_true",
                    help="assert the expected plank split and print the board extent")
    ap.add_argument("--json", action="store_true", help="write the manifest to stdout")
    args = ap.parse_args()

    if args.json:
        json.dump(manifest(args.maze_json), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    sys.exit(0 if selftest(args.maze_json) else 1)


if __name__ == "__main__":
    main()
