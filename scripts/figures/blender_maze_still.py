"""Render a still of the FR3 maze scene. Runs INSIDE Blender.

Not launched directly -- render_still.py builds the command line. To run it by
hand:

    ~/opt/blender-5.0.1-linux-x64/blender --background \
        --python scripts/figures/blender_maze_still.py -- \
        --html out/figures/scene/static.html \
        --manifest out/figures/scene/planks.json \
        --out out/figures/static.png

Two modes:

    --mode static   one configuration, held. Needs no trajectory.
    --mode ghosts   chronophotography: N poses along the motion, ghosted, on one
                    frame. Needs an animated HTML.

Prints MAZE_STILL_DONE on success. The driver requires that line rather than
trusting the exit status, because Blender exits 0 on a good many things that are
not a successful render.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

import blender_common as bc  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    bc.add_common_arguments(ap)
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--mode", choices=("static", "ghosts"), default="static")
    ap.add_argument("--frame", type=int, default=0,
                    help="--mode static: which animation frame to hold (default: 0)")

    ap.add_argument("--n-poses", type=int, default=6, help="--mode ghosts: pose count")
    ap.add_argument("--t-start", type=float, default=0.0,
                    help="fraction of the recording the sweep starts at")
    ap.add_argument("--t-end", type=float, default=1.0,
                    help="fraction of the recording the sweep ends at")
    ap.add_argument("--spacing", choices=("arclength", "time"), default="arclength",
                    help="how poses are spaced along the motion (default: arclength, "
                         "which does not bunch at a retimed trajectory's slow ends)")
    ap.add_argument("--mid-bias", type=float, default=0.0,
                    help="push poses toward the middle of the interval; 0 is even")
    ap.add_argument("--pose-nudge", type=float, nargs="*", default=None,
                    help="per-pose shifts along the path, for separating a stacked pair")
    ap.add_argument("--drop-poses", type=int, nargs="*", default=None,
                    help="pose indices to remove entirely")
    ap.add_argument("--alpha", type=float, default=0.45, help="ghost alpha (default: 0.45)")
    ap.add_argument("--endpoint-alpha", type=float, default=None,
                    help="alpha for the first and last pose; off by default because a "
                         "mixed ramp reads as an artifact rather than as a cue")
    ap.add_argument("--static-tol", type=float, default=1e-4,
                    help="travel below which a link counts as welded and is drawn once")
    ap.add_argument("--ghost-shadows", choices=("none", "last", "all"), default="last",
                    help="which poses cast shadows (default: last -- every pose casting "
                         "dims the ones behind it and reads as uneven opacity)")
    ap.add_argument("--trace-progress", type=float, default=1.0,
                    help="fraction of the trace to draw, for a partial path")

    ap.add_argument("--frame-subject", default="scene",
                    choices=("scene", "board", "robot", "robot+board"),
                    help="what the camera frames (default: scene, i.e. everything)")
    return ap.parse_args(bc.script_args())


def framing_subject(parts, ghosts, welded, choice):
    """The point set the camera is fitted to.

    Separated from the lighting subject because the two want different things:
    framing usually has to include the whole 1.4 m board, while the lighting rig
    should stay sized to the arm, or the exposure collapses as the framing widens.
    """
    board = parts["planks"]
    robot = ghosts + welded if ghosts else parts["robot"]
    if choice == "board":
        return board
    if choice == "robot":
        return robot
    return board + robot


def main():
    args = parse_args()

    objs = bc.import_scene(args.html, target_fps=30.0)
    parts = bc.classify(objs, args.manifest, expect_collision=args.expect_collision)

    # Before restyle_maze and bake_ghosts: this walks obj.data.materials and
    # would otherwise stamp over the role colours or miss the ghost copies.
    bc.improve_scene_quality()
    bc.restyle_maze(parts, args)
    bc.apply_lid_variant(parts, args)
    trace = bc.add_tip_trace(args)

    ghosts, welded = [], []
    if args.mode == "ghosts":
        f0, f1, window = bc.frame_window(args.t_start, args.t_end)
        if f1 <= f0:
            raise SystemExit(
                f"[ghosts] the scene has a single frame ({f0}..{f1}), so there is no "
                "motion to sweep. Export with a --trajectory, and check that the export "
                "reported more than one recorded frame.")
        track = bc.scan_motion(parts["robot"], window)
        truly_moving, welded = bc.split_static(parts["robot"], track, args.static_tol)
        frames = bc.sample_frames(window, track, truly_moving, args)
        ghosts = bc.bake_ghosts(truly_moving, frames, args)
        # bake_ghosts deletes the originals it copied from, so anything still
        # holding them would be touching removed objects. Same hazard as culling
        # lids; see blender_common._delete_objects.
        parts["robot"] = ghosts + welded
    else:
        scene = bpy.context.scene
        frame = max(scene.frame_start, min(args.frame, scene.frame_end))
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        print(f"[static] holding frame {frame} of {scene.frame_start}..{scene.frame_end}")

    # Resolution first: the camera fit reads the aspect ratio.
    bc.configure_output(args)
    bc.configure_render(args)

    subject = framing_subject(parts, ghosts, welded, args.frame_subject)
    light_subject = (ghosts + welded) if ghosts else parts["robot"]
    center, radius, azimuth, target, cam = bc.setup_camera(args, subject, light_subject)
    bc.setup_studio_lighting(args, center, radius, azimuth, target)
    bc.setup_world(args)

    if args.camera_json:
        bc.write_camera_json(cam, args.camera_json)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.context.scene.render.filepath = os.path.abspath(args.out)
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {args.out}")
    print("MAZE_STILL_DONE")


if __name__ == "__main__":
    main()
