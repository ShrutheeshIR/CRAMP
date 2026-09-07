#!/usr/bin/env python3
"""Export the FR3-marker maze scene from Drake to a meshcat HTML file for Blender.

    # static scene, no trajectory needed
    python3 scripts/figures/export_maze_scene.py

    # animated, from a VAMP ambient path
    python3 scripts/figures/export_maze_scene.py --trajectory path/to/problem_0.txt

    # static, with the marker placed in the maze by IK
    python3 scripts/figures/export_maze_scene.py --ik-tip 0.55,0.0

This is the only file in the figure pipeline that imports pydrake. It builds the
scene through fr3_trajopt/scene.py unchanged, optionally records a TOPPRA-retimed
trajectory into a meshcat animation, and writes four artifacts consumed by the
Blender and matplotlib halves:

    <name>.html       meshcat.StaticHtml() -- geometry plus any recorded animation
    planks.json       maze_manifest.manifest() -- plank roles and meshcat paths
    tip_path.json     marker tip world positions over time (only with --trajectory)
    scene_meta.json   provenance: source path + hash, timing, versions, verification

Two things here are easy to get wrong and produce output that looks fine:

1. `context.SetTime(t)` must be called before each publish while recording.
   MeshcatVisualizer stamps every transform with the context's time, so without
   it every frame lands at t=0 and the animation collapses to a single pose --
   which then renders as a chronophotography figure with N identical ghosts.
   None of the existing fr3_trajopt/notebooks playback loops set the time (they
   do not record), so copying one of them straight into a recorder is the trap.

2. `meshcat.Delete("collision")` must run before StaticHtml(). scene.py attaches
   a second visualizer at Role.kProximity with visible_by_default=False; that
   geometry is registered when the diagram is BUILT, and StaticHtml serialises
   hidden geometry just the same. The FR3 URDF is spherized, so a leak imports
   into Blender as a string of collision spheres along every link.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# fr3_trajopt is a plain module directory, not an installed package -- same
# sys.path insert its own notebooks/ scripts use, for the same reason.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "fr3_trajopt"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import maze_manifest  # noqa: E402

from pydrake.all import StartMeshcat  # noqa: E402

import scene as fr3_scene  # noqa: E402
from retiming import retime_ambient_path, verify_retimed_trajectory  # noqa: E402
from trajectory import load_ambient_path  # noqa: E402

# Standard Franka "ready" configuration -- the pose the arm parks in, elbow up
# and clear of the table. Used when no trajectory and no explicit q is given.
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# The marker body points along +X of the fr3_tip frame: fr3_tip_joint offsets the
# tip from fr3_marker by (0.08125, 0.0035, 0.003625) with zero rotation, so the
# long axis is X. Used by --ik-tip to aim the marker at the board.
MARKER_AXIS_IN_TIP = np.array([1.0, 0.0, 0.0])

# Top of the in-channel floor plates, in world z: the height the marker tip rides
# at while tracing. Derived from the maze manifest rather than hardcoded.
DEFAULT_OUT = os.path.join(_REPO_ROOT, "out", "figures", "scene")

MISSING_TRAJECTORY_HELP = """\
No trajectory file was given, and this mode needs one.

Expected: an "ambient path" text file -- one comma-separated 7-DOF joint
configuration per line, as written by VAMP's write_ambient_path. For example:

    vamp/resources/fr3_marker/maze_solver_benchmark_trajectories_python/problem_0.txt

Note that vamp/ is an UNINITIALIZED submodule in this checkout. If you expect
that file to exist, run:

    git submodule update --init --recursive

Be aware that the fr3_marker VAMP work is not present on any public branch of
CoMMALab/vamp (the pinned submodule commit is also gone upstream), so the
submodule alone may not produce it -- the trajectories likely have to come from
wherever they were originally generated.

No fallback trajectory will be invented: a figure of a made-up path would be
worse than no figure.
"""


def channel_floor_z(planks):
    """World z of the top face of the in-channel floor plates."""
    floors = maze_manifest.planks_by_role(planks, maze_manifest.ROLE_FLOOR)
    return max(p["aabb_max"][2] for p in floors)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tip_pose(plant, plant_context, q):
    """World pose of the marker tip at configuration q.

    The same forward-kinematics call trajopt._measure_tip_pose makes. Duplicated
    here as three lines rather than imported, so that fr3_trajopt/ stays
    untouched and this pipeline does not depend on trajopt.py -- whose smoothing
    is documented as not producing a usable result.
    """
    plant.SetPositions(plant_context, q)
    return plant.CalcRelativeTransform(
        plant_context, plant.world_frame(),
        plant.GetFrameByName(fr3_scene.FR3_TIP_FRAME))


def solve_tip_ik(plant, plant_context, xy, z, q_guess=HOME_Q, min_distance=0.0,
                 collision_checker=None):
    """Place the marker tip at (x, y, z) pointing straight down, via Drake IK.

    A convenience for static figures only, to put the marker somewhere in the
    maze when there is no trajectory to take a pose from. It constrains the tip
    and nothing else, so the other six links can and sometimes do end up
    intersecting the board.

    `min_distance` adds a clearance constraint that would prevent that, and it is
    OFF by default deliberately. Turning it on makes this a constrained planning
    problem: the marker is meant to sit inside a narrow channel, so it is
    legitimately within millimetres of the walls, and the useful clearance values
    are small, position-dependent, and infeasible over much of the board. Tuning
    that is planning work, not figure work.

    The right source of a good-looking configuration is a real planned
    trajectory. Until one is available, prefer the default home pose for static
    figures, or pass --q with a configuration you already trust. If a rendered
    pose intersects the board, change the pose rather than reaching for a solver.

    Returns q, or raises with what to try instead.
    """
    from pydrake.all import InverseKinematics, Solve

    ik = InverseKinematics(plant, plant_context)
    tip_frame = plant.GetFrameByName(fr3_scene.FR3_TIP_FRAME)
    world = plant.world_frame()

    target = np.array([xy[0], xy[1], z])
    tol = 0.002
    ik.AddPositionConstraint(tip_frame, np.zeros(3), world, target - tol, target + tol)
    # Marker axis (tip +X) pointing along world -Z, within ~3 degrees.
    ik.AddAngleBetweenVectorsConstraint(
        tip_frame, MARKER_AXIS_IN_TIP, world, np.array([0.0, 0.0, -1.0]), 0.0, 0.05)
    if min_distance > 0.0:
        ik.AddMinimumDistanceLowerBoundConstraint(min_distance, 0.05)

    prog = ik.prog()
    prog.SetInitialGuess(ik.q(), q_guess)
    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError(
            f"IK failed to place the marker tip at {target} pointing down "
            f"(solution_result={result.get_solution_result()}"
            + (f", with a {min_distance * 1000:.1f} mm clearance constraint"
               if min_distance > 0 else "") + "). "
            "Pick a different --ik-tip, drop --ik-min-distance if you set it, or pass "
            "--q with a configuration you already trust.")

    q = result.GetSolution(ik.q())
    if collision_checker is not None and not collision_checker.CheckConfigCollisionFree(q):
        print("[export] NOTE: this IK pose is in collision with the board, so the render "
              "will show the arm intersecting it. --ik-tip constrains only the marker "
              "tip. Use a real trajectory, or --q, for a pose that has to look right.")
    return q


def sample_tip_path(traj, plant, plant_context, num_samples):
    """(times, xyz) of the marker tip along a retimed trajectory."""
    ts = np.linspace(traj.start_time(), traj.end_time(), num_samples)
    xyz = np.array([tip_pose(plant, plant_context, traj.value(t).flatten()).translation()
                    for t in ts])
    return ts, xyz


def record_trajectory(meshcat, diagram, context, plant, plant_context, traj, fps):
    """Record a retimed trajectory into meshcat's animation. Returns frame count.

    `set_visualizations_while_recording=False` keeps the browser from being sent
    every frame as it is recorded (the recording is published in one go at the
    end); it does not affect what lands in the animation.
    """
    meshcat.StartRecording(frames_per_second=fps, set_visualizations_while_recording=False)

    duration = traj.end_time()
    # Half a frame of slop so the final configuration is always included rather
    # than being dropped by floating-point drift at the interval's end.
    times = np.arange(0.0, duration + 0.5 / fps, 1.0 / fps)
    for t in times:
        t = min(float(t), duration)
        context.SetTime(t)  # <-- MeshcatVisualizer timestamps from this. See module docstring.
        plant.SetPositions(plant_context, traj.value(t).flatten())
        diagram.ForcedPublish(context)

    meshcat.StopRecording()
    meshcat.PublishRecording()
    return len(times)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", help="VAMP ambient path .txt (one comma-separated "
                                         "7-DOF configuration per line)")
    ap.add_argument("--static", action="store_true",
                    help="with --trajectory, export a single held pose instead of the "
                         "animation")
    ap.add_argument("--pose", choices=("home", "zero", "first", "mid", "last"),
                    default="home",
                    help="which configuration a static export holds (default: home; "
                         "first/mid/last require --trajectory)")
    ap.add_argument("--q", help="explicit static configuration, 7 comma-separated joint "
                                "values; overrides --pose")
    ap.add_argument("--ik-tip", metavar="X,Y",
                    help="static only: solve IK to put the marker tip at world (x, y) on "
                         "the channel floor, pointing down")
    ap.add_argument("--ik-min-distance", type=float, default=0.0,
                    help="optional clearance in metres the IK solution must keep from the "
                         "board. OFF by default: see solve_tip_ik. Turning it on turns "
                         "placing the marker into a constrained planning problem, which "
                         "is not what this script is for")
    ap.add_argument("--ik-tip-gap", type=float, default=0.003,
                    help="metres to hold the tip above the channel floor (default: 0.003)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output directory (default: "
                                                       f"{os.path.relpath(DEFAULT_OUT, _REPO_ROOT)})")
    ap.add_argument("--name", help="output basename (default: the trajectory stem, or "
                                   "'static')")
    ap.add_argument("--fps", type=float, default=64.0,
                    help="meshcat recording rate (default: 64, Drake's own default; the "
                         "Blender importer interpolates DOWN from this, never up)")
    ap.add_argument("--no-retime", action="store_true",
                    help="record the raw waypoints at a fixed rate instead of "
                         "TOPPRA-retiming them")
    ap.add_argument("--tip-samples", type=int, default=2000,
                    help="samples along the tip path written to tip_path.json")
    ap.add_argument("--keep-collision", action="store_true",
                    help="do NOT delete the proximity geometry before export (debug only; "
                         "it will import into Blender as collision spheres)")
    ap.add_argument("--serve", action="store_true",
                    help="keep the Meshcat server alive after exporting so you can "
                         "inspect the scene in a browser")
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    needs_trajectory = args.pose in ("first", "mid", "last")
    if needs_trajectory and not args.trajectory:
        raise SystemExit(f"--pose {args.pose} requires --trajectory.\n\n"
                         + MISSING_TRAJECTORY_HELP)

    animate = bool(args.trajectory) and not args.static
    name = args.name or (
        os.path.splitext(os.path.basename(args.trajectory))[0] if args.trajectory
        else "static")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f"{name}.html")
    if os.path.exists(html_path) and not args.force:
        raise SystemExit(f"{html_path} exists; pass --force to overwrite.")

    planks = maze_manifest.load_planks()
    counts = maze_manifest.role_counts(planks)
    if counts != maze_manifest.EXPECTED_COUNTS:
        raise SystemExit(f"[export] maze plank counts {counts} != "
                         f"{maze_manifest.EXPECTED_COUNTS}; the maze JSON has changed and "
                         "maze_manifest.py needs updating before any figure is trusted.")
    floor_z = channel_floor_z(planks)
    print(f"[export] {len(planks)} planks: "
          + ", ".join(f"{r} {counts[r]}" for r in maze_manifest.ROLES))
    print(f"[export] channel floor z = {floor_z:.4f} m")

    meshcat = StartMeshcat()
    plant, collision_checker, diagram = fr3_scene.make_default_fr3_infrastructure(meshcat)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    print(f"[export] plant nq={plant.num_positions()} nv={plant.num_velocities()}")

    meta = {
        "name": name,
        "animated": animate,
        "maze_offset_xyz": list(maze_manifest.MAZE_OFFSET_XYZ),
        "channel_floor_z": floor_z,
        "board_aabb_min": list(maze_manifest.board_aabb(planks)[0]),
        "board_aabb_max": list(maze_manifest.board_aabb(planks)[1]),
        "tip_frame": fr3_scene.FR3_TIP_FRAME,
        "plank_counts": counts,
        "drake_path": _drake_path(),
        "pydrake_version": _pydrake_version(),
        "generated_by": "scripts/figures/export_maze_scene.py",
    }

    waypoints = None
    if args.trajectory:
        waypoints = load_ambient_path(args.trajectory)
        if waypoints.ndim != 2 or waypoints.shape[0] < 2:
            raise SystemExit(f"[export] {args.trajectory} parsed to shape "
                             f"{waypoints.shape}; need at least 2 waypoints.")
        if waypoints.shape[1] != plant.num_positions():
            raise SystemExit(f"[export] {args.trajectory} has {waypoints.shape[1]} columns "
                             f"but the plant has {plant.num_positions()} DOF.")
        print(f"[export] loaded {len(waypoints)} waypoints from {args.trajectory}")
        meta["trajectory"] = {
            "path": os.path.abspath(args.trajectory),
            "sha256": sha256_of(args.trajectory),
            "num_waypoints": int(len(waypoints)),
        }

    tip_path = None

    if animate:
        if args.no_retime:
            traj = _uniform_time_trajectory(waypoints, args.fps)
            print(f"[export] NOT retimed: raw waypoints at {args.fps} fps, "
                  f"{traj.end_time():.3f} s")
        else:
            traj = retime_ambient_path(waypoints, plant)
            print(f"[export] TOPPRA retimed duration: {traj.end_time():.3f} s")
            report = verify_retimed_trajectory(traj, plant, collision_checker)
            _print_report(report)
            meta["verification"] = _jsonable(report)

        n_frames = record_trajectory(
            meshcat, diagram, context, plant, plant_context, traj, args.fps)
        if n_frames < 2:
            raise SystemExit("[export] recorded fewer than 2 frames -- the animation "
                             "would be a single pose. Check --fps against the "
                             "trajectory duration.")
        print(f"[export] recorded {n_frames} frames over {traj.end_time():.3f} s "
              f"at {args.fps} fps")
        meta["recording"] = {"fps": args.fps, "frames": n_frames,
                             "duration": float(traj.end_time()),
                             "retimed": not args.no_retime}

        ts, xyz = sample_tip_path(traj, plant, plant_context, args.tip_samples)
        tip_path = {
            "t": ts.tolist(),
            "xyz": xyz.tolist(),
            "z_min": float(xyz[:, 2].min()),
            "z_max": float(xyz[:, 2].max()),
            "channel_floor_z": floor_z,
        }
        print(f"[export] tip z range [{tip_path['z_min']:.4f}, {tip_path['z_max']:.4f}] m "
              f"(channel floor {floor_z:.4f}) -- pick --trace-lift above this")
    else:
        q = _static_configuration(args, waypoints, plant, plant_context, floor_z,
                                  collision_checker)
        plant.SetPositions(plant_context, q)
        diagram.ForcedPublish(context)
        pose = tip_pose(plant, plant_context, q)
        print(f"[export] static pose q = [" + ", ".join(f"{v:.4f}" for v in q) + "]")
        print(f"[export] marker tip at " + np.array2string(pose.translation(), precision=4))
        meta["static_q"] = [float(v) for v in q]
        meta["static_tip_xyz"] = [float(v) for v in pose.translation()]

    # Must come after every publish (the visualizer re-registers geometry each
    # time) and before StaticHtml. See the module docstring.
    if not args.keep_collision:
        meshcat.Delete("collision")
        meshcat.Delete("proximity")  # Drake's default prefix; harmless if absent
        print("[export] deleted proximity geometry from the meshcat tree")
    else:
        print("[export] WARNING: keeping proximity geometry (--keep-collision)")

    html = meshcat.StaticHtml()
    with open(html_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(html_path) / 1e6
    print(f"[export] wrote {html_path} ({size_mb:.1f} MB)")
    if size_mb < 1.0:
        print("[export] WARNING: the HTML is under 1 MB, which is far smaller than a "
              "scene with 124 boxes and the FR3 meshes should be. Check for an empty "
              "or partially-built scene before rendering it.")

    manifest_path = os.path.join(out_dir, "planks.json")
    with open(manifest_path, "w") as f:
        json.dump(maze_manifest.manifest(), f, indent=2)
    print(f"[export] wrote {manifest_path}")

    if tip_path is not None:
        tip_path_file = os.path.join(out_dir, "tip_path.json")
        with open(tip_path_file, "w") as f:
            json.dump(tip_path, f)
        print(f"[export] wrote {tip_path_file} ({len(tip_path['t'])} samples)")
        meta["tip_path"] = os.path.basename(tip_path_file)

    meta["html"] = os.path.basename(html_path)
    meta["html_bytes"] = os.path.getsize(html_path)
    meta_path = os.path.join(out_dir, "scene_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] wrote {meta_path}")

    if args.serve:
        print(f"[export] Meshcat URL: {meshcat.web_url()}")
        print("[export] holding the server open; Ctrl-C to exit.")
        while True:
            time.sleep(1.0)


def _static_configuration(args, waypoints, plant, plant_context, floor_z,
                          collision_checker=None):
    """Resolve the configuration a static export should hold."""
    if args.q:
        q = np.array([float(v) for v in args.q.split(",")], dtype=float)
        if q.shape != (plant.num_positions(),):
            raise SystemExit(f"--q needs {plant.num_positions()} values, got {q.shape[0]}")
        return q

    if args.ik_tip:
        xy = [float(v) for v in args.ik_tip.split(",")]
        if len(xy) != 2:
            raise SystemExit("--ik-tip takes exactly two values: X,Y")
        target_z = floor_z + args.ik_tip_gap
        print(f"[export] solving IK for tip at ({xy[0]:.4f}, {xy[1]:.4f}, {target_z:.4f}) "
              f"({args.ik_tip_gap * 1000:.0f} mm above the channel floor)")
        return solve_tip_ik(plant, plant_context, xy, target_z,
                            min_distance=args.ik_min_distance,
                            collision_checker=collision_checker)

    if args.pose in ("first", "mid", "last"):
        index = {"first": 0, "mid": len(waypoints) // 2, "last": -1}[args.pose]
        return waypoints[index]

    if args.pose == "zero":
        return np.zeros(plant.num_positions())
    return HOME_Q


def _uniform_time_trajectory(waypoints, fps):
    """A raw first-order-hold through the waypoints, one waypoint per frame.

    Only for --no-retime. Deliberately not a spline: the point of this mode is to
    see the planner's output unsmoothed and untimed.
    """
    from pydrake.all import PiecewisePolynomial

    breaks = np.arange(len(waypoints)) / float(fps)
    return PiecewisePolynomial.FirstOrderHold(breaks, waypoints.T)


def _print_report(report):
    print("[export] verification (dense resample against real, unscaled limits):")
    print(f"[export]   collision-free:         {report['collision_free']}"
          f" ({report['num_collision_samples']} colliding samples)")
    print(f"[export]   within velocity limits: {report['within_velocity_limits']}")
    print(f"[export]   within accel limit:     {report['within_acceleration_limit']}")
    if not (report["collision_free"] and report["within_velocity_limits"]
            and report["within_acceleration_limit"]):
        print("[export]   WARNING: the retimed trajectory failed verification. It is "
              "still being exported for visualization, but do not present it as an "
              "executable result.")


def _jsonable(report):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else
                (bool(v) if isinstance(v, (bool, np.bool_)) else v))
            for k, v in report.items()}


def _pydrake_version():
    try:
        import pydrake
        return getattr(pydrake, "__version__", "unknown")
    except Exception:
        return "unknown"


def _drake_path():
    try:
        from pydrake.common import GetDrakePath
        return GetDrakePath()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
