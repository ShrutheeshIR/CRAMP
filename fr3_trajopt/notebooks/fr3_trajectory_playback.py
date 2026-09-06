"""Play back a saved FR3 ambient (joint-space) maze trajectory in Meshcat, TOPPRA-retimed
to real joint velocity/acceleration limits (see retiming.py) instead of a fixed-dt slideshow.

Run with:
    python notebooks/fr3_trajectory_playback.py <trajectory_path> [--speed 1.0] [--loop]

e.g.:
    python notebooks/fr3_trajectory_playback.py \
        ../vamp/resources/fr3_marker/maze_solver_benchmark_trajectories_python/problem_0.txt

Then open the printed Meshcat URL in a browser. The script blocks so the server stays
alive after playback; Ctrl-C to exit.
"""

import argparse
import os
import sys
import time

# fr3_trajopt is a plain module directory, not an installed package -- see
# fr3_model_test.py for why this sys.path insert is needed instead of a package install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydrake.all import StartMeshcat

from retiming import retime_ambient_path, verify_retimed_trajectory
from scene import make_default_fr3_infrastructure
from trajectory import load_ambient_path


def print_verification_report(report: dict) -> None:
    print("Verification (dense resample against real, unscaled limits):")
    print(f"  collision-free:         {report['collision_free']}"
          f" ({report['num_collision_samples']} colliding samples)")
    print(f"  within velocity limits: {report['within_velocity_limits']}"
          f" (max |v| per joint: {report['max_abs_velocity']})")
    print(f"  within accel limit:     {report['within_acceleration_limit']}"
          f" (max |a| per joint: {report['max_abs_acceleration']})")
    if not (report["collision_free"] and report["within_velocity_limits"]
            and report["within_acceleration_limit"]):
        print("  WARNING: retimed trajectory failed verification -- do not trust this for "
              "real execution. Playing it back anyway for inspection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_path", help="Path to an ambient-path .txt file "
                         "(one comma-separated 7-DOF joint configuration per line).")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed multiplier (default: 1.0 = real time).")
    parser.add_argument("--loop", action="store_true",
                         help="Replay the trajectory forever instead of once.")
    args = parser.parse_args()

    waypoints = load_ambient_path(args.trajectory_path)
    print(f"Loaded {len(waypoints)} waypoints from {args.trajectory_path}")

    meshcat = StartMeshcat()
    plant, collision_checker, diagram = make_default_fr3_infrastructure(meshcat)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    print(f"Meshcat URL: {meshcat.web_url()}")

    traj = retime_ambient_path(waypoints, plant)
    print(f"Retimed to {traj.end_time():.3f} s.")
    print_verification_report(verify_retimed_trajectory(traj, plant, collision_checker))

    # Throttled to a fixed frame rate: `time.sleep(0.0)` is a no-op, not a real
    # throttle -- an unthrottled loop calls ForcedPublish (which serializes the whole
    # scene over a websocket to the browser) as fast as Python can spin, often hundreds
    # of times a second for no visual benefit, and if the browser/websocket can't drain
    # messages that fast, publishing can itself start blocking -- which is why
    # wall-clock playback can run much longer than traj.end_time() actually claims.
    frame_period = 1.0 / 60.0

    def play_once():
        t0 = time.time()
        next_frame = t0
        duration = traj.end_time()
        while True:
            t = (time.time() - t0) * args.speed
            if t > duration:
                break
            plant.SetPositions(plant_context, traj.value(t).flatten())
            diagram.ForcedPublish(context)
            next_frame += frame_period
            sleep_for = next_frame - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
        plant.SetPositions(plant_context, traj.value(duration).flatten())
        diagram.ForcedPublish(context)

    play_once()
    while args.loop:
        play_once()

    print("Playback done. Holding process open so Meshcat stays live. Ctrl-C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
