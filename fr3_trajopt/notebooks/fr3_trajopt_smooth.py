"""Smooth a saved FR3 ambient (joint-space) maze trajectory via KinematicTrajectoryOptimization
and visualize the result in Meshcat -- WITHOUT retiming (no TOPPRA, no real velocity/
acceleration limits yet; see trajopt.py's module docstring for why this step comes first).
The result is played back over uniformly-spaced pseudo-time samples, not real seconds.

Run with:
    python notebooks/fr3_trajopt_smooth.py <trajectory_path> [--dt 0.03] [--loop]

e.g.:
    python notebooks/fr3_trajopt_smooth.py \
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

import numpy as np
from pydrake.all import StartMeshcat

from scene import make_default_fr3_infrastructure
from trajectory import load_ambient_path
from trajopt import smooth_ambient_path, verify_smoothed_path


def print_verification_report(report: dict) -> None:
    print("Verification (dense resample -- trajopt's own n_constr_pts sampling can miss "
          "violations between constraint points, this doesn't):")
    print(f"  collision-free:       {report['collision_free']}"
          f" ({report['num_collision_samples']} colliding samples)")
    print(f"  within z bound:       {report['within_z_bound']}"
          f" (max deviation {report['max_z_deviation'] * 1000:.3f} mm,"
          f" allowed {report['z_half_range'] * 1000:.3f} mm)")
    print(f"  within orientation:   {report['within_orientation_bound']}"
          f" (max deviation {report['max_angle_deviation']:.5f} rad,"
          f" allowed {report['angle_tolerance']:.5f} rad)")
    if not (report["collision_free"] and report["within_z_bound"]
            and report["within_orientation_bound"]):
        print("  WARNING: smoothed path failed verification -- do not trust this. "
              "Playing it back anyway for inspection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_path", help="Path to an ambient-path .txt file "
                         "(one comma-separated 7-DOF joint configuration per line).")
    parser.add_argument("--num_samples", type=int, default=200,
                         help="Number of pseudo-time samples to play back (default: 200).")
    parser.add_argument("--dt", type=float, default=0.03,
                         help="Seconds to hold between playback samples (default: 0.03). "
                         "This is NOT the trajectory's real timing -- there isn't one yet.")
    parser.add_argument("--loop", action="store_true",
                         help="Replay the smoothed path forever instead of once.")
    args = parser.parse_args()

    waypoints = load_ambient_path(args.trajectory_path)
    print(f"Loaded {len(waypoints)} waypoints from {args.trajectory_path}")

    meshcat = StartMeshcat()
    plant, collision_checker, diagram = make_default_fr3_infrastructure(meshcat)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    print(f"Meshcat URL: {meshcat.web_url()}")

    smoothed, success = smooth_ambient_path(waypoints, plant, plant_context)
    print(f"Smoothing {'succeeded' if success else 'did NOT report success'}.")
    print_verification_report(
        verify_smoothed_path(smoothed, waypoints, plant, plant_context, collision_checker)
    )

    rs = np.linspace(0.0, 1.0, args.num_samples)

    def play_once():
        for r in rs:
            plant.SetPositions(plant_context, smoothed.value(r).flatten())
            diagram.ForcedPublish(context)
            time.sleep(args.dt)

    play_once()
    while args.loop:
        play_once()

    print("Playback done. Holding process open so Meshcat stays live. Ctrl-C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
