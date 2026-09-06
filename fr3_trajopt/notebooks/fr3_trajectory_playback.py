"""Step 2: play back a saved FR3 ambient (joint-space) maze trajectory in Meshcat.

Run with:
    python notebooks/fr3_trajectory_playback.py <trajectory_path> [--dt 0.05] [--loop]

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

from scene import make_default_fr3_infrastructure
from trajectory import load_ambient_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_path", help="Path to an ambient-path .txt file "
                         "(one comma-separated 7-DOF joint configuration per line).")
    parser.add_argument("--dt", type=float, default=0.05,
                         help="Seconds to hold between waypoints (default: 0.05).")
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

    def play_once():
        for q in waypoints:
            plant.SetPositions(plant_context, q)
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
