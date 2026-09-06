"""Dumb, simple baseline: TOPPRA-retime the raw waypoints directly -- NO trajopt
smoothing at all, just retiming.py's retime_ambient_path straight on the input file.
Animates the result in Meshcat in real time and plots the actual per-joint velocity
curves. Useful as a reference point to compare a later trajopt+TOPPRA result against.

Run with:
    python notebooks/fr3_toppra_baseline.py <trajectory_path> [--speed 1.0]

e.g.:
    python notebooks/fr3_toppra_baseline.py \
        ../vamp/resources/fr3_marker/maze_solver_benchmark_trajectories_python/problem_0.txt

Then open the printed Meshcat URL in a browser. The script blocks so the server stays
alive after playback; Ctrl-C to exit. Saves a velocity plot next to this script.
"""

import argparse
import os
import sys
import time

# fr3_trajopt is a plain module directory, not an installed package -- see
# fr3_model_test.py for why this sys.path insert is needed instead of a package install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import StartMeshcat

from retiming import retime_ambient_path, verify_retimed_trajectory
from scene import make_default_fr3_infrastructure
from trajectory import load_ambient_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_path", help="Path to an ambient-path .txt file "
                         "(one comma-separated 7-DOF joint configuration per line).")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed multiplier (default: 1.0 = real time).")
    args = parser.parse_args()

    waypoints = load_ambient_path(args.trajectory_path)
    print(f"Loaded {len(waypoints)} waypoints from {args.trajectory_path}")

    meshcat = StartMeshcat()
    plant, collision_checker, diagram = make_default_fr3_infrastructure(meshcat)
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    print(f"Meshcat URL: {meshcat.web_url()}")

    traj = retime_ambient_path(waypoints, plant)
    print(f"Retimed duration: {traj.end_time():.3f} s")

    report = verify_retimed_trajectory(traj, plant, collision_checker)
    print(f"collision-free: {report['collision_free']}"
          f" ({report['num_collision_samples']} colliding samples)")
    print(f"within velocity limits: {report['within_velocity_limits']}"
          f" (max |v| per joint: {report['max_abs_velocity']})")
    print(f"within accel limit: {report['within_acceleration_limit']}"
          f" (max |a| per joint: {report['max_abs_acceleration']})")

    ts = np.linspace(traj.start_time(), traj.end_time(), 500)
    vs = np.array([traj.EvalDerivative(t, 1).flatten() for t in ts])

    fig, ax = plt.subplots()
    for j in range(vs.shape[1]):
        ax.plot(ts, vs[:, j], label=f"fr3_joint{j + 1}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint velocity (rad/s)")
    ax.set_title("TOPPRA baseline (raw waypoints, no trajopt smoothing)")
    ax.legend()
    plot_path = os.path.join(os.path.dirname(__file__), "fr3_toppra_baseline_velocities.png")
    fig.savefig(plot_path)
    print(f"Saved velocity plot to {plot_path}")

    # Throttled to a fixed frame rate: an unthrottled loop calls ForcedPublish (which
    # serializes the whole scene over a websocket to the browser) as fast as Python can
    # spin, often hundreds of times a second for no visual benefit -- and if the
    # browser/websocket can't drain messages that fast, publishing can itself start
    # blocking, which is why wall-clock playback can run much longer than
    # traj.end_time() actually claims.
    frame_period = 1.0 / 60.0
    duration = traj.end_time()
    t0 = time.time()
    next_frame = t0
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

    print("Playback done. Holding process open so Meshcat stays live. Ctrl-C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
