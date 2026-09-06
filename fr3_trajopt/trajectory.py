"""Loading for saved FR3 ambient (joint-space) trajectories.

File format: one waypoint per line, comma-separated joint values -- matches
vamp/scripts/cpp/fr3_maze_solver_benchmark.cc's write_ambient_path and
vamp/scripts/fr3_marker_maze_example.py's write_ambient_path (e.g.
vamp/resources/fr3_marker/maze_solver_benchmark_trajectories_python/problem_0.txt).
"""

import numpy as np


def load_ambient_path(path: str) -> np.ndarray:
    """Parse a comma-separated-per-line ambient path file into an (N, 7) array."""
    with open(path, "r") as f:
        rows = [line.split(",") for line in f if line.strip()]
    return np.array([[float(v) for v in row] for row in rows], dtype=float)
