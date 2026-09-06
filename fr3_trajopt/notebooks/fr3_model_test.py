"""Step 1: build the FR3-marker + maze Drake scene and visualize it in Meshcat.

Run with:
    python notebooks/fr3_model_test.py

Then open the printed Meshcat URL in a browser. The script blocks so the
server stays alive; Ctrl-C to exit.
"""

import os
import sys
import time

# fr3_trajopt is a plain module directory, not an installed package -- add its root
# (the parent of this notebooks/ dir) to sys.path so `import scene` resolves regardless
# of where this checkout lives on disk (host, container, bind mount, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydrake.all import StartMeshcat

from scene import make_default_fr3_infrastructure

meshcat = StartMeshcat()

plant, collision_checker, diagram = make_default_fr3_infrastructure(meshcat)
print(f"Drake infrastructure ready (nq={plant.num_positions()}, nv={plant.num_velocities()}).")
print(f"Meshcat URL: {meshcat.web_url()}")

print("Holding process open so Meshcat stays live. Ctrl-C to exit.")
while True:
    time.sleep(1.0)
