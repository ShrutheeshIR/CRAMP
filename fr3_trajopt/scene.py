"""Drake plant/scene-graph construction for the FR3-marker robot in the maze environment.

Mirrors the pattern used in rby1-constrained-planning/src/rby1_planning.py
(make_default_rby1_infrastructure / SceneBox / _add_obstacle), adapted for a
fixed-base 7-DOF FR3 arm with a marker end-effector instead of RBY1's mobile
bimanual setup.

Plain module, not an installed package -- import it by adding this directory
to sys.path (see notebooks/fr3_model_test.py), not via `pip install`.
"""

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from pydrake.all import (
    Box,
    CollisionCheckerParams,
    CoulombFriction,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    RobotDiagramBuilder,
    Role,
    RollPitchYaw,
    SceneGraphCollisionChecker,
)

import common

# Local, symlinked-mesh copy of cricket/resources/fr3_marker/fr3_expo_spherized.urdf with
# one line patched: the upstream URDF references "meshes/visual/hand.obj", which doesn't
# exist on disk (only hand.dae does) -- see models/fr3_marker/README.md.
FR3_MARKER_URDF = os.path.join(
    common.RepoDir(), "models", "fr3_marker", "fr3_expo_spherized.urdf"
)
# Local copy of vamp/resources/environments/maze_cuboids.json -- see
# resources/environments/README.md.
MAZE_CUBOIDS_JSON = os.path.join(
    common.RepoDir(), "resources", "environments", "maze_cuboids.json"
)

# Root link of the FR3 URDF chain (welded to the world) and the marker tip frame
# (fr3_link0 -> ... -> fr3_link8 -> fr3_hand -> fr3_marker_holder -> fr3_marker -> fr3_tip).
FR3_ROOT_LINK = "fr3_link0"
FR3_TIP_FRAME = "fr3_tip"


@dataclass(frozen=True)
class SceneBox:
    """A static box obstacle (e.g. a maze wall) anchored in the world frame.

    Attributes:
        size:  (dx, dy, dz) full box dimensions in metres.
        xyz:   (x, y, z) world position of the box CENTRE.
        rpy:   (roll, pitch, yaw) orientation in the world frame, radians.
        name:  Geometry name; must be unique across obstacles.
        color: RGBA of the visual box.
    """

    size: Tuple[float, float, float]
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "obstacle"
    color: Tuple[float, float, float, float] = (0.55, 0.55, 0.6, 1.0)


# Cuboid position offset applied on load, matching vamp/scripts/fr3_marker_maze_example.py's
# load_maze_environment() exactly ("push forward/up ... on the assumption of a shared
# maze/mount frame with the iiwa rig") -- the maze JSON's raw x/y/z are in the iiwa rig's
# frame, not FR3's, and this offset is what puts the maze in the right place relative to
# the FR3 base welded at the world origin here.
MAZE_OFFSET_XYZ = (0.285 * 2, 0.0, 0.05)


def load_maze_boxes(json_path: str = MAZE_CUBOIDS_JSON) -> List[SceneBox]:
    """Parse VAMP's maze-cuboid JSON (name/x/y/z/dx/dy/dz/roll/pitch/yaw) into SceneBoxes,
    applying the same position offset fr3_marker_maze_example.py's load_maze_environment()
    does (see MAZE_OFFSET_XYZ)."""
    with open(json_path, "r") as f:
        raw = json.load(f)
    dx, dy, dz = MAZE_OFFSET_XYZ
    return [
        SceneBox(
            size=(b["dx"], b["dy"], b["dz"]),
            xyz=(b["x"] + dx, b["y"] + dy, b["z"] + dz),
            rpy=(b["roll"], b["pitch"], b["yaw"]),
            name=b["name"],
        )
        for b in raw
    ]


def _add_obstacle(plant, obs: SceneBox) -> None:
    """Register a static box obstacle on the world body. Must precede Finalize."""
    X_WB = RigidTransform(RollPitchYaw(*obs.rpy), np.asarray(obs.xyz, dtype=float))
    shape = Box(*obs.size)
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_WB, shape, f"{obs.name}_collision", CoulombFriction(0.9, 0.8)
    )
    plant.RegisterVisualGeometry(
        plant.world_body(), X_WB, shape, f"{obs.name}_visual",
        np.asarray(obs.color, dtype=float),
    )


def make_default_fr3_infrastructure(
    meshcat=None,
    *,
    obstacles: Optional[Union[SceneBox, Iterable[SceneBox], str]] = "maze",
    edge_step_size: float = 0.01,
):
    """Build the FR3-marker Drake plant + collision checker + diagram.

    Args:
        meshcat: a Meshcat instance (from pydrake.all.StartMeshcat()), or None to skip
            visualization entirely.
        obstacles: "maze" (default) loads resources/environments/maze_cuboids.json (a local
            copy of vamp's) as static box obstacles; pass a SceneBox / iterable of SceneBox
            to override, or None for an empty scene (arm only).
        edge_step_size: resolution used by the SceneGraphCollisionChecker for edge checks.

    Returns:
        (plant, collision_checker, diagram)
    """
    builder = RobotDiagramBuilder(time_step=0.0)
    plant = builder.plant()
    parser = Parser(plant)
    (fr3_model,) = parser.AddModels(FR3_MARKER_URDF)

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(FR3_ROOT_LINK, fr3_model))

    if obstacles == "maze":
        obs_list = load_maze_boxes()
    elif obstacles is None:
        obs_list = []
    elif isinstance(obstacles, SceneBox):
        obs_list = [obstacles]
    else:
        obs_list = list(obstacles)
    for obs in obs_list:
        _add_obstacle(plant, obs)

    if meshcat is not None:
        viz_params = MeshcatVisualizerParams()
        viz_params.delete_on_initialization_event = False
        viz_params.role = Role.kIllustration
        viz_params.prefix = "visual"
        MeshcatVisualizer.AddToBuilder(
            builder.builder(), builder.scene_graph(), meshcat, viz_params
        )

        col_viz_params = MeshcatVisualizerParams()
        col_viz_params.delete_on_initialization_event = False
        col_viz_params.role = Role.kProximity
        col_viz_params.prefix = "collision"
        col_viz_params.visible_by_default = False
        MeshcatVisualizer.AddToBuilder(
            builder.builder(), builder.scene_graph(), meshcat, col_viz_params
        )

    coll_params = CollisionCheckerParams()
    coll_params.robot_model_instances = [fr3_model]
    plant.Finalize()
    diagram = builder.Build()

    coll_params.model = diagram
    coll_params.edge_step_size = edge_step_size
    collision_checker = SceneGraphCollisionChecker(coll_params)

    diagram.ForcedPublish(diagram.CreateDefaultContext())

    return plant, collision_checker, diagram
