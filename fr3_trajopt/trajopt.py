"""Reshape a raw ambient waypoint path into a smooth, collision-aware B-spline via
KinematicTrajectoryOptimization -- the step rby1_planning.py's _reach_trajopt performs
before TOPPRA (see its module comment on TOPPRA_RELAXATION_LADDER: "what is hard to
retime is the jagged shortcut path... a smoothed B-spline through the same mapping is
fine"). retiming.py's cubic spline through the raw waypoints was NOT a substitute for
this: it's C2-continuous, but still shaped like the jagged input, just twice
differentiable -- an actual optimization is what removes the sharp turns themselves.

Deliberately simpler than rby1_planning.py's version in one respect: no guess-
multiplicity LADDER (which tries multiplicity 1 first and only falls back to full
multiplicity if that leaves the guess infeasible -- see _path_to_bspline's docstring
there). We skip straight to full multiplicity, since a maze's narrow corridors make a
sparse control-point B-spline's corner-cutting far more likely to actually clip a wall
than in rby1's open-space reach motions: at multiplicity 1, decimating hundreds of
waypoints down to num_control_points and connecting them with a smooth curve does NOT
pass through the decimated points, so the seed itself is plausibly already colliding,
and a non-convex solver started from an infeasible guess can get stuck reporting
kInfeasibleConstraints instead of escaping to a feasible region. Full multiplicity
(duplicating each decimated waypoint, nudged apart by eps) makes the seed curve nearly
interpolate those points directly instead.

No stability/support-polygon term either (that's RBY1-specific, for a mobile-base
bimanual robot).

The maze task pins the marker tip's world pose almost entirely while tracing the maze:
only x, y vary; z and orientation are held fixed the whole time (see
vamp/scripts/fr3_marker_maze_example.py's Z_HEIGHT/DOWN_QUAT and its TSR bounds, which
pin dz to 0). Nothing in a bare position-bounds + collision + energy-cost formulation
knows about that constraint, so the optimizer is free to lift the marker off the maze
plane anywhere between the pinned start/goal while it reshapes the path -- this needs an
explicit path constraint on the tip frame's pose (see _measure_tip_pose /
_maze_tip_pose_constraints), not a reformulation into VAMP's own task-space
parameterization: Drake computes forward kinematics (and its AutoDiff derivatives)
through this plant natively, so the constraint is just as easy to add directly in
ambient/joint space as VAMP's task-space redundancy elimination would be, without
needing to port VAMP's IK.

The target z-height/orientation are MEASURED from the given waypoints via this plant's
own forward kinematics, not hardcoded to VAMP's stated constants: VAMP's kinematic model
and this Drake URDF are technically different chains, and trusting an externally-
asserted "exact" value here would repeat the same mistake the min-distance floor bug
above already made once (see _feasible_min_dist_bound).

IMPORTANT: KinematicTrajectoryOptimization only enforces the collision and plane/
orientation constraints AT its `n_constr_pts` sampled path locations -- it says nothing
about the curve BETWEEN them (rby1_planning.py's verify_trajectory docstring states this
plainly: "Trajopt constrains only n_constr_pts samples of a B-spline and says nothing
about the curve between them"). A B-spline threading a maze's narrow corridors can clip
a wall between two constraint-satisfying samples even when the solver reports success.
Raising n_constr_pts (below) narrows the gaps a violation can hide in, but is a
mitigation, not a proof -- verify_smoothed_path is the actual check: it densely
resamples a finished path and independently verifies collision and plane/orientation
adherence, the same role rby1_planning.py's verify_trajectory plays there and
retiming.py's verify_retimed_trajectory plays for the retiming step. Always call it on
a result before trusting it.

This module does NOT retime -- it returns a path over pseudo-time r in [0, 1], not real
seconds. Time-parameterization against real velocity/acceleration limits is retiming.py's
separate, later step (paused for now, per the plan).
"""

import numpy as np
from pydrake.all import (
    BsplineBasis,
    BsplineTrajectory,
    IpoptSolver,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    OrientationConstraint,
    PositionConstraint,
    Quaternion,
    RotationMatrix,
    SnoptSolver,
)

from scene import FR3_TIP_FRAME


def _seed_bspline(
    waypoints: np.ndarray,
    num_control_points: int,
    spline_order: int,
    q_lb: np.ndarray,
    q_ub: np.ndarray,
    eps: float = 1e-4,
) -> BsplineTrajectory:
    """Decimate the raw waypoints down to num_control_points, then duplicate each one to
    full multiplicity (spline_order - 1 copies), nudged apart by `eps` along the local
    path direction, so the resulting B-spline nearly interpolates the decimated waypoints
    instead of smoothly cutting corners between them (see this module's docstring for
    why that distinction matters here). Mirrors rby1_planning.py's _path_to_bspline at
    full multiplicity, without its per-waypoint multiplicity ladder.

    q_lb/q_ub clip the perturbed control points back into the position bounds trajopt
    will impose, since eps could otherwise nudge one a hair past a bound the original
    waypoint respected exactly.
    """
    n = len(waypoints)
    idx = np.linspace(0, n - 1, num_control_points).round().astype(int)
    decimated = waypoints[idx]  # (num_control_points, dof)

    block = max(1, spline_order - 1)
    ctrl = np.repeat(decimated, block, axis=0).T  # (dof, num_control_points * block)

    for i in range(num_control_points):
        if i == 0:
            direction = decimated[1] - decimated[0] if num_control_points > 1 else np.zeros_like(decimated[0])
            offsets = eps * np.arange(block)
        elif i == num_control_points - 1:
            direction = decimated[-1] - decimated[-2]
            offsets = -eps * np.arange(block)
        else:
            direction = decimated[i + 1] - decimated[i - 1]
            offsets = eps * (np.arange(block) - (block - 1) / 2.0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-12 else np.zeros_like(direction)
        s, e = i * block, (i + 1) * block
        ctrl[:, s:e] += np.outer(direction, offsets)

    ctrl = np.clip(ctrl, q_lb.reshape(-1, 1), q_ub.reshape(-1, 1))

    basis = BsplineBasis(spline_order, ctrl.shape[1],
                          initial_parameter_value=0.0, final_parameter_value=1.0)
    return BsplineTrajectory(basis, ctrl)


def _path_min_clearance(q_samples: np.ndarray, plant, plant_context) -> float:
    """Minimum signed distance over a set of full-plant configurations. Negative means
    penetration. Mirrors rby1_planning.py's _path_min_clearance -- used to pick
    trajopt's distance floor from what a path actually achieves, instead of a constant.
    """
    worst = np.inf
    port = plant.get_geometry_query_input_port()
    for q in q_samples:
        if q is None or not np.all(np.isfinite(q)):
            return -np.inf
        plant.SetPositions(plant_context, q)
        qobj = port.Eval(plant_context)
        pens = qobj.ComputePointPairPenetration()
        if pens:
            worst = min(worst, -float(max(p.depth for p in pens)))
            continue
        pairs = qobj.ComputeSignedDistancePairwiseClosestPoints(0.05)
        if pairs:
            worst = min(worst, float(min(p.distance for p in pairs)))
    return 0.05 if worst is np.inf else worst


def _feasible_min_dist_bound(requested: float, path_clearance: float, slack: float = 1e-4) -> float:
    """Lower trajopt's distance floor to something the path actually satisfies. A floor
    above what the guess achieves is infeasible at the guess itself, so the solver
    starts outside the feasible set -- this is very plausibly why smoothing reported
    kInfeasibleConstraints with a fixed 1mm floor: a maze's walls can leave a genuinely
    sub-millimetre margin (rby1_planning.py measured 0.33-1.26mm on its own paths).
    Mirrors rby1_planning.py's _feasible_min_dist_bound.
    """
    if not np.isfinite(path_clearance):
        return 0.0
    usable = max(0.0, path_clearance - slack)
    if usable < requested:
        print(f"[trajopt] lowering min-distance floor {requested * 1000:.3f} -> "
              f"{usable * 1000:.3f} mm: the path only clears "
              f"{path_clearance * 1000:.3f} mm")
        return usable
    return requested


def _measure_tip_pose(waypoints: np.ndarray, plant, plant_context, tip_frame_name: str):
    """Measure the marker tip's world z-height and orientation (as wxyz quaternions)
    across the given waypoints, via THIS plant's own forward kinematics. Used to derive
    the maze plane/orientation constraint from what the trusted input path actually
    achieves here, rather than from VAMP's stated constants (see this module's docstring
    for why that distinction matters).
    """
    tip_frame = plant.GetFrameByName(tip_frame_name)
    world_frame = plant.world_frame()
    zs = np.empty(len(waypoints))
    quats = np.empty((len(waypoints), 4))  # wxyz
    for i, q in enumerate(waypoints):
        plant.SetPositions(plant_context, q)
        X_WT = plant.CalcRelativeTransform(plant_context, world_frame, tip_frame)
        zs[i] = X_WT.translation()[2]
        quats[i] = X_WT.rotation().ToQuaternion().wxyz()
    return zs, quats


def _maze_plane_bounds(
    waypoints: np.ndarray,
    plant,
    plant_context,
    tip_frame_name: str = FR3_TIP_FRAME,
    slack: float = 1e-4,
    angle_slack: float = 1e-3,
):
    """Numeric (z_center, z_half_range, ref_quat, angle_tolerance) bounds derived from
    what the given waypoints actually achieve at the marker tip -- shared by
    _maze_tip_pose_constraints (which wraps these into Drake Constraints trajopt
    enforces AT its sampled points) and verify_smoothed_path (which rechecks a finished
    trajectory densely, CATCHING any drift trajopt's sparse sampling missed BETWEEN
    them -- see this module's docstring on why both are needed).
    """
    zs, quats = _measure_tip_pose(waypoints, plant, plant_context, tip_frame_name)

    z_center = 0.5 * (zs.min() + zs.max())
    z_half_range = 0.5 * (zs.max() - zs.min()) + slack

    # Angular deviation of every waypoint's tip orientation from the first waypoint's,
    # via the same quaternion-dot-product formula vamp/scripts/fr3_marker_maze_example.py
    # uses for its own "SE3 distance" (se3_distance): 2*acos(|q_a . q_b|).
    ref_quat = quats[0]
    dots = np.clip(np.abs(quats @ ref_quat), -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    angle_tolerance = float(angles.max()) + angle_slack

    return z_center, z_half_range, ref_quat, angle_tolerance


def _maze_tip_pose_constraints(
    waypoints: np.ndarray,
    plant,
    plant_context,
    *,
    tip_frame_name: str = FR3_TIP_FRAME,
    slack: float = 1e-4,
    angle_slack: float = 1e-3,
):
    """Build (position_constraint, orientation_constraint) pinning the marker tip's
    z-height and full orientation to what the given waypoints actually achieve (x, y
    stay free) -- the constraint VAMP's own TSR sampler enforces during planning, which
    trajopt must also enforce or nothing stops it lifting the marker off the maze board
    while reshaping the path between the pinned start/goal. See this module's docstring.
    """
    z_center, z_half_range, ref_quat, angle_tolerance = _maze_plane_bounds(
        waypoints, plant, plant_context, tip_frame_name, slack, angle_slack,
    )

    tip_frame = plant.GetFrameByName(tip_frame_name)
    world_frame = plant.world_frame()

    z_lb = np.array([-np.inf, -np.inf, z_center - z_half_range])
    z_ub = np.array([np.inf, np.inf, z_center + z_half_range])
    position_constraint = PositionConstraint(
        plant, world_frame, z_lb, z_ub, tip_frame, np.zeros(3), plant_context,
    )

    R_target = RotationMatrix(Quaternion(*ref_quat))
    orientation_constraint = OrientationConstraint(
        plant, world_frame, R_target, tip_frame, RotationMatrix(), angle_tolerance,
        plant_context,
    )
    return position_constraint, orientation_constraint


def verify_smoothed_path(
    traj,
    waypoints: np.ndarray,
    plant,
    plant_context,
    collision_checker,
    *,
    tip_frame_name: str = FR3_TIP_FRAME,
    num_samples: int = 2000,
) -> dict:
    """Densely resample a smoothed path and check it against collision and the maze
    plane/orientation bounds -- the check for what trajopt's sparse `n_constr_pts`
    sampling can miss between constraint points (see this module's docstring and
    rby1_planning.py's verify_trajectory, which exists for exactly this reason). Does
    not raise; returns a report so the caller decides what to do with a failure (e.g.
    fall back to the raw un-smoothed path).

    Returns a dict with keys: "collision_free" (bool), "num_collision_samples" (int),
    "within_z_bound" (bool), "max_z_deviation" (float, metres),
    "within_orientation_bound" (bool), "max_angle_deviation" (float, radians).
    """
    z_center, z_half_range, ref_quat, angle_tolerance = _maze_plane_bounds(
        waypoints, plant, plant_context, tip_frame_name,
    )

    rs = np.linspace(0.0, 1.0, num_samples)
    qs = np.array([traj.value(r).flatten() for r in rs])

    collision_free = [collision_checker.CheckConfigCollisionFree(q) for q in qs]
    zs, quats = _measure_tip_pose(qs, plant, plant_context, tip_frame_name)

    max_z_deviation = float(np.abs(zs - z_center).max())
    dots = np.clip(np.abs(quats @ ref_quat), -1.0, 1.0)
    max_angle_deviation = float((2.0 * np.arccos(dots)).max())

    return {
        "collision_free": all(collision_free),
        "num_collision_samples": len(collision_free) - sum(collision_free),
        "within_z_bound": max_z_deviation <= z_half_range,
        "max_z_deviation": max_z_deviation,
        "z_half_range": z_half_range,
        "within_orientation_bound": max_angle_deviation <= angle_tolerance,
        "max_angle_deviation": max_angle_deviation,
        "angle_tolerance": angle_tolerance,
    }


def smooth_ambient_path(
    waypoints: np.ndarray,
    plant,
    plant_context,
    *,
    num_control_points: int = 100,
    spline_order: int = 4,
    min_dist_lower_bound: float = 0.001,
    min_dist_influence: float = 0.05,
    n_constr_pts: int = 150,
):
    """Reshape a raw waypoint path into a smooth, collision-aware B-spline.

    Args:
        waypoints: (N, dof) array, already collision-free/on-plane (e.g. VAMP's
            shortcut ambient path). Only its start/end and shape are trusted; the
            optimizer is free to reshape everything in between.
        plant: finalized MultibodyPlant (e.g. from
            scene.make_default_fr3_infrastructure) with the maze obstacles already
            registered.
        plant_context: a Context for `plant`, obtained from a diagram context (e.g.
            plant.GetMyContextFromRoot(diagram.CreateDefaultContext())) so the scene
            graph query object used for collision checking sees the maze geometry.
        num_control_points: control points in the seed/decision B-spline. VAMP's raw
            path is already densely sampled and shortcut, so this can be small.
        min_dist_lower_bound: hard collision-free floor the solver must satisfy (metres).
        min_dist_influence: width of the smooth collision-penalty zone above the floor.
        n_constr_pts: number of path locations (evenly spaced in r) the collision AND
            maze-plane/orientation constraints are sampled at (see
            _maze_tip_pose_constraints -- the marker tip's z-height and orientation are
            pinned to what `waypoints` itself measures, x/y stay free).

    Returns:
        (traj, success): traj is the resulting BsplineTrajectory, queryable via
        .value(r) for r in [0, 1] (pseudo-time, NOT seconds). success is False if the
        solver didn't report success; traj is still its best iterate in that case, not
        None -- the caller decides whether that's usable.
    """
    dof = plant.num_positions()
    if waypoints.shape[1] != dof:
        raise ValueError(f"waypoints has {waypoints.shape[1]} columns, plant has {dof} DOF")

    q_lb = plant.GetPositionLowerLimits()
    q_ub = plant.GetPositionUpperLimits()

    guess = _seed_bspline(waypoints, num_control_points, spline_order, q_lb, q_ub)

    trajopt = KinematicTrajectoryOptimization(guess)
    trajopt.AddPositionBounds(q_lb, q_ub)
    trajopt.AddPathPositionConstraint(waypoints[0], waypoints[0], 0.0)
    trajopt.AddPathPositionConstraint(waypoints[-1], waypoints[-1], 1.0)

    # Measured on the path's own waypoints, which is what the full-multiplicity guess
    # tracks -- see _feasible_min_dist_bound's docstring for why the floor must not
    # exceed this.
    path_clearance = _path_min_clearance(waypoints, plant, plant_context)
    min_dist_lower_bound = _feasible_min_dist_bound(min_dist_lower_bound, path_clearance)

    min_dist_constraint = MinimumDistanceLowerBoundConstraint(
        plant, min_dist_lower_bound, plant_context, None, min_dist_influence,
    )
    tip_position_constraint, tip_orientation_constraint = _maze_tip_pose_constraints(
        waypoints, plant, plant_context,
    )
    for s in np.linspace(0.0, 1.0, n_constr_pts):
        trajopt.AddPathPositionConstraint(min_dist_constraint, s)
        trajopt.AddPathPositionConstraint(tip_position_constraint, s)
        trajopt.AddPathPositionConstraint(tip_orientation_constraint, s)

    trajopt.AddPathEnergyCost()

    prog = trajopt.prog()
    snopt = SnoptSolver()
    solver = snopt if snopt.available() else IpoptSolver()
    print(f"[trajopt] solver: {solver.solver_id().name()}"
          + ("" if snopt.available() else " (SNOPT unavailable, no license found)"))

    result = solver.Solve(prog)
    success = result.is_success()
    if not success:
        print(f"[trajopt] solver did not report success "
              f"(solution_result={result.get_solution_result()}); "
              f"returning its best iterate anyway.")

    return trajopt.ReconstructTrajectory(result), success
