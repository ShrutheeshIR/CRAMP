"""Time-parameterize a raw ambient (joint-space) waypoint path into a real,
dynamically-executable Drake trajectory via TOPPRA.

Velocity limits are the FR3 URDF's per-joint values
(models/fr3_marker/fr3_expo_spherized.urdf, <limit velocity="...">). Acceleration limits
are NOT in that URDF -- Franka's own description package doesn't publish them. The uniform
bound below is libfranka's actual rate-limiting constant (`kMaxJointAcceleration` in
libfranka/include/franka/rate_limiting.h), i.e. the real limit the robot's controller
enforces, not a guessed number.

TOPPRA only enforces velocity/acceleration bounds at the discrete gridpoints it solves
over -- between them, the actual continuous trajectory can run slightly over the nominal
bound (a known discretization artifact of the method, confirmed empirically here: with
default gridpoints, sampling the retimed trajectory densely showed velocity overshoots of
a few percent, worse for acceleration). This module compensates by feeding TOPPRA bounds
already scaled down by `safety_margin` (default 0.9), rather than the raw limits -- this
is a mitigation, not a proof of compliance. Before trusting a retimed trajectory for real
execution, resample it densely and verify velocity/acceleration/collision bounds
independently (see the safety-verification step in the wider trajopt plan); don't rely on
TOPPRA's own claimed feasibility alone.
"""

import numpy as np
from pydrake.all import (
    CalcGridPointsOptions,
    PathParameterizedTrajectory,
    PiecewisePolynomial,
    Toppra,
)

# From models/fr3_marker/fr3_expo_spherized.urdf's <limit velocity="..."> per fr3_joint1..7.
JOINT_VELOCITY_LIMITS = np.array([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0])

# libfranka's kMaxJointAcceleration (rate_limiting.h) -- uniform across all 7 joints; this is
# the actual bound the real robot's controller enforces, not a URDF value (the URDF has none).
JOINT_ACCELERATION_LIMIT = 10.0

# Multiplies both limits above before handing them to TOPPRA, to leave headroom for the
# between-gridpoint overshoot described above. Not a substitute for downstream verification.
DEFAULT_SAFETY_MARGIN = 0.9


def waypoints_to_path(waypoints: np.ndarray) -> PiecewisePolynomial:
    """Turn an (N, dof) waypoint array into an arc-length-parameterized path q(s), where s
    runs from 0 to the total joint-space arc length.

    Uses a C2-continuous cubic spline rather than linear interpolation, since a
    piecewise-linear q(s) has a kink in dq/ds at every waypoint -- that's a real
    discontinuity in commanded velocity at each waypoint, independent of (and in addition
    to) the gridpoint-overshoot issue described in this module's docstring.
    """
    deltas = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    breaks = np.concatenate([[0.0], np.cumsum(deltas)])
    if breaks[-1] == 0.0:
        raise ValueError("Path has zero arc length (all waypoints identical).")
    return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(breaks, waypoints.T)


def retime_ambient_path(
    waypoints: np.ndarray,
    plant,
    *,
    velocity_limits: np.ndarray = JOINT_VELOCITY_LIMITS,
    acceleration_limit: float = JOINT_ACCELERATION_LIMIT,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> PathParameterizedTrajectory:
    """TOPPRA-retime a raw waypoint path against real joint velocity/acceleration limits.

    Args:
        waypoints: (N, dof) array of joint configurations.
        plant: the MultibodyPlant the path is defined over (e.g. from
            scene.make_default_fr3_infrastructure) -- used only for its DOF count/ordering,
            not its collision geometry (the input path is assumed already collision-free).
        velocity_limits: per-joint |q_dot| bound, same order as waypoints' columns, BEFORE
            safety_margin is applied.
        acceleration_limit: uniform |q_ddot| bound applied to every joint, BEFORE
            safety_margin is applied.
        safety_margin: scales both limits down before handing them to TOPPRA (see module
            docstring for why). Pass 1.0 to disable.

    Returns:
        A PathParameterizedTrajectory q(t), queryable via .value(t) /
        .EvalDerivative(t, 1) / .EvalDerivative(t, 2) for position/velocity/acceleration,
        with .end_time() giving the total retimed duration.
    """
    dof = plant.num_positions()
    if waypoints.shape[1] != dof:
        raise ValueError(f"waypoints has {waypoints.shape[1]} columns, plant has {dof} DOF")

    path = waypoints_to_path(waypoints)
    gridpoints = Toppra.CalcGridPoints(path, CalcGridPointsOptions())

    toppra = Toppra(path, plant, gridpoints)
    v_limit = velocity_limits * safety_margin
    toppra.AddJointVelocityLimit(-v_limit, v_limit)
    a_limit = np.full(dof, acceleration_limit * safety_margin)
    toppra.AddJointAccelerationLimit(-a_limit, a_limit)

    time_scaling = toppra.SolvePathParameterization()
    if time_scaling is None:
        raise RuntimeError("TOPPRA failed to find a feasible time parameterization.")

    return PathParameterizedTrajectory(path, time_scaling)


def verify_retimed_trajectory(
    traj: PathParameterizedTrajectory,
    plant,
    collision_checker,
    *,
    velocity_limits: np.ndarray = JOINT_VELOCITY_LIMITS,
    acceleration_limit: float = JOINT_ACCELERATION_LIMIT,
    num_samples: int = 2000,
) -> dict:
    """Densely resample a retimed trajectory and check it against the REAL (unscaled)
    limits and against collision -- the independent check that safety_margin is a
    mitigation for, not a replacement of. Does not raise; returns a report so the caller
    decides what to do with a failure (e.g. fall back to the un-retimed raw path).

    Returns a dict with keys: "collision_free" (bool), "num_collision_samples" (int),
    "within_velocity_limits" (bool), "max_abs_velocity" (per-joint array),
    "within_acceleration_limit" (bool), "max_abs_acceleration" (per-joint array).
    """
    ts = np.linspace(traj.start_time(), traj.end_time(), num_samples)
    qs = np.array([traj.value(t).flatten() for t in ts])
    vs = np.array([traj.EvalDerivative(t, 1).flatten() for t in ts])
    accs = np.array([traj.EvalDerivative(t, 2).flatten() for t in ts])

    collision_free = [collision_checker.CheckConfigCollisionFree(q) for q in qs]
    max_abs_v = np.abs(vs).max(axis=0)
    max_abs_a = np.abs(accs).max(axis=0)

    return {
        "collision_free": all(collision_free),
        "num_collision_samples": len(collision_free) - sum(collision_free),
        "within_velocity_limits": bool((max_abs_v <= velocity_limits).all()),
        "max_abs_velocity": max_abs_v,
        "within_acceleration_limit": bool((max_abs_a <= acceleration_limit).all()),
        "max_abs_acceleration": max_abs_a,
    }
