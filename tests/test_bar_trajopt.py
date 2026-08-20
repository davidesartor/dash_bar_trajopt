"""Checks for the bar-angle trajectory-optimization evaluator."""

import math

import numpy as np
import pytest

from dash_mjlab.trajopt import BarAngleTrajOptEnv
from dash_mjlab.trajopt.example import (
  TARGET_ANGLE,
  hold_spawn,
  two_phase_push,
)


@pytest.fixture(scope="module")
def env() -> BarAngleTrajOptEnv:
  return BarAngleTrajOptEnv()


def test_untouched_bar_costs_target(env: BarAngleTrajOptEnv) -> None:
  """Holding the spawn pose never touches the bar, so the cost is exactly the
  commanded angle -- gravity must not move a bar on a vertical hinge."""
  assert env.evaluate(hold_spawn, target_angle=0.6) == pytest.approx(0.6, abs=0.02)


def test_deterministic(env: BarAngleTrajOptEnv) -> None:
  assert env.evaluate(two_phase_push, TARGET_ANGLE) == env.evaluate(
    two_phase_push, TARGET_ANGLE
  )


def test_precompute_matches_stepwise(env: BarAngleTrajOptEnv) -> None:
  """A law that is a function of time alone cannot tell the two query modes
  apart, so the costs must agree bit for bit."""
  assert env.evaluate(two_phase_push, TARGET_ANGLE, precompute=True) == env.evaluate(
    two_phase_push, TARGET_ANGLE, precompute=False
  )


def test_cost_wraps(env: BarAngleTrajOptEnv) -> None:
  """The cost is the shortest angular distance, never the long way round."""
  cost = env.evaluate(hold_spawn, target_angle=2 * math.pi - 0.3)
  assert cost == pytest.approx(0.3, abs=0.02)


def test_out_of_range_targets_are_clamped(env: BarAngleTrajOptEnv) -> None:
  def far_outside(t: np.ndarray) -> np.ndarray:
    return np.broadcast_to([100.0, -100.0, 50.0, -50.0], (*t.shape[:-1], 4))

  assert math.isfinite(env.evaluate(far_outside, target_angle=0.5))


@pytest.mark.parametrize("precompute", [True, False])
def test_non_finite_target_raises(env: BarAngleTrajOptEnv, precompute: bool) -> None:
  def nan_pitch(t: np.ndarray) -> np.ndarray:
    return np.broadcast_to([float("nan"), 0.0, 0.0, -0.4], (*t.shape[:-1], 4))

  with pytest.raises(ValueError, match="non-finite"):
    env.evaluate(nan_pitch, target_angle=0.5, precompute=precompute)


def test_wrong_output_shape_raises(env: BarAngleTrajOptEnv) -> None:
  def three_joints(t: np.ndarray) -> np.ndarray:
    return np.broadcast_to([-0.3, 0.0, 0.0], (*t.shape[:-1], 3))

  with pytest.raises(ValueError, match="must return shape"):
    env.evaluate(three_joints, target_angle=0.5)


def test_unvectorized_law_raises(env: BarAngleTrajOptEnv) -> None:
  """A law that ignores the batch axis returns the wrong shape rather than
  silently driving every step from one sample."""
  with pytest.raises(ValueError, match="must return shape"):
    env.evaluate(lambda t: np.array([-0.3, 0.0, 0.0, -0.4]), target_angle=0.5)


def test_example_push_reaches_target(env: BarAngleTrajOptEnv) -> None:
  """The searched demo trajectory lands the bar near its target (0.005 rad
  when it was found; the loose bound is headroom for physics-engine drift)."""
  cost, traj = env.evaluate(two_phase_push, TARGET_ANGLE, return_trajectory=True)
  assert cost < 0.2
  # And the bar is parked, not swinging through the target at the buzzer.
  final_speed = abs(traj["bar_angle"][-1] - traj["bar_angle"][-2]) / env.timestep
  assert final_speed < 0.5


def test_trajectory_output_shapes(env: BarAngleTrajOptEnv) -> None:
  cost, traj = env.evaluate(hold_spawn, 0.5, return_trajectory=True)
  n = env.num_steps
  assert traj["time"].shape == (n,)
  assert traj["bar_angle"].shape == (n,)
  assert traj["joint_pos"].shape == (n, 4)
  assert traj["joint_target"].shape == (n, 4)
  assert np.isfinite(traj["joint_pos"]).all()
