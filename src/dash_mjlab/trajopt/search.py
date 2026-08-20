"""Black-box search over control laws for the trajopt environments.

The env scores control laws; an optimizer searches vectors. A
:class:`Parameterization` bridges the two -- it owns the box the optimizer
samples in and decodes a vector into a law -- and :class:`Objective` is the
composition the optimizer actually calls::

  env = BarAngleTrajOptEnv()
  objective = Objective(env, TwoPhaseSmoothstep(), target_angle=-0.5)
  result = minimize(objective, RandomSearch(objective.bounds), n_iter=200)
  env.evaluate(objective.parameterization(result.best_params), -0.5, render=True)

The env speaks normalized units -- phase in [0, 1), joint targets in [-1, 1]
with 0 the spawn pose -- so nothing here needs to know about radians, joint
limits or the horizon in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
from jaxtyping import Float

from dash_mjlab.trajopt.bar_env import BarAngleTrajOptEnv, ControlLaw


class Parameterization(Protocol):
  """A finite-dimensional slice of function space: a parameter vector in a box
  decodes to one control law."""

  @property
  def bounds(self) -> Float[np.ndarray, "D 2"]:
    """Per-parameter (low, high). Defines both the domain and D."""
    ...

  def __call__(self, params: Float[np.ndarray, " D"]) -> ControlLaw: ...


@dataclass
class Objective:
  """Parameters -> rollout cost. The only thing the optimizer ever calls."""

  env: BarAngleTrajOptEnv
  parameterization: Parameterization
  target_angle: float
  precompute: bool = True

  @property
  def bounds(self) -> Float[np.ndarray, "D 2"]:
    return self.parameterization.bounds

  @property
  def dim(self) -> int:
    return len(self.bounds)

  def __call__(self, params: Float[np.ndarray, "... D"]) -> Float[np.ndarray, "..."]:
    """Cost of each parameter vector. Batch axes are the caller's; the rollouts
    themselves are serial, one MuJoCo instance."""
    params = np.asarray(params, dtype=float)
    batch_shape = params.shape[:-1]
    costs = [self._evaluate(p) for p in params.reshape(-1, params.shape[-1])]
    return np.array(costs).reshape(batch_shape)

  def _evaluate(self, params: Float[np.ndarray, " D"]) -> float:
    cost = self.env.evaluate(
      self.parameterization(params),
      self.target_angle,
      precompute=self.precompute,
    )
    return cast(float, cost)


def _smoothstep(a: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
  a = np.clip(a, 0.0, 1.0)
  return 3 * a**2 - 2 * a**3


@dataclass(frozen=True)
class TwoPhaseSmoothstep:
  """Spawn -> pose A -> pose B, each leg a smoothstep in phase.

  D = 12: pose A (4 joints), pose B (4 joints), then (start, duration) for each
  leg. This is the family the demo push in ``trajopt.example`` came out of.
  """

  min_duration: float = 0.01

  @property
  def bounds(self) -> Float[np.ndarray, "D 2"]:
    pose = [(-1.0, 1.0)] * 4
    leg_timing = [(0.0, 1.0), (self.min_duration, 1.0)]
    return np.array([*pose, *pose, *leg_timing, *leg_timing], dtype=float)

  def __call__(self, params: Float[np.ndarray, " D"]) -> ControlLaw:
    pose_a, pose_b = params[:4], params[4:8]
    start_a, duration_a, start_b, duration_b = params[8:]

    def control_law(s: Float[np.ndarray, "... 1"]) -> Float[np.ndarray, "... 4"]:
      leg_a = _smoothstep((s - start_a) / duration_a)
      leg_b = _smoothstep((s - start_b) / duration_b)
      return leg_a * pose_a + leg_b * (pose_b - pose_a)

    return control_law


class Optimizer(Protocol):
  """Ask for candidates, tell it what they cost. Minimization."""

  def ask(self, n: int) -> Float[np.ndarray, "n D"]:
    """Propose ``n`` parameter vectors inside the bounds."""
    ...

  def tell(
    self, params: Float[np.ndarray, "n D"], costs: Float[np.ndarray, " n"]
  ) -> None:
    """Absorb the observed costs of previously asked parameters."""
    ...


@dataclass
class Result:
  best_params: Float[np.ndarray, " D"]
  best_cost: float
  params: Float[np.ndarray, "N D"]
  costs: Float[np.ndarray, " N"]


class RandomSearch:
  """Uniform sampling in the box. Baseline, and a live check that the wiring
  works before the real optimizer lands."""

  def __init__(self, bounds: Float[np.ndarray, "D 2"], seed: int = 0):
    self.bounds = np.asarray(bounds, dtype=float)
    self.rng = np.random.default_rng(seed)

  def ask(self, n: int) -> Float[np.ndarray, "n D"]:
    low, high = self.bounds[:, 0], self.bounds[:, 1]
    return self.rng.uniform(low, high, size=(n, len(self.bounds)))

  def tell(
    self, params: Float[np.ndarray, "n D"], costs: Float[np.ndarray, " n"]
  ) -> None:
    del params, costs  # Memoryless.


@dataclass
class BayesianOptimizer:
  """Surrogate-model search over the parameter box. PORT TARGET: everything
  below raises; the loop in ``minimize`` needs nothing else."""

  bounds: Float[np.ndarray, "D 2"]
  seed: int = 0
  n_initial: int = 16
  observed_params: list[Float[np.ndarray, " D"]] = field(default_factory=list)
  observed_costs: list[float] = field(default_factory=list)

  def ask(self, n: int) -> Float[np.ndarray, "n D"]:
    """Design of experiments while under ``n_initial`` observations, then
    maximize the acquisition function over the fitted surrogate."""
    raise NotImplementedError

  def tell(
    self, params: Float[np.ndarray, "n D"], costs: Float[np.ndarray, " n"]
  ) -> None:
    """Append the observations and refit the surrogate."""
    raise NotImplementedError


def minimize(
  objective: Objective,
  optimizer: Optimizer,
  n_iter: int,
  batch_size: int = 1,
) -> Result:
  """Run ``n_iter`` ask/tell rounds and return the best candidate seen."""
  all_params, all_costs = [], []
  for _ in range(n_iter):
    params = optimizer.ask(batch_size)
    costs = objective(params)
    optimizer.tell(params, costs)
    all_params.append(params)
    all_costs.append(costs)

  params, costs = np.concatenate(all_params), np.concatenate(all_costs)
  best = int(np.argmin(costs))
  return Result(params[best], float(costs[best]), params, costs)
