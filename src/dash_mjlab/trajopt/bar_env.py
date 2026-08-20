"""Trajectory-optimization evaluation environment for the bar-angle task.

Not an RL environment. This is a deterministic black-box cost evaluator: a
collaborator hands over one vectorized control law -- phase in, right-arm
joint targets out -- and gets back a scalar cost, the angular
distance between where the bar was asked to point and where it points when the
clock runs out. Same law in, same cost out, every time: plain single-threaded
MuJoCo on CPU, no randomization, no torch.

The scene is the same robot and bar as the RL task ``Mjlab-Bar-Angle-Dash-
UpperBody`` (the specs are shared, not copied), minus the RL stack: a
horizontal bar on a vertical hinge in front of the robot. Gravity has no
torque about the hinge, so the bar stays wherever it is pushed, less what the
joint damping bleeds off.

Only the right arm is driven. The law's output columns map to, in order:

  0  r_shoulder_pitch
  1  r_shoulder_roll
  2  r_shoulder_yaw
  3  r_elbow_pitch

The law works in normalized units, so a search never has to know the robot:
it takes a phase in [0, 1) (the fraction of the horizon elapsed) with a
trailing axis of size one, and returns four numbers in [-1, 1]. The env
decodes those into radians -- 0 holds the spawn pose, +-1 reaches the joint's
upper/lower limit -- so the do-nothing law is the zero function and the
reachable set is exactly the cube. Values outside it are clipped.

The batch axes are the caller's: the env may query the whole horizon in one
call or one step at a time (see ``precompute``), and a properly vectorized law
cannot tell the difference.

Tracking is a plain PD torque law, kp * (desired - q) - kd * qd, clamped to
the 30 Nm effort limit. The default kp = 20 is deliberately soft -- gravity
sags the shoulder visibly below its commanded position -- so the trajectories
have to reason about dynamics, not just kinematics. The left arm is PD-held at
the spawn pose with the stiff gains the RL task trains with; it never reaches
the bar.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import mujoco
import numpy as np
from jaxtyping import Float

if TYPE_CHECKING:
  from mjviser import ViserMujocoScene

from dash_mjlab.robots.dash_upper_body_constants import (
  ARMS_READY_KEYFRAME,
  TORSO_MOUNT_HEIGHT,
  get_spec,
)
from dash_mjlab.tasks.bar_angle.env_cfgs import (
  BAR_HEIGHT,
  BAR_LENGTH,
  PIVOT_X,
  get_bar_spec,
)

ControlLaw = Callable[[Float[np.ndarray, "... 1"]], Float[np.ndarray, "... 4"]]

ACTIVE_JOINTS: tuple[str, ...] = (
  "r_shoulder_pitch",
  "r_shoulder_roll",
  "r_shoulder_yaw",
  "r_elbow_pitch",
)

_PASSIVE_JOINTS: tuple[str, ...] = (
  "l_shoulder_pitch",
  "l_shoulder_roll",
  "l_shoulder_yaw",
  "l_elbow_pitch",
)

# The stiff hold on the parked arm, matching the RL task's actuator gains
# (dash_upper_body_constants): 200/6 on shoulder pitch and roll, 100/3 on
# shoulder yaw and elbow.
_PASSIVE_KP = np.array([200.0, 200.0, 100.0, 100.0])
_PASSIVE_KD = np.array([6.0, 6.0, 3.0, 3.0])

_EFFORT_LIMIT = 30.0
# Matches the RL entity's actuator armature, so the two setups share dynamics.
_ARMATURE = 0.01

# Spawn pose, same as the RL task's ready keyframe: arms slightly forward,
# elbows broken, hands straddling the bar's spawn line.
_SPAWN_POSE = {
  "shoulder_pitch": -0.3,
  "shoulder_roll": 0.0,
  "shoulder_yaw": 0.0,
  "elbow_pitch": -0.4,
}
assert ARMS_READY_KEYFRAME.joint_pos is not None  # Keep the two poses in sync.
assert (
  _SPAWN_POSE["shoulder_pitch"] == ARMS_READY_KEYFRAME.joint_pos[".*_shoulder_pitch"]
)
assert _SPAWN_POSE["elbow_pitch"] == ARMS_READY_KEYFRAME.joint_pos[".*_elbow_pitch"]


def _wrap_to_pi(angle: float) -> float:
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _build_model(timestep: float) -> mujoco.MjModel:
  """Robot at the origin, bar post at PIVOT_X, plus a target ghost.

  The ghost is a mocap body carrying a translucent copy of the bar: rotating
  its quaternion displays the commanded angle in the viewer without touching
  the physics.
  """
  spec = get_spec()
  spec.option.timestep = timestep
  torso = spec.body("torso")
  torso.pos = np.array([0.0, 0.0, TORSO_MOUNT_HEIGHT])
  for joint in spec.joints:
    joint.armature = _ARMATURE

  frame = spec.worldbody.add_frame(pos=[PIVOT_X, 0.0, 0.0])
  spec.attach(get_bar_spec(), frame=frame)

  ghost = spec.worldbody.add_body(
    name="target_ghost", mocap=True, pos=[PIVOT_X, 0.0, 0.0]
  )
  ghost.add_geom(
    name="target_ghost_visual",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=[0.0, 0.0, BAR_HEIGHT, -BAR_LENGTH, 0.0, BAR_HEIGHT],
    size=[0.01, 0.0, 0.0],
    rgba=(0.1, 0.8, 0.2, 0.4),
    group=2,
    contype=0,
    conaffinity=0,
    density=0.0,
  )

  # A floor: the RL scene gets one from mjlab's terrain config, here it has to
  # be explicit. Purely cosmetic-plus-safety -- nothing should ever reach it.
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    rgba=(0.35, 0.38, 0.40, 1.0),
  )
  # Lighting: the compiled spec would otherwise only have the dim default
  # headlight, which renders video frames nearly black. One overhead light
  # with shadows for depth, one fill from the camera side.
  spec.worldbody.add_light(
    name="overhead",
    pos=[0.3, 0.0, 2.5],
    dir=[0.0, 0.0, -1.0],
    castshadow=True,
  )
  spec.worldbody.add_light(
    name="fill",
    pos=[1.5, -1.0, 1.5],
    dir=[-0.6, 0.45, -0.4],
    castshadow=False,
  )
  # Camera for offscreen video capture: from the robot's front-right, kept
  # aimed at the torso, which puts both the arm and the whole bar arc in
  # frame.
  spec.worldbody.add_camera(
    name="video",
    mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
    targetbody="torso",
    pos=[1.5, -1.1, 1.3],
  )
  return spec.compile()


class BarAngleTrajOptEnv:
  """Deterministic rollout evaluator for control-law candidates.

  The control law maps phase in [0, 1), shape ``(..., 1)``, to normalized
  right-arm joint targets in [-1, 1], shape ``(..., 4)``, columns in the order
  ``ACTIVE_JOINTS``: r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw,
  r_elbow_pitch. Zero holds the spawn pose; the env decodes to radians and
  clips. Typical use::

    def control_law(s):  # (..., 1) -> (..., 4)
      return np.concatenate(
        [shoulder_pitch(s), shoulder_roll(s), shoulder_yaw(s), elbow_pitch(s)],
        axis=-1,
      )

    env = BarAngleTrajOptEnv()
    cost = env.evaluate(control_law, target_angle=0.6)

  One instance can evaluate any number of candidates; each ``evaluate`` starts
  from the identical initial state. The instance is not thread-safe (it owns a
  single MjData); create one per worker for parallel search.
  """

  def __init__(
    self,
    horizon: float = 5.0,
    kp: float = 20.0,
    kd: float = 1.0,
    timestep: float = 0.005,
    initial_bar_angle: float = 0.0,
  ):
    self.horizon = horizon
    self.kp = kp
    self.kd = kd
    self.timestep = timestep
    self.initial_bar_angle = initial_bar_angle

    self.model = _build_model(timestep)
    self.data = mujoco.MjData(self.model)

    def qadr(name: str) -> int:
      return self.model.joint(name).qposadr[0]

    def vadr(name: str) -> int:
      return self.model.joint(name).dofadr[0]

    self._active_qadr = np.array([qadr(n) for n in ACTIVE_JOINTS])
    self._active_vadr = np.array([vadr(n) for n in ACTIVE_JOINTS])
    self._active_range = np.array(
      [self.model.joint(n).range for n in ACTIVE_JOINTS]
    )  # (4, 2)
    self._passive_qadr = np.array([qadr(n) for n in _PASSIVE_JOINTS])
    self._passive_vadr = np.array([vadr(n) for n in _PASSIVE_JOINTS])
    # Attaching the bar spec prefixes its names (e.g. "/bar_joint"), so find
    # the joint by suffix rather than assuming the exact prefix convention.
    (bar_joint,) = [
      self.model.joint(i).name
      for i in range(self.model.njnt)
      if self.model.joint(i).name.endswith("bar_joint")
    ]
    self._bar_qadr = qadr(bar_joint)
    self._bar_vadr = vadr(bar_joint)

    def spawn(names: tuple[str, ...]) -> np.ndarray:
      return np.array([_SPAWN_POSE[n.split("_", 1)[1]] for n in names])

    self._active_spawn = spawn(ACTIVE_JOINTS)
    self._passive_spawn = spawn(_PASSIVE_JOINTS)
    self._ghost_mocap_id = self.model.body("target_ghost").mocapid[0]
    # Lazily created on the first render and reused after, so repeated
    # rendered evaluations stream into the same browser tab.
    self._viser_scene: ViserMujocoScene | None = None

  @property
  def num_steps(self) -> int:
    return round(self.horizon / self.timestep)

  def bar_angle(self) -> float:
    """Current bar hinge angle (rad). 0 points at the robot, positive swings
    the free end to the robot's right."""
    return float(self.data.qpos[self._bar_qadr])

  def _reset(self, target_angle: float) -> None:
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[self._active_qadr] = self._active_spawn
    self.data.qpos[self._passive_qadr] = self._passive_spawn
    self.data.qpos[self._bar_qadr] = self.initial_bar_angle
    # Point the ghost at the target: the ghost geom is a copy of the bar's, so
    # the same hinge angle is a pure z rotation.
    self.data.mocap_quat[self._ghost_mocap_id] = [
      math.cos(target_angle / 2.0),
      0.0,
      0.0,
      math.sin(target_angle / 2.0),
    ]
    mujoco.mj_forward(self.model, self.data)

  def _decode(self, action: Float[np.ndarray, "M 4"]) -> Float[np.ndarray, "M 4"]:
    """Normalized joint targets in [-1, 1] -> radians: 0 is the spawn pose,
    +-1 the joint's upper/lower limit. Piecewise linear, since the spawn pose
    is not centred in its range."""
    action = np.clip(action, -1.0, 1.0)
    reach_up = self._active_range[:, 1] - self._active_spawn
    reach_down = self._active_spawn - self._active_range[:, 0]
    return self._active_spawn + action * np.where(action >= 0.0, reach_up, reach_down)

  def _query(
    self, control_law: ControlLaw, times: Float[np.ndarray, " M"]
  ) -> Float[np.ndarray, "M 4"]:
    """Desired joint positions (rad) at `times`, decoded from the law's
    normalized output."""
    phases = times[:, None] / self.horizon
    action = np.asarray(control_law(phases), dtype=float)
    expected = (len(times), len(ACTIVE_JOINTS))
    if action.shape != expected:
      raise ValueError(
        f"Control law given phases of shape {phases.shape} must return "
        f"shape {expected} ({', '.join(ACTIVE_JOINTS)}), got {action.shape}."
      )
    if not np.isfinite(action).all():
      bad = phases[~np.isfinite(action).all(axis=-1)][0, 0]
      raise ValueError(f"Control law returned a non-finite value at phase {bad:.3f}.")
    return self._decode(action)

  def _get_viser_scene(self) -> ViserMujocoScene:
    if self._viser_scene is None:
      import viser
      from mjviser import ViserMujocoScene

      # The server announces its own URL (default http://localhost:8080) and
      # stays up for the lifetime of this env instance.
      server = viser.ViserServer(label="dash-bar-trajopt")
      self._viser_scene = ViserMujocoScene(server, self.model, num_envs=1)
    return self._viser_scene

  def evaluate(
    self,
    control_law: ControlLaw,
    target_angle: float,
    *,
    precompute: bool = True,
    return_trajectory: bool = False,
    render: bool = False,
    render_backend: Literal["viser", "native"] = "viser",
    video_path: str | None = None,
    video_fps: int = 30,
    video_size: tuple[int, int] = (480, 640),
  ) -> float | tuple[float, dict[str, np.ndarray]]:
    """Roll out the control law and return the terminal cost.

    Args:
      control_law: vectorized map from phase (shape ``(..., 1)``, 0 to 1) to
        normalized joint targets (shape ``(..., 4)``, -1 to 1) in
        ``ACTIVE_JOINTS`` order. Zero holds the spawn pose, +-1 is the joint's
        limit; values outside the cube are clipped, non-finite raise.
      target_angle: commanded bar angle in rad.
      precompute: query the law once for the whole horizon before stepping.
        The law is a function of time alone, so this is equivalent to asking
        it step by step and saves ``num_steps - 1`` calls. Set False to have
        it queried one step at a time, which is what a state-dependent law
        would need.
      return_trajectory: also return the full rollout -- keys ``time``,
        ``bar_angle``, ``joint_pos``, ``joint_target`` -- for debugging a
        candidate. Off by default so the search loop pays nothing for it.
      render: play the rollout at real time in a viewer. For watching single
        candidates, not for use inside a search loop.
      render_backend: ``"viser"`` (default) serves the scene to the browser --
        the server starts on the first rendered call, prints its URL, and is
        reused by later calls, so keep the process alive while watching. If no
        browser is connected yet, the rollout waits for one so the animation
        is not played into an empty room. ``"native"`` opens a MuJoCo window
        instead (needs a local display).
      video_path: if set, record the rollout offscreen to this file (e.g.
        ``"push.mp4"``). Independent of ``render``, runs at full speed, and
        works headless. Frames come from the built-in "video" camera.
      video_fps: frame rate of the written video.
      video_size: (height, width) of the written video.

    Returns:
      The cost: |shortest angular distance between target and final bar
      angle|, in radians. With ``return_trajectory``, a ``(cost, trajectory)``
      tuple instead.
    """
    self._reset(target_angle)

    n = self.num_steps
    # Always allocated: a few float arrays per rollout are noise next to the
    # physics, and it keeps every variable unconditionally bound.
    times = np.arange(n) * self.timestep
    schedule = self._query(control_law, times) if precompute else None
    bar_angles = np.empty(n)
    joint_pos = np.empty((n, 4))
    joint_target = np.empty((n, 4))

    viewer = None
    scene = None
    if render:
      if render_backend == "viser":
        scene = self._get_viser_scene()
        # Don't roll out into an empty room: without this, the animation has
        # already played by the time the user opens the printed URL and all
        # they ever see is the final pose.
        if not scene.server.get_clients():
          print("Waiting for a browser to connect before starting the rollout...")
          while not scene.server.get_clients():
            time.sleep(0.1)
          # Give the page a beat to finish loading the meshes.
          time.sleep(1.0)
        scene.update_from_mjdata(self.data)
      elif render_backend == "native":
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(self.model, self.data)
      else:
        raise ValueError(f"Unknown render_backend: {render_backend!r}")

    renderer = None
    frames: list[np.ndarray] = []
    if video_path is not None:
      renderer = mujoco.Renderer(self.model, height=video_size[0], width=video_size[1])

    try:
      for k in range(n):
        t = times[k]
        if schedule is not None:
          desired = schedule[k]
        else:
          desired = self._query(control_law, times[k : k + 1])[0]

        q = self.data.qpos[self._active_qadr]
        qd = self.data.qvel[self._active_vadr]
        tau_active = self.kp * (desired - q) - self.kd * qd
        q_p = self.data.qpos[self._passive_qadr]
        qd_p = self.data.qvel[self._passive_vadr]
        tau_passive = _PASSIVE_KP * (self._passive_spawn - q_p) - _PASSIVE_KD * qd_p

        self.data.qfrc_applied[self._active_vadr] = np.clip(
          tau_active, -_EFFORT_LIMIT, _EFFORT_LIMIT
        )
        self.data.qfrc_applied[self._passive_vadr] = np.clip(
          tau_passive, -_EFFORT_LIMIT, _EFFORT_LIMIT
        )
        mujoco.mj_step(self.model, self.data)

        bar_angles[k] = self.bar_angle()
        joint_pos[k] = self.data.qpos[self._active_qadr]
        joint_target[k] = desired
        if renderer is not None and t >= len(frames) / video_fps:
          renderer.update_scene(self.data, camera="video")
          frames.append(renderer.render())
        if scene is not None:
          # ~66 Hz scene sync is enough for the browser; the sleep paces the
          # whole rollout to real time.
          if k % 3 == 0:
            scene.update_from_mjdata(self.data)
          time.sleep(self.timestep)
        elif viewer is not None:
          if not viewer.is_running():
            break
          viewer.sync()
          time.sleep(self.timestep)
      if scene is not None:
        scene.update_from_mjdata(self.data)  # Show the final settled state.
      if renderer is not None and video_path is not None:
        # imageio over mediapy: it ships its own ffmpeg binary, so writing
        # works on machines with no system ffmpeg installed.
        import imageio

        imageio.mimwrite(video_path, cast("list[Any]", frames), fps=video_fps)
    finally:
      if viewer is not None:
        viewer.close()
      if renderer is not None:
        renderer.close()

    cost = abs(_wrap_to_pi(target_angle - self.bar_angle()))
    if return_trajectory:
      return cost, {
        "time": times,
        "bar_angle": bar_angles,
        "joint_pos": joint_pos,
        "joint_target": joint_target,
      }
    return cost
