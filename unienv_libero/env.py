"""LIBERO tasks as a UniEnv environment, in the layout of the LIBERO demonstration datasets.

:class:`FromLiberoEnv` runs one task of a LIBERO suite (``libero_spatial``,
``libero_object``, ``libero_goal``, ``libero_10``, ``libero_90``) through LIBERO's
``OffScreenRenderEnv`` (robosuite on MuJoCo). Its observations use the keys,
dtypes, image convention and space metadata of the demonstration HDF5 files read
by :class:`~unienv_libero.data_adaptor.LiberoHDF5Data` (LIBERO's
``scripts/create_dataset.py`` layout), so a policy trained on the data runs in
the environment unchanged::

    obs/agentview_rgb    <- agentview_image            uint8 (H, W, 3)
    obs/eye_in_hand_rgb  <- robot0_eye_in_hand_image   uint8 (H, W, 3)
    obs/ee_pos           <- robot0_eef_pos             float64 (3,)
    obs/ee_ori           <- axis-angle of robot0_eef_quat
    obs/ee_states        <- ee_pos + ee_ori             float64 (6,)
    obs/gripper_states   <- robot0_gripper_qpos        float64 (2,)
    obs/joint_states     <- robot0_joint_pos           float64 (7,)
    (with use_depth: obs/{agentview,eye_in_hand}_depth <- *_depth, float32 (H, W, 1))

Images keep robosuite's convention, as in the datasets: with robosuite's default
``"opengl"`` convention they are upside down (metadata ``flipped="vertical"``);
:meth:`FromLiberoEnv.render` returns an upright frame. The context is the task's
language instruction (as in the benchmark and the dataset file names); actions
are LIBERO's normalized OSC_POSE deltas in [-1, 1]^7 (gripper -1 open, 1 close).

The spaces and the observation mapping (:func:`libero_env_spaces`,
:func:`libero_observation`) only need numpy and unienv; LIBERO itself is imported
when an environment is built (``pip install "unienv_libero[env]"``).
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.env_base.env import Env
from unienv_interface.space import BoxSpace, DictSpace, TextSpace
from unienv_interface.space.space_utils.metadata_utils import annotate, metadata

from .data_adaptor import libero_space_metadata

__all__ = [
    "FromLiberoEnv",
    "LIBERO_SUITES",
    "LIBERO_MAX_EPISODE_STEPS",
    "libero_env_spaces",
    "libero_observation",
]

LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")

#: Episode step limits per suite used by the common LIBERO evaluations (OpenVLA,
#: LeRobot): a margin over the longest training demonstration of the suite.
LIBERO_MAX_EPISODE_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

_CAMERAS = (("agentview", "agentview"), ("eye_in_hand", "robot0_eye_in_hand"))
_B = NumpyComputeBackend


def libero_env_spaces(
    image_size: Tuple[int, int] = (128, 128),
    use_depth: bool = False,
    image_convention: Optional[str] = "opengl",
) -> Tuple[DictSpace, BoxSpace]:
    """Observation and action spaces of :class:`FromLiberoEnv`, with dataset metadata.

    The observation space equals ``LiberoHDF5Data(...).single_space["obs"]`` of a
    dataset recorded at the same image size (metadata included); the action space
    is the dataset's ``actions`` with the controller's [-1, 1] bounds.
    """
    height, width = (int(image_size[0]), int(image_size[1]))
    obs = {}
    for key, _ in _CAMERAS:
        obs[f"{key}_rgb"] = BoxSpace(_B, 0, 255, np.uint8, shape=(height, width, 3))
        if use_depth:
            obs[f"{key}_depth"] = BoxSpace(_B, -np.inf, np.inf, np.float32, shape=(height, width, 1))
    for key, dim in (("ee_pos", 3), ("ee_ori", 3), ("ee_states", 6), ("gripper_states", 2), ("joint_states", 7)):
        obs[key] = BoxSpace(_B, -np.inf, np.inf, np.float64, shape=(dim,))
    episode = DictSpace(_B, {
        "obs": DictSpace(_B, obs),
        "actions": BoxSpace(_B, -1.0, 1.0, np.float64, shape=(7,)),
    })
    episode = annotate(episode, libero_space_metadata(episode, image_convention=image_convention))
    return episode["obs"], episode["actions"]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite's ``transform_utils.quat2axisangle`` for an (x, y, z, w) quaternion."""
    quat = np.asarray(quat, dtype=np.float64)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * 2.0 * np.arccos(w) / den


def libero_observation(raw: Dict[str, Any], use_depth: bool = False) -> Dict[str, np.ndarray]:
    """Map a robosuite observation of a LIBERO env to the dataset layout (see the module docstring)."""
    ee_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float64)
    ee_ori = _quat2axisangle(raw["robot0_eef_quat"])
    obs = {}
    for key, camera in _CAMERAS:
        obs[f"{key}_rgb"] = np.ascontiguousarray(raw[f"{camera}_image"], dtype=np.uint8)
        if use_depth:
            depth = np.asarray(raw[f"{camera}_depth"], dtype=np.float32)
            obs[f"{key}_depth"] = depth if depth.ndim == 3 else depth[..., None]
    obs["ee_pos"] = ee_pos
    obs["ee_ori"] = ee_ori
    obs["ee_states"] = np.concatenate([ee_pos, ee_ori])
    obs["gripper_states"] = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64)
    obs["joint_states"] = np.asarray(raw["robot0_joint_pos"], dtype=np.float64)
    return obs


def _ensure_libero_config() -> None:
    """Write LIBERO's default ``config.yaml`` if it is missing.

    ``import libero.libero`` otherwise asks on stdin where to keep the datasets,
    which blocks scripts, tests and servers. The defaults are the ones LIBERO would
    write (paths inside the installed package; datasets next to it).
    """
    config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    config_file = os.path.join(config_dir, "config.yaml")
    if os.path.exists(config_file):
        return
    spec = importlib.util.find_spec("libero")  # does not execute libero/libero/__init__.py
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "FromLiberoEnv needs LIBERO: pip install 'unienv_libero[env]'"
        )
    root = os.path.join(list(spec.submodule_search_locations)[0], "libero")
    os.makedirs(config_dir, exist_ok=True)
    paths = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.join(root, "..", "datasets"),
        "assets": os.path.join(root, "assets"),
    }
    with open(config_file, "w") as f:
        f.writelines(f"{key}: {value}\n" for key, value in paths.items())


class FromLiberoEnv(Env[NumpyArrayType, str, Dict[str, Any], NumpyArrayType, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]):
    """One LIBERO task as an unbatched UniEnv environment (see the module docstring).

    Args:
        task_suite: One of :data:`LIBERO_SUITES`.
        task_id: Index of the task in the suite.
        image_size: ``(height, width)`` of the camera images (the datasets use 128 x 128).
        use_depth: Also observe the cameras' (normalized, non-linear) depth buffers.
        max_episode_steps: Steps after which an episode is truncated; defaults to the
            suite's :data:`LIBERO_MAX_EPISODE_STEPS`. ``None`` or ``0`` for no limit.
        settle_steps: Steps of a no-op action (gripper open) run after every reset so
            that objects settle, as the OpenVLA / LeRobot evaluations do (10). They do
            not count toward ``max_episode_steps``.
        render_mode: ``"rgb_array"``: :meth:`render` returns the upright agentview image.
        seed: Seeds the choice of initial states (see :meth:`reset`).

    Episodes start from the benchmark's fixed initial states of the task (as in the
    standard evaluation): :meth:`reset` takes ``init_state_id`` (an index into
    :attr:`init_states`) or ``init_state`` (any flattened MuJoCo state, e.g. a demo's
    ``init_state`` attribute), and otherwise draws an index from its seeded RNG.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_suite: str = "libero_spatial",
        task_id: int = 0,
        *,
        image_size: Union[int, Tuple[int, int]] = 128,
        use_depth: bool = False,
        max_episode_steps: Optional[int] = -1,
        settle_steps: int = 0,
        render_mode: Optional[str] = "rgb_array",
        seed: Optional[int] = None,
    ):
        if task_suite not in LIBERO_SUITES:
            raise ValueError(f"unknown LIBERO suite {task_suite!r}; expected one of {LIBERO_SUITES}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")
        _ensure_libero_config()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import robosuite.macros as robosuite_macros

        suite = benchmark.get_benchmark_dict()[task_suite]()
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {suite.n_tasks}) for {task_suite}")
        self.task_suite = task_suite
        self.task_id = int(task_id)
        self.task = suite.get_task(task_id)
        self.language_instruction: str = self.task.language
        self.init_states: np.ndarray = np.asarray(suite.get_task_init_states(task_id))

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.use_depth = bool(use_depth)
        if max_episode_steps is not None and max_episode_steps < 0:
            max_episode_steps = LIBERO_MAX_EPISODE_STEPS[task_suite]
        self.max_episode_steps: Optional[int] = max_episode_steps or None
        self.settle_steps = int(settle_steps)
        self._render_mode = render_mode
        self.image_convention: str = robosuite_macros.IMAGE_CONVENTION

        bddl_file = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=self.image_size[0],
            camera_widths=self.image_size[1],
            camera_depths=self.use_depth,
            # The step limit and termination are handled here (robosuite's horizon
            # would otherwise raise on the step after it).
            ignore_done=True,
        )

        self.backend = NumpyComputeBackend
        self.device = None
        self.observation_space, self.action_space = libero_env_spaces(
            self.image_size, self.use_depth, self.image_convention,
        )
        self.context_space = TextSpace(
            NumpyComputeBackend, max_length=max(256, len(self.language_instruction)),
            metadata=metadata("task", label="language instruction of the LIBERO task"),
        )
        self.rng = NumpyComputeBackend.random.random_number_generator(seed)
        self._steps = 0
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ properties

    @property
    def render_mode(self) -> Optional[str]:
        return self._render_mode

    @property
    def render_fps(self) -> Optional[int]:
        return int(self._env.env.control_freq)

    @property
    def batch_size(self) -> Optional[int]:
        return None

    @property
    def num_init_states(self) -> int:
        return int(self.init_states.shape[0])

    @property
    def demo_file_name(self) -> str:
        """Name of the task's file in the LIBERO datasets (``<suite>/<name>``)."""
        return f"{self.task_suite}/{self.task.name}_demo.hdf5"

    # ------------------------------------------------------------------ episode

    def reset(
        self,
        *,
        mask=None,
        seed: Optional[int] = None,
        init_state_id: Optional[int] = None,
        init_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, np.ndarray], Dict[str, Any]]:
        """Start an episode from one of the task's initial states.

        ``init_state`` (a flattened MuJoCo state) wins over ``init_state_id``; without
        either, an initial state index is drawn from the environment's RNG, which
        ``seed`` resets first (the same seed gives the same episode).
        """
        assert mask is None, "FromLiberoEnv is not batched"
        if seed is not None:
            self.rng = NumpyComputeBackend.random.random_number_generator(seed)
            self._env.seed(seed)
        if init_state is None:
            if init_state_id is None:
                init_state_id = int(self.rng.integers(self.num_init_states))
            init_state = self.init_states[init_state_id]
        self._env.reset()
        raw = self._env.set_init_state(np.asarray(init_state, dtype=np.float64))
        for _ in range(self.settle_steps):
            raw, _, _, _ = self._env.step(self._noop_action())
        self._steps = 0
        self._last_obs = libero_observation(raw, self.use_depth)
        info = {"init_state_id": init_state_id, "success": bool(self._env.check_success())}
        return self.language_instruction, self._last_obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        raw, reward, _, info = self._env.step(action)
        self._steps += 1
        success = bool(self._env.check_success())
        self._last_obs = libero_observation(raw, self.use_depth)
        truncated = (not success) and self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        info = dict(info or {})
        info["success"] = success
        return self._last_obs, float(reward), success, truncated, info

    def _noop_action(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)

    # ------------------------------------------------------------------ rendering

    def render(self):
        if self.render_mode != "rgb_array" or self._last_obs is None:
            return None
        frame = self._last_obs["agentview_rgb"]
        # The observations keep robosuite's convention; the rendered frame is upright
        return np.ascontiguousarray(frame[::-1]) if self.image_convention == "opengl" else frame.copy()

    def close(self):
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()
            self._env = None
