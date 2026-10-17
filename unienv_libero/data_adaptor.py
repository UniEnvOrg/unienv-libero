from typing import Dict, Any, Optional, Union, Tuple
import os
import numpy as np
from functools import cached_property

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import Space, BoxSpace, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    annotate,
    depth_metadata,
    image_metadata,
    metadata,
    proprio_metadata,
)
from unienv_data.base import BatchBase, IndexableType
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.batches.combined_batch import convert_single_index_to_batch, convert_index_to_batch
import h5py

__all__ = [
    'LiberoHDF5Data',
    'libero_space_metadata',
]

# Cameras of the LIBERO datasets: observation key -> robosuite camera name.
_LIBERO_CAMERAS = {
    "agentview": "agentview",
    "eye_in_hand": "robot0_eye_in_hand",
}
# robosuite OSC_POSE (LIBERO's controller) maps the normalized [-1, 1] action
# to +-0.05 m position and +-0.5 rad axis-angle deltas; the gripper command is
# -1 (open) .. 1 (close).
_OSC_POSE_ACTION_RANGE = [
    [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5, -1.0],
    [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0],
]


def _last_dim(space: Space) -> Optional[int]:
    shape = getattr(space, "shape", None)
    return shape[-1] if shape else None


def libero_space_metadata(
    space: DictSpace,
    image_convention: Optional[str] = "opengl",
) -> Dict[str, Dict[str, Any]]:
    """``annotate`` mapping for the space of one LIBERO demo group (``data/demo_<i>``).

    Keyed on the layout written by LIBERO's ``scripts/create_dataset.py``;
    only paths present in ``space`` are returned. ``image_convention`` is the
    file's ``data.attrs["macros_image_convention"]`` (robosuite's
    ``IMAGE_CONVENTION``): with ``"opengl"`` the images are stored upside down
    (``flipped="vertical"``).

    - ``obs/{agentview,eye_in_hand}_rgb``: uint8 (H, W, 3) RGB renders
    - ``obs/{agentview,eye_in_hand}_depth`` (``--use-depth``): robosuite's raw
      depth buffer, normalized and non-linear, not meters, so no ``units``
    - ``obs/ee_pos`` (m), ``obs/ee_ori`` (axis-angle, rad), ``obs/ee_states``
      (both): ``robot0_eef_pos`` / ``robot0_eef_quat`` in the world frame
    - ``obs/joint_states``: arm joint positions (rad); ``obs/gripper_states``:
      gripper finger joint positions (m)
    - ``robot_states``: gripper qpos (2) + eef pos (3) + eef quat (4, xyzw)
    - ``actions``: OSC_POSE delta, normalized to [-1, 1]
    - ``states``: the flattened MuJoCo state (privileged); ``rewards`` / ``dones``
    """
    flip = {"flipped": "vertical"} if image_convention == "opengl" else {}
    convention = {"image_convention": image_convention} if image_convention is not None else {}
    obs_space = space.spaces.get("obs")
    obs_keys = obs_space.spaces if isinstance(obs_space, DictSpace) else {}
    table: Dict[str, Dict[str, Any]] = {}

    for key, camera in _LIBERO_CAMERAS.items():
        if f"{key}_rgb" in obs_keys:
            table[f"obs/{key}_rgb"] = image_metadata(source=camera, **convention, **flip)
        if f"{key}_depth" in obs_keys:
            depth_space = obs_keys[f"{key}_depth"]
            table[f"obs/{key}_depth"] = depth_metadata(
                units=None, layout="HWC" if len(depth_space.shape) == 3 else "HW",
                source=camera, **convention, **flip,
                label="raw robosuite depth buffer (normalized, non-linear; not meters)",
            )

    proprio = {
        "ee_pos": proprio_metadata(units="m", frame="world", names=["x", "y", "z"]),
        "ee_ori": proprio_metadata(units="rad", frame="world", rotation="axis_angle", names=["ax", "ay", "az"]),
        "ee_states": proprio_metadata(
            units=["m", "m", "m", "rad", "rad", "rad"], frame="world", rotation="axis_angle",
            names=["x", "y", "z", "ax", "ay", "az"],
        ),
        "joint_states": proprio_metadata(units="rad"),
        "gripper_states": proprio_metadata(units="m", label="gripper finger joint positions"),
    }
    for key, md in proprio.items():
        if key in obs_keys:
            table[f"obs/{key}"] = md

    if "actions" in space.spaces:
        if _last_dim(space["actions"]) == 7:
            table["actions"] = action_metadata(
                control="eef_pose", mode="delta", normalized=True, range=_OSC_POSE_ACTION_RANGE,
                frame="world", rotation="axis_angle",
                names=["x", "y", "z", "ax", "ay", "az", "gripper"],
                units=["m", "m", "m", "rad", "rad", "rad", "normalized"],
                controller="OSC_POSE", label="OSC_POSE delta; gripper -1 open, 1 close",
            )
        else:
            table["actions"] = action_metadata()
    if "robot_states" in space.spaces and _last_dim(space["robot_states"]) == 9:
        table["robot_states"] = proprio_metadata(
            frame="world", rotation="quat_xyzw",
            names=["gripper_q0", "gripper_q1", "x", "y", "z", "qx", "qy", "qz", "qw"],
            units=["m", "m", "m", "m", "m", "normalized", "normalized", "normalized", "normalized"],
        )
    if "states" in space.spaces:
        table["states"] = metadata("privileged", label="flattened MuJoCo simulator state")
    if "rewards" in space.spaces:
        table["rewards"] = metadata(None, label="reward (1 on task success)")
    if "dones" in space.spaces:
        table["dones"] = metadata(None, label="episode done flag")
    return table

class LiberoHDF5Data(BatchBase[Dict[str, Any], NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]):
    """Adaptor to convert Libero HDF5 data to UniEnvPy format."""

    def __init__(
        self,
        hdf5_path: Union[str, os.PathLike],
    ):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.episode_root = self.hdf5_file['data']
        self.n_episodes = len(self.episode_root)
        # Build space
        first_episode = self.episode_root[next(iter(self.episode_root.keys()))]
        _, _, first_space = HDF5Storage.build_space_from_hdf5_file(first_episode)
        image_convention = self.episode_root.attrs.get("macros_image_convention", "opengl")
        if isinstance(image_convention, bytes):
            image_convention = image_convention.decode()
        first_space = annotate(
            first_space,
            libero_space_metadata(first_space, image_convention=str(image_convention)),
        )
        self.episode_names = list(self.episode_root.keys())
        self.episode_lengths = np.array([self.episode_root[ep_name]['actions'].shape[0] for ep_name in self.episode_names], dtype=np.int64)
        episode_ends = np.cumsum(self.episode_lengths)
        self.episode_starts = np.concatenate(([0], episode_ends[:-1]))
        self.length_total = int(episode_ends[-1])

        super().__init__(
            first_space,
            DictSpace(
                NumpyComputeBackend,
                {
                    "step_i": BoxSpace(
                        NumpyComputeBackend,
                        low=0,
                        high=np.inf,
                        dtype=np.int64,
                        shape=(),
                        metadata=metadata(None, label="step index within the episode"),
                    ),
                    "episode_id": BoxSpace(
                        NumpyComputeBackend,
                        low=0,
                        high=np.inf,
                        dtype=np.int64,
                        shape=(),
                        metadata=metadata(None, label="episode index within the file"),
                    ),
                }
            )
        )

    def __len__(self) -> int:
        return self.length_total

    def num_trajectories(self) -> int:
        return self.n_episodes
    
    def get_episode_at(self, episode_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        episode_name = self.episode_names[episode_idx]
        episode_data = self.episode_root[episode_name]
        read_data = HDF5Storage.get_from(
            episode_data,
            self.single_space,
            slice(None),
            reduce_io=False
        )
        metadata = {
            "step_i": np.arange(self.episode_lengths[episode_idx], dtype=np.int64),
            "episode_id": episode_idx,
        }
        return read_data, metadata

    def get_episode_subset(
        self,
        episode_idx: int,
        step_idx: Union[IndexableType, np.ndarray]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        episode_name = self.episode_names[episode_idx]
        episode_data = self.episode_root[episode_name]
        read_data = HDF5Storage.get_from(
            episode_data,
            self.single_space,
            step_idx,
            reduce_io=False
        )

        B = read_data['actions'].shape[0] if not isinstance(step_idx, int) else 1
        metadata = {
            "step_i": np.arange(int(self.episode_lengths[episode_idx]), dtype=np.int64)[step_idx] if not isinstance(step_idx, int) else np.array(step_idx, dtype=np.int64),
            "episode_id": np.full((B,), episode_idx, dtype=np.int64) if not isinstance(step_idx, int) else np.array(episode_idx, dtype=np.int64),
        }
        return read_data, metadata

    def get_at_with_metadata(self, idx):
        if isinstance(idx, int):
            episode_idx, step_idx = convert_single_index_to_batch(
                self.backend,
                self.episode_starts,
                len(self),
                idx,
                device=None,
            )
            return self.get_episode_subset(episode_idx, step_idx)
        else:
            batch_size, batch_list = convert_index_to_batch(
                self.backend,
                self.episode_starts,
                len(self),
                idx,
                device=None,
            )
            result_space = sbu.batch_space(
                self.single_space,
                batch_size,
            )
            metadata_space = sbu.batch_space(
                self.single_metadata_space,
                batch_size,
            )
            result = result_space.create_empty()
            metadata = metadata_space.create_empty()
            
            for batch_index, index_into_batch, mask in batch_list:
                batch_result, metadata_result = self.get_episode_subset(
                    batch_index,
                    index_into_batch,
                )
                result = sbu.set_at(
                    result_space,
                    result,
                    mask,
                    batch_result,
                )
                metadata = sbu.set_at(
                    metadata_space,
                    metadata,
                    mask,
                    metadata_result,
                )
            return result, metadata

    def get_at(self, idx):
        return self.get_at_with_metadata(idx)[0]
    
    @cached_property
    def language_instruction(self) -> str:
        hdf5_path_base = os.path.basename(self.hdf5_path)
        # Format: XXXX_SCENE[0-10]_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf_demo.hdf5
        first_pos_scene = hdf5_path_base.find("SCENE")
        if first_pos_scene == -1:
            name_base = hdf5_path_base
        else:
            name_base = hdf5_path_base[first_pos_scene + len("SCENE"):]
            underscore_pos = name_base.find("_")
            if underscore_pos != -1:
                name_base = name_base[underscore_pos + 1:]
        instruction_end = name_base.rfind("_demo")
        if instruction_end == -1:
            instruction_end = name_base.rfind(".hdf5")
        instruction = name_base[:instruction_end].replace("_", " ")
        return instruction