from typing import Dict, Any, Optional, Union, Tuple, Sequence, List
import glob
import os
import re
import numpy as np
from functools import cached_property

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import Space, BoxSpace, DictSpace, TextSpace
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
from unienv_data.batches.combined_batch import CombinedBatch, convert_single_index_to_batch, convert_index_to_batch
import h5py

__all__ = [
    'LiberoHDF5Data',
    'LiberoSuiteData',
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

#: Room for the language instruction in the batch metadata (the LIBERO ones are < 150 characters)
_INSTRUCTION_MAX_LENGTH = 512


def _demo_sort_key(name: str):
    """``demo_<n>`` in numeric order (h5py lists ``demo_10`` before ``demo_2``)."""
    match = re.fullmatch(r"demo_(\d+)", name)
    return (0, int(match.group(1)), name) if match else (1, 0, name)


def _drop_paths(space: DictSpace, paths: Sequence[str]) -> DictSpace:
    """``space`` without the (``/``-separated, possibly nested) keys in ``paths``."""
    spaces = dict(space.spaces)
    nested: Dict[str, List[str]] = {}
    for path in paths:
        head, _, rest = str(path).strip("/").partition("/")
        if head not in spaces:
            raise KeyError(f"cannot exclude {path!r}: no key {head!r} (keys: {sorted(spaces)})")
        if rest:
            nested.setdefault(head, []).append(rest)
        else:
            spaces.pop(head)
    for head, rests in nested.items():
        if head in spaces:
            if not isinstance(spaces[head], DictSpace):
                raise KeyError(f"cannot exclude {head}/{rests[0]}: {head!r} is not a group")
            spaces[head] = _drop_paths(spaces[head], rests)
    return DictSpace(space.backend, spaces, device=space.device, metadata=space.metadata)


def _take(tree: Any, index: np.ndarray) -> Any:
    """``leaf[index]`` for every array of a nested dict."""
    if isinstance(tree, dict):
        return {key: _take(value, index) for key, value in tree.items()}
    return np.asarray(tree)[index]


class LiberoHDF5Data(BatchBase[Dict[str, Any], NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]):
    """A LIBERO demonstration HDF5 file (one task) as a UniEnv batch.

    Samples are the steps of the file's demonstrations (``data/demo_<i>``, in
    numeric order), with the space inferred from the file and annotated with
    :func:`libero_space_metadata`. Every sample's metadata holds its ``step_i``,
    ``episode_id`` (``episode_id_offset`` + the demonstration's index) and the
    task's ``language_instruction``.

    Args:
        hdf5_path: The demonstration file.
        exclude_keys: Keys left out of the space (and not read), ``/``-separated
            for nested ones, e.g. ``["states", "obs/eye_in_hand_rgb"]``.
        episode_id_offset: Added to the episode ids (to keep them unique across
            files, as :class:`LiberoSuiteData` does).
    """

    def __init__(
        self,
        hdf5_path: Union[str, os.PathLike],
        *,
        exclude_keys: Optional[Sequence[str]] = None,
        episode_id_offset: int = 0,
    ):
        self.hdf5_path = hdf5_path
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.episode_root = self.hdf5_file['data']
        self.episode_names = sorted(self.episode_root.keys(), key=_demo_sort_key)
        self.n_episodes = len(self.episode_names)
        self.episode_id_offset = int(episode_id_offset)
        # Build space
        first_episode = self.episode_root[self.episode_names[0]]
        _, _, first_space = HDF5Storage.build_space_from_hdf5_file(first_episode)
        if exclude_keys:
            first_space = _drop_paths(first_space, exclude_keys)
        image_convention = self.episode_root.attrs.get("macros_image_convention", "opengl")
        if isinstance(image_convention, bytes):
            image_convention = image_convention.decode()
        first_space = annotate(
            first_space,
            libero_space_metadata(first_space, image_convention=str(image_convention)),
        )
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
                        metadata=metadata(None, label="episode index (within the file, plus episode_id_offset)"),
                    ),
                    "language_instruction": TextSpace(
                        NumpyComputeBackend,
                        max_length=_INSTRUCTION_MAX_LENGTH,
                        metadata=metadata("task", label="language instruction of the task"),
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
        length = int(self.episode_lengths[episode_idx])
        metadata = {
            "step_i": np.arange(length, dtype=np.int64),
            "episode_id": np.full((length,), self.episode_id_offset + episode_idx, dtype=np.int64),
            "language_instruction": np.full((length,), self.language_instruction, dtype=object),
        }
        return read_data, metadata

    def get_episode_subset(
        self,
        episode_idx: int,
        step_idx: Union[IndexableType, np.ndarray]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        episode_name = self.episode_names[episode_idx]
        episode_data = self.episode_root[episode_name]
        if isinstance(step_idx, (int, np.integer, slice)):
            step_idx = int(step_idx) if isinstance(step_idx, np.integer) else step_idx
            read_data = HDF5Storage.get_from(episode_data, self.single_space, step_idx, reduce_io=False)
        else:
            step_idx = np.asarray(step_idx)
            if step_idx.dtype == np.bool_:
                step_idx = np.flatnonzero(step_idx)
            step_idx = np.where(step_idx < 0, step_idx + int(self.episode_lengths[episode_idx]), step_idx)
            # h5py only reads strictly increasing indices: read each step once, in order
            # (consecutive runs as slices), then put the rows back in the requested order
            unique, inverse = np.unique(step_idx, return_inverse=True)
            read_data = _take(
                HDF5Storage.get_from(episode_data, self.single_space, unique, reduce_io=True),
                inverse.reshape(step_idx.shape),
            )

        episode_id = self.episode_id_offset + episode_idx
        if isinstance(step_idx, int):
            metadata = {
                "step_i": np.array(step_idx, dtype=np.int64),
                "episode_id": np.array(episode_id, dtype=np.int64),
                "language_instruction": self.language_instruction,
            }
        else:
            steps = np.arange(int(self.episode_lengths[episode_idx]), dtype=np.int64)[step_idx]
            metadata = {
                "step_i": steps,
                "episode_id": np.full(steps.shape, episode_id, dtype=np.int64),
                "language_instruction": np.full(steps.shape, self.language_instruction, dtype=object),
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

    def close(self):
        hdf5_file = getattr(self, "hdf5_file", None)  # absent if __init__ failed to open it
        if hdf5_file is not None:
            hdf5_file.close()
            self.hdf5_file = None
            self.episode_root = None
    
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


class LiberoSuiteData(CombinedBatch[Dict[str, Any], NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]):
    """Several LIBERO demonstration files (e.g. a suite's directory) as one batch.

    Each file is a :class:`LiberoHDF5Data`; samples are their steps, file after
    file. The metadata of every sample adds ``batch_index`` (the file, see
    :attr:`files` / :attr:`language_instructions`) to the file's ``step_i``,
    ``language_instruction`` and ``episode_id``, which is unique across the files
    (:attr:`episode_offsets`).

    The files must share their space. Within a suite they do; across suites (or
    within ``libero_90``) the flattened simulator ``states`` has scene-dependent
    sizes: leave it out with ``exclude_keys=["states"]`` (the error names the
    keys that differ).

    Args:
        files: A directory (its ``*.hdf5`` files, sorted by name) or paths.
        exclude_keys: Passed to every :class:`LiberoHDF5Data`.
        use_thread_pool: Read the files of a batch in parallel threads.
    """

    def __init__(
        self,
        files: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        *,
        exclude_keys: Optional[Sequence[str]] = None,
        use_thread_pool: bool = False,
    ):
        if isinstance(files, (str, os.PathLike)):
            if not os.path.isdir(files):
                raise ValueError(f"{files!r} is not a directory; pass a directory or a list of files")
            paths = sorted(glob.glob(os.path.join(os.fspath(files), "*.hdf5")))
        else:
            paths = [os.fspath(path) for path in files]
        if len(paths) < 2:
            raise ValueError(
                f"LiberoSuiteData combines several files, got {len(paths)}; use LiberoHDF5Data for one file"
            )
        datasets: List[LiberoHDF5Data] = []
        try:
            offset = 0
            for path in paths:
                dataset = LiberoHDF5Data(path, exclude_keys=exclude_keys, episode_id_offset=offset)
                datasets.append(dataset)
                offset += dataset.num_trajectories()
            mismatched = self._mismatched_keys(datasets)
            if mismatched:
                raise ValueError(
                    "the files do not share their space; keys that differ: "
                    + ", ".join(mismatched)
                    + " (leave them out with exclude_keys=[...])"
                )
        except BaseException:
            for dataset in datasets:
                dataset.close()
            raise
        self.files: List[str] = paths
        self.episode_offsets = np.array([d.episode_id_offset for d in datasets], dtype=np.int64)
        super().__init__(datasets, use_thread_pool=use_thread_pool)

    @staticmethod
    def _mismatched_keys(datasets: Sequence[LiberoHDF5Data]) -> List[str]:
        def leaves(space: Space, prefix: str = "") -> Dict[str, Space]:
            if isinstance(space, DictSpace):
                out: Dict[str, Space] = {}
                for key, sub in space.spaces.items():
                    out.update(leaves(sub, f"{prefix}{key}/"))
                return out
            return {prefix.rstrip("/"): space}

        reference = leaves(datasets[0].single_space)
        mismatched = set()
        for dataset in datasets[1:]:
            other = leaves(dataset.single_space)
            for key in set(reference) | set(other):
                if key not in reference or key not in other or reference[key] != other[key]:
                    mismatched.add(key)
        return sorted(mismatched)

    @property
    def language_instructions(self) -> List[str]:
        """The instruction of every file, indexed by ``batch_index``."""
        return [dataset.language_instruction for dataset in self.batches]

    def num_trajectories(self) -> int:
        return int(sum(dataset.num_trajectories() for dataset in self.batches))

    def close(self) -> None:
        # Unlike a plain CombinedBatch view, the suite opened its files and owns them
        super().close()
        for dataset in getattr(self, "batches", []):
            dataset.close()
