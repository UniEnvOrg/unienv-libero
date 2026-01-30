from typing import Dict, Any, Optional, Union, Tuple
import os
import numpy as np
from functools import cached_property

from unienv_interface.backends.numpy import NumpyComputeBackend, NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
from unienv_interface.space import Space, BoxSpace, DictSpace
from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_data.base import BatchBase, IndexableType
from unienv_data.storages.hdf5 import HDF5Storage
from unienv_data.batches.combined_batch import convert_single_index_to_batch, convert_index_to_batch
import h5py

__all__ = [
    'LiberoHDF5Data'
]

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
                    ),
                    "episode_id": BoxSpace(
                        NumpyComputeBackend,
                        low=0,
                        high=np.inf,
                        dtype=np.int64,
                        shape=(),
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