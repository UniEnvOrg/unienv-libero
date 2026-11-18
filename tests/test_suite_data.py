"""LiberoSuiteData and LiberoHDF5Data's indexing on small synthetic files with LIBERO's layout."""

import h5py
import numpy as np
import pytest

from unienv_interface.space.space_utils import batch_utils as sbu
from unienv_interface.space.space_utils.metadata_utils import validate_metadata
from unienv_libero import LiberoHDF5Data, LiberoSuiteData

H = W = 4


def _write_task(path, task_value, n_demos=3, lengths=None, state_dim=10):
    """Demo i has (i + 2) steps unless ``lengths``; actions encode (task, demo, step) for checking reads."""
    lengths = lengths or [i + 2 for i in range(n_demos)]
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["macros_image_convention"] = "opengl"
        for demo, T in enumerate(lengths):
            g = data.create_group(f"demo_{demo}")
            steps = np.arange(T)
            g.create_dataset("actions", data=np.stack([np.full(T, task_value), np.full(T, demo), steps, *np.zeros((4, T))], -1).astype(np.float64))
            g.create_dataset("states", data=np.zeros((T, state_dim)))
            g.create_dataset("robot_states", data=np.zeros((T, 9)))
            g.create_dataset("rewards", data=np.zeros(T, np.uint8))
            g.create_dataset("dones", data=np.zeros(T, np.uint8))
            obs = g.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.full((T, H, W, 3), task_value, np.uint8))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((T, H, W, 3), np.uint8))
            obs.create_dataset("ee_pos", data=np.zeros((T, 3)))
            obs.create_dataset("joint_states", data=np.zeros((T, 7)))
    return str(path)


@pytest.fixture
def suite_dir(tmp_path):
    _write_task(tmp_path / "pick_up_the_bowl_demo.hdf5", 1)
    _write_task(tmp_path / "put_the_cup_on_the_plate_demo.hdf5", 2, n_demos=2)
    return tmp_path


def _expected(action):
    return int(action[0]), int(action[1]), int(action[2])  # task, demo, step


def test_demos_are_in_numeric_order(tmp_path):
    path = _write_task(tmp_path / "t_demo.hdf5", 1, lengths=[2] * 12)
    data = LiberoHDF5Data(path)
    try:
        assert data.episode_names[:3] == ["demo_0", "demo_1", "demo_2"] and data.episode_names[-1] == "demo_11"
        episode, md = data.get_episode_at(10)
        assert _expected(episode["actions"][0])[1] == 10
        np.testing.assert_array_equal(md["episode_id"], [10, 10])
    finally:
        data.close()


def test_unsorted_and_repeated_indices(tmp_path):
    data = LiberoHDF5Data(_write_task(tmp_path / "t_demo.hdf5", 1))  # demos of 2, 3, 4 steps
    try:
        idx = np.array([8, 2, 8, 0, 3, -1])
        batch, md = data.get_at_with_metadata(idx)
        flat = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
        assert [_expected(a)[1:] for a in batch["actions"]] == [flat[i] for i in idx]
        assert list(md["language_instruction"]) == ["t"] * len(idx)
        assert sbu.batch_space(data.single_metadata_space, len(idx)).contains(md)
        single, single_md = data.get_at_with_metadata(4)
        assert _expected(single["actions"])[1:] == (1, 2) and single_md["language_instruction"] == "t"
    finally:
        data.close()


def test_suite_combines_files_with_unique_episode_ids(suite_dir):
    suite = LiberoSuiteData(suite_dir)
    try:
        validate_metadata(suite.single_metadata_space)
        assert [p.rsplit("/", 1)[-1] for p in suite.files] == ["pick_up_the_bowl_demo.hdf5", "put_the_cup_on_the_plate_demo.hdf5"]
        assert suite.language_instructions == ["pick up the bowl", "put the cup on the plate"]
        assert len(suite) == (2 + 3 + 4) + (2 + 3) and suite.num_trajectories() == 5
        np.testing.assert_array_equal(suite.episode_offsets, [0, 3])

        idx = np.array([13, 0, 9, 4, 13, 10])
        batch, md = suite.get_at_with_metadata(idx)
        assert sbu.batch_space(suite.single_space, len(idx)).contains(batch)
        assert sbu.batch_space(suite.single_metadata_space, len(idx)).contains(md)
        for i, action in enumerate(batch["actions"]):
            task, demo, step = _expected(action)
            assert md["batch_index"][i] == task - 1
            assert md["episode_id"][i] == suite.episode_offsets[task - 1] + demo
            assert md["step_i"][i] == step
            assert md["language_instruction"][i] == suite.language_instructions[task - 1]
            assert batch["obs"]["agentview_rgb"][i].max() == task
    finally:
        suite.close()
    assert all(dataset.hdf5_file is None for dataset in suite.batches)


def test_suite_from_a_list_of_files_and_excluded_keys(suite_dir):
    files = sorted(str(p) for p in suite_dir.glob("*.hdf5"))[::-1]
    suite = LiberoSuiteData(files, exclude_keys=["states", "obs/eye_in_hand_rgb"])
    try:
        assert suite.language_instructions[0] == "put the cup on the plate"
        assert "states" not in suite.single_space.spaces
        assert "eye_in_hand_rgb" not in suite.single_space["obs"].spaces
        batch = suite.get_at(np.arange(len(suite)))
        assert "states" not in batch and "eye_in_hand_rgb" not in batch["obs"]
    finally:
        suite.close()


def test_files_with_different_spaces_name_the_keys(tmp_path):
    _write_task(tmp_path / "a_demo.hdf5", 1, state_dim=10)
    _write_task(tmp_path / "b_demo.hdf5", 2, state_dim=14)
    with pytest.raises(ValueError, match=r"keys that differ: states"):
        LiberoSuiteData(tmp_path)
    suite = LiberoSuiteData(tmp_path, exclude_keys=["states"])
    assert len(suite) == 18
    suite.close()


def test_suite_needs_several_files_and_known_keys(tmp_path):
    with pytest.raises(ValueError, match="several files"):
        LiberoSuiteData([_write_task(tmp_path / "a_demo.hdf5", 1)])
    with pytest.raises(ValueError, match="not a directory"):
        LiberoSuiteData(str(tmp_path / "a_demo.hdf5"))
    _write_task(tmp_path / "b_demo.hdf5", 2)
    with pytest.raises(KeyError, match="nope"):
        LiberoSuiteData(tmp_path, exclude_keys=["obs/nope"])
