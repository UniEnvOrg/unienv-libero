"""LIBERO space metadata on a small synthetic HDF5 file with the real key layout.

The layout follows LIBERO's ``scripts/create_dataset.py``: ``data/demo_<i>``
with ``actions``, ``states``, ``robot_states``, ``rewards``, ``dones`` and an
``obs`` group, and ``data.attrs["macros_image_convention"]``.
"""

import h5py
import numpy as np
import pytest

from unienv_interface.space.space_utils.metadata_utils import validate_metadata
from unienv_libero import LiberoHDF5Data

T, H, W = 5, 16, 16


def _write_demo(group, *, use_depth):
    group.create_dataset("actions", data=np.zeros((T, 7), dtype=np.float64))
    group.create_dataset("states", data=np.zeros((T, 47), dtype=np.float64))
    group.create_dataset("robot_states", data=np.zeros((T, 9), dtype=np.float64))
    group.create_dataset("rewards", data=np.zeros((T,), dtype=np.uint8))
    group.create_dataset("dones", data=np.zeros((T,), dtype=np.uint8))
    obs = group.create_group("obs")
    obs.create_dataset("agentview_rgb", data=np.zeros((T, H, W, 3), dtype=np.uint8))
    obs.create_dataset("eye_in_hand_rgb", data=np.zeros((T, H, W, 3), dtype=np.uint8))
    if use_depth:
        obs.create_dataset("agentview_depth", data=np.zeros((T, H, W, 1), dtype=np.float32))
        obs.create_dataset("eye_in_hand_depth", data=np.zeros((T, H, W, 1), dtype=np.float32))
    obs.create_dataset("gripper_states", data=np.zeros((T, 2), dtype=np.float64))
    obs.create_dataset("joint_states", data=np.zeros((T, 7), dtype=np.float64))
    obs.create_dataset("ee_states", data=np.zeros((T, 6), dtype=np.float64))
    obs.create_dataset("ee_pos", data=np.zeros((T, 3), dtype=np.float64))
    obs.create_dataset("ee_ori", data=np.zeros((T, 3), dtype=np.float64))


@pytest.fixture(params=[False, True], ids=["rgb", "rgbd"])
def libero_file(request, tmp_path):
    path = tmp_path / "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["macros_image_convention"] = "opengl"
        for i in range(2):
            _write_demo(data.create_group(f"demo_{i}"), use_depth=request.param)
    return path, request.param


def test_libero_spaces_are_described(libero_file):
    path, use_depth = libero_file
    data = LiberoHDF5Data(path)
    try:
        space = data.single_space
        validate_metadata(space)
        validate_metadata(data.single_metadata_space)

        for camera, source in (("agentview", "agentview"), ("eye_in_hand", "robot0_eye_in_hand")):
            md = space["obs"][f"{camera}_rgb"].metadata
            assert (md["kind"], md["format"], md["layout"], md["source"]) == ("image", "rgb", "HWC", source)
            assert md["flipped"] == "vertical"
            assert md["image_convention"] == "opengl"
            if use_depth:
                depth = space["obs"][f"{camera}_depth"].metadata
                assert (depth["kind"], depth["layout"]) == ("depth", "HWC")
                assert "units" not in depth

        actions = space["actions"].metadata
        assert (actions["kind"], actions["control"], actions["mode"]) == ("action", "eef_pose", "delta")
        assert actions["normalized"] is True
        assert actions["rotation"] == "axis_angle"
        assert len(actions["names"]) == 7

        assert space["obs"]["ee_pos"].metadata["units"] == "m"
        assert space["obs"]["ee_ori"].metadata["rotation"] == "axis_angle"
        assert space["obs"]["joint_states"].metadata["units"] == "rad"
        assert space["obs"]["gripper_states"].metadata["kind"] == "proprio"
        assert space["robot_states"].metadata["rotation"] == "quat_xyzw"
        assert space["states"].metadata["kind"] == "privileged"
        assert "label" in data.single_metadata_space["step_i"].metadata

        # Reading data is unaffected.
        sample = data.get_at(0)
        assert sample["obs"]["agentview_rgb"].shape == (H, W, 3)
    finally:
        data.hdf5_file.close()


def test_libero_opencv_convention_is_not_flipped(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["macros_image_convention"] = "opencv"
        _write_demo(data.create_group("demo_0"), use_depth=False)
    data = LiberoHDF5Data(path)
    try:
        md = data.single_space["obs"]["agentview_rgb"].metadata
        assert "flipped" not in md and md["image_convention"] == "opencv"
    finally:
        data.hdf5_file.close()


def test_close_closes_the_hdf5_file(libero_file):
    path, _ = libero_file
    data = LiberoHDF5Data(path)
    hdf5_file = data.hdf5_file
    assert len(data) > 0 and hdf5_file.id.valid
    data.close()
    assert not hdf5_file.id.valid
    data.close()  # closing twice (and again from __del__) is fine
