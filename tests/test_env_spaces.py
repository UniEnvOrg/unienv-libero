"""FromLiberoEnv's spaces and observation mapping, without LIBERO / MuJoCo."""

import os

import h5py
import numpy as np
import pytest

from unienv_interface.space.space_utils.metadata_utils import spaces_equal_with_metadata, validate_metadata
from unienv_libero import LiberoHDF5Data, libero_env_spaces, libero_observation
from unienv_libero import env as libero_env

from test_space_metadata import H, W, _write_demo


@pytest.mark.parametrize("use_depth", [False, True], ids=["rgb", "rgbd"])
def test_env_observation_space_equals_the_dataset_space(tmp_path, use_depth):
    path = tmp_path / "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["macros_image_convention"] = "opengl"
        _write_demo(data.create_group("demo_0"), use_depth=use_depth)
    dataset = LiberoHDF5Data(path)
    try:
        obs_space, action_space = libero_env_spaces((H, W), use_depth=use_depth, image_convention="opengl")
        assert spaces_equal_with_metadata(obs_space, dataset.single_space["obs"])
        validate_metadata(obs_space)
        validate_metadata(action_space)
        assert action_space.shape == (7,) and action_space.metadata == dataset.single_space["actions"].metadata
        assert float(action_space.low.min()) == -1.0 and float(action_space.high.max()) == 1.0
    finally:
        dataset.close()


def test_opencv_convention_is_not_flagged_flipped():
    obs_space, _ = libero_env_spaces((8, 8), image_convention="opencv")
    assert "flipped" not in obs_space["agentview_rgb"].metadata


def _raw(h=8, w=8, depth=False):
    rng = np.random.default_rng(0)
    # 90 degrees about z, as robosuite's (x, y, z, w) quaternion
    quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    raw = {
        "agentview_image": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, -0.2, 0.9]),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
        "robot0_joint_pos": np.linspace(-1, 1, 7),
        "object-state": np.zeros(10),
    }
    if depth:
        raw["agentview_depth"] = rng.random((h, w, 1), dtype=np.float32)
        raw["robot0_eye_in_hand_depth"] = rng.random((h, w), dtype=np.float32)  # 2-D is accepted too
    return raw


@pytest.mark.parametrize("use_depth", [False, True], ids=["rgb", "rgbd"])
def test_observation_mapping(use_depth):
    raw = _raw(depth=use_depth)
    obs = libero_observation(raw, use_depth=use_depth)
    obs_space, _ = libero_env_spaces((8, 8), use_depth=use_depth)
    assert obs_space.contains(obs)
    assert set(obs) == set(obs_space.spaces)
    np.testing.assert_array_equal(obs["agentview_rgb"], raw["agentview_image"])  # convention kept, no flip
    np.testing.assert_allclose(obs["ee_ori"], [0.0, 0.0, np.pi / 2])
    np.testing.assert_array_equal(obs["ee_states"], np.concatenate([obs["ee_pos"], obs["ee_ori"]]))
    np.testing.assert_array_equal(obs["joint_states"], raw["robot0_joint_pos"])
    if use_depth:
        assert obs["eye_in_hand_depth"].shape == (8, 8, 1)


def test_identity_quaternion_maps_to_zero_rotation():
    raw = _raw()
    raw["robot0_eef_quat"] = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(libero_observation(raw)["ee_ori"], np.zeros(3))


def test_libero_config_is_written_without_prompting(tmp_path, monkeypatch):
    package = tmp_path / "site" / "libero"
    (package / "libero").mkdir(parents=True)
    spec = type("Spec", (), {"submodule_search_locations": [str(package)]})()
    monkeypatch.setattr(libero_env.importlib.util, "find_spec", lambda name: spec if name == "libero" else None)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "config"))
    libero_env._ensure_libero_config()
    lines = dict(line.split(": ", 1) for line in (tmp_path / "config" / "config.yaml").read_text().splitlines())
    assert lines["bddl_files"] == os.path.join(package, "libero", "bddl_files")
    assert lines["init_states"] == os.path.join(package, "libero", "init_files")
    (tmp_path / "config" / "config.yaml").write_text("benchmark_root: custom\n")
    libero_env._ensure_libero_config()  # an existing config is left alone
    assert (tmp_path / "config" / "config.yaml").read_text() == "benchmark_root: custom\n"


def test_missing_libero_raises_an_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(libero_env.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "config"))
    with pytest.raises(ImportError, match=r"unienv_libero\[env\]"):
        libero_env._ensure_libero_config()
