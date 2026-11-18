"""FromLiberoEnv in the LIBERO simulator (skipped without LIBERO).

Set ``LIBERO_DEMO_FILE`` to a LIBERO dataset file of libero_spatial task 0
(``libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5``)
to also replay recorded demonstrations.
"""

import os

import numpy as np
import pytest

pytest.importorskip("libero")

from unienv_interface.space.space_utils.metadata_utils import spaces_equal_with_metadata, validate_metadata
from unienv_libero import FromLiberoEnv, LiberoHDF5Data


@pytest.fixture(scope="module")
def env():
    env = FromLiberoEnv("libero_spatial", 0, seed=0)
    yield env
    env.close()


def test_spaces_context_and_reset(env):
    validate_metadata(env.observation_space)
    validate_metadata(env.action_space)
    assert env.max_episode_steps == 220 and env.num_init_states > 0
    ctx, obs, info = env.reset(init_state_id=0)
    assert ctx == env.language_instruction and env.context_space.contains(ctx)
    assert env.observation_space.contains(obs)
    assert info["init_state_id"] == 0 and not info["success"]


def test_step(env):
    env.reset(init_state_id=1)
    obs, reward, terminated, truncated, info = env.step(env.sample_action())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and not truncated and info["success"] == terminated


def test_seeded_resets_are_reproducible(env):
    _, first, info_first = env.reset(seed=11)
    _, second, info_second = env.reset(seed=11)
    assert info_first["init_state_id"] == info_second["init_state_id"]
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_truncation_after_max_episode_steps():
    env = FromLiberoEnv("libero_spatial", 0, max_episode_steps=3, settle_steps=2)
    try:
        env.reset(init_state_id=0)
        flags = [env.step(np.zeros(7))[3] for _ in range(3)]
        assert flags == [False, False, True]
    finally:
        env.close()


def test_render_is_the_upright_agentview(env):
    _, obs, _ = env.reset(init_state_id=0)
    frame = env.render()
    assert frame.shape == obs["agentview_rgb"].shape
    np.testing.assert_array_equal(frame, obs["agentview_rgb"][::-1])


@pytest.mark.skipif(not os.environ.get("LIBERO_DEMO_FILE"), reason="set LIBERO_DEMO_FILE to replay demonstrations")
def test_recorded_demonstrations_replay_successfully(env):
    dataset = LiberoHDF5Data(os.environ["LIBERO_DEMO_FILE"])
    try:
        assert spaces_equal_with_metadata(env.observation_space, dataset.single_space["obs"])
        assert dataset.language_instruction == env.language_instruction
        for episode_idx in range(2):
            episode, _ = dataset.get_episode_at(episode_idx)
            init_state = dataset.episode_root[dataset.episode_names[episode_idx]].attrs["init_state"]
            _, obs, _ = env.reset(init_state=init_state)
            np.testing.assert_allclose(obs["ee_pos"], episode["obs"]["ee_pos"][0], atol=1e-2)
            success = False
            for action in episode["actions"]:
                *_, terminated, _, _ = env.step(action)
                if terminated:
                    success = True
                    break
            assert success, f"demo {episode_idx} did not succeed when replayed"
    finally:
        dataset.close()
