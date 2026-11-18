# unienv-libero

[LIBERO](https://libero-project.github.io/) for [UniEnv](https://github.com/UniEnvOrg/UniEnv): the
demonstration datasets and the simulated tasks, with one observation layout.

- `LiberoHDF5Data` reads a LIBERO demonstration HDF5 file (one task) as a UniEnv batch.
- `FromLiberoEnv` runs a task of a LIBERO suite as a UniEnv environment whose observations use the
  dataset's keys, dtypes, image convention and space metadata, so a policy trained on the data runs in
  the environment unchanged.

```bash
pip install unienv_libero          # datasets
pip install "unienv_libero[env]"   # + the simulated tasks (LIBERO, robosuite 1.4, MuJoCo < 3.10)
```

## Environment

```python
from unienv_libero import FromLiberoEnv

env = FromLiberoEnv("libero_spatial", task_id=0, settle_steps=10)
instruction, obs, info = env.reset(seed=0)        # context: the task's language instruction
obs, reward, terminated, truncated, info = env.step(env.sample_action())
```

- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.
- Episodes start from the benchmark's fixed initial states: `reset(init_state_id=i)`, `reset(init_state=...)`
  (any flattened MuJoCo state, e.g. a demonstration's `init_state` attribute), or an index drawn from the
  seeded RNG. `reset(seed=s)` is reproducible.
- `terminated` is task success (`info["success"]`); `truncated` after `max_episode_steps` (by default the
  suite's limit used by the OpenVLA / LeRobot evaluations: 220 / 280 / 300 / 520 / 400 steps).
- `settle_steps` runs no-op steps after each reset so objects settle (those evaluations use 10).
- Actions are LIBERO's normalized OSC_POSE deltas in [-1, 1]^7 (gripper -1 open, 1 close).
- `render()` returns the upright agentview image.

Observations (`image_size=128` and no depth by default, as in the datasets):

| Key | | From robosuite |
|---|---|---|
| `agentview_rgb`, `eye_in_hand_rgb` | uint8 (H, W, 3) | `agentview_image`, `robot0_eye_in_hand_image` |
| `ee_pos`, `ee_ori`, `ee_states` | float64 (3,), (3,), (6,) | `robot0_eef_pos`, axis-angle of `robot0_eef_quat` |
| `gripper_states`, `joint_states` | float64 (2,), (7,) | `robot0_gripper_qpos`, `robot0_joint_pos` |
| `{agentview,eye_in_hand}_depth` (`use_depth=True`) | float32 (H, W, 1) | `*_depth` |

Images keep robosuite's `"opengl"` convention, like the datasets: they are upside down, which the space
metadata records (`flipped="vertical"`).

On first use LIBERO downloads its scene assets from the Hugging Face Hub (`~/.cache/libero/assets`).
`FromLiberoEnv` writes LIBERO's default `~/.libero/config.yaml` (or `$LIBERO_CONFIG_PATH`) if it is missing,
so importing LIBERO does not stop to ask for a dataset folder. Headless rendering uses EGL: set
`MUJOCO_GL=egl`.

Installing LIBERO builds `egl_probe`, which needs CMake; with CMake 4 set
`CMAKE_POLICY_VERSION_MINIMUM=3.5` for the build.

## Datasets

```python
from unienv_libero import LiberoHDF5Data, LiberoSuiteData

task = LiberoHDF5Data("libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
suite = LiberoSuiteData("libero_spatial/")                    # every task file of a directory (or a list of files)
batch, metadata = suite.get_at_with_metadata([5, 4021, 77])
metadata["language_instruction"], metadata["episode_id"], metadata["batch_index"]
```

- `LiberoHDF5Data` reads one demonstration file (one task); demonstrations are in numeric order
  (`episode_id` *i* is `demo_<i>`). `LiberoSuiteData` combines files, one after the other.
- Every sample's metadata holds its `step_i`, `episode_id` (unique across the files of a `LiberoSuiteData`;
  `episode_offsets` gives each file's first id), its task's `language_instruction` and, for a suite, the
  `batch_index` of its file (see `files` / `language_instructions`).
- `exclude_keys=["states", "obs/eye_in_hand_rgb"]` leaves keys out of the space; they are not read. The files
  of a suite must share their space: the flattened simulator `states` has scene-dependent sizes across suites
  and within `libero_90`, so exclude it there (the error names the keys that differ).
- Any index array works (unsorted, repeated, negative).

The space is inferred from the file and annotated with UniEnv space metadata (`libero_space_metadata`),
following LIBERO's `scripts/create_dataset.py` layout: cameras, end-effector / joint / gripper states and
`actions` as normalized OSC_POSE deltas. `FromLiberoEnv(...).demo_file_name` names a task's file in the
LIBERO datasets (e.g. the `yifengzhu-hf/LIBERO-datasets` dataset on the Hugging Face Hub).

## Tests

`pytest tests` runs without LIBERO. With `unienv_libero[env]` installed the simulator tests run too; set
`LIBERO_DEMO_FILE` to the libero_spatial task 0 demonstration file to also replay recorded demonstrations.

## License

Apache-2.0. LIBERO (MIT), robosuite and MuJoCo are installed separately and keep their own licenses.
