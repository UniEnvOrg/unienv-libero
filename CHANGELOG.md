# Changelog

## 0.0.1a2 (2026-11-18)

- `LiberoSuiteData`: several demonstration files (e.g. a suite's directory) as one batch, with episode ids
  unique across the files and the file's `batch_index`; files with differing spaces raise an error naming the
  keys (leave them out with `exclude_keys`).
- `LiberoHDF5Data`: `exclude_keys` (also nested, e.g. `"obs/eye_in_hand_rgb"`); `episode_id_offset`; every
  sample's metadata holds the task's `language_instruction`.
- **Changed:** demonstrations are in numeric order (`demo_0, demo_1, demo_2, ...`); h5py's alphabetical
  order (`demo_0, demo_1, demo_10, ...`) gave `episode_id` 2 to `demo_10`.
- Fixed: reading with an unsorted or repeated index array (e.g. random sampling) failed in h5py ("Indexing
  elements must be in increasing order").

## 0.0.1a1 (2026-11-18)

- `FromLiberoEnv`: LIBERO tasks as a UniEnv environment with the observation layout, dtypes, image
  convention and space metadata of the demonstration datasets (`libero_env_spaces`, `libero_observation`);
  the language instruction as context; the benchmark's initial states (or any MuJoCo state);
  reproducible seeded resets; task success as `terminated`; per-suite step limits; settle steps; upright
  `render()`. LIBERO is an optional dependency (`unienv_libero[env]`).
- `LiberoHDF5Data`: LIBERO demonstration files as a UniEnv batch with space metadata; `close()`.
