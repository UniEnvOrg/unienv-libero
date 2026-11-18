# Changelog

## 0.0.1a1 (unreleased)

- `FromLiberoEnv`: LIBERO tasks as a UniEnv environment with the observation layout, dtypes, image
  convention and space metadata of the demonstration datasets (`libero_env_spaces`, `libero_observation`);
  the language instruction as context; the benchmark's initial states (or any MuJoCo state);
  reproducible seeded resets; task success as `terminated`; per-suite step limits; settle steps; upright
  `render()`. LIBERO is an optional dependency (`unienv_libero[env]`).
- `LiberoHDF5Data`: LIBERO demonstration files as a UniEnv batch with space metadata; `close()`.
